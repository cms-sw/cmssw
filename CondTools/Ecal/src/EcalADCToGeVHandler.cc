#include "CondTools/Ecal/interface/EcalADCToGeVHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"


#include<iostream>

popcon::EcalADCToGeVHandler::EcalADCToGeVHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalADCToGeVHandler")) {

	std::cout << "EcalADCToGeV Source handler constructor\n" << std::endl;
        m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
        m_lastRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_file_lowfield= ps.getParameter<std::string>("FileLowField");
        m_file_highfield= ps.getParameter<std::string>("FileHighField");



        std::cout << m_sid<<"/"<<m_user<<"/"<<m_location<<"/"<<m_gentag   << std::endl;


}

popcon::EcalADCToGeVHandler::~EcalADCToGeVHandler()
{
}


void popcon::EcalADCToGeVHandler::getNewObjects()
{

	std::cout << "------- Ecal - > getNewObjects\n";

  std::ostringstream ss; 
  ss<<"ECAL ";

	unsigned int max_since=0;
	max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
	std::cout << "max_since : "  << max_since << std::endl;
  bool magnet_high = false; 
  bool something_to_transfer = false;
  if(tagInfo().size) {
	Ref ped_db = lastPayload();
	
	// we parse the last record in the DB and check if it is low or high field 

	std::cout << "retrieved last payload "  << std::endl;


	EcalADCToGeVConstant the_cal ;

	//unused	float adc_eb=ped_db->getEBValue();
	float adc_ee=ped_db->getEEValue();
 
	//	float the_value_high_eb=0.03894;
	//      float the_value_high_ee=0.06285;
	// float the_value_high_eb=0.03894;
	// float the_value_high_ee=0.06378;

	
	// bool magnet_high=true; 
	// if(adc_eb!= the_value_high_eb || adc_ee!= the_value_high_ee ) magnet_high=false; 


	//unused	float the_value_low_eb=0.03894;
	float the_value_low_ee=0.05678;
	if( adc_ee!= the_value_low_ee ) magnet_high=true; 
  }  // check if there is already a payload
  else something_to_transfer = true;

  // here we connect to the online DB to check the value of the magnetic field 
	std::cout << "Connecting to ONLINE DB ... " << std::endl;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	std::cout << "Connection done" << std::endl;
	
	if (!econn)
	  {
	    std::cout << " Problem with OMDS: connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<std::endl;
	    throw cms::Exception("OMDS not available");
	  } 


	std::cout << "Retrieving last run from ONLINE DB ... " << std::endl;
	std::map<EcalLogicID, RunDat> rundat;
	RunIOV rp ;
	run_t runmax=10000000;
	std::string location_p5="P5_Co";
	econn->fetchValidDataSet(&rundat , &rp, location_p5 ,runmax);
	
	unsigned long long  irun=(unsigned long long) rp.getRunNumber();

	std::cout<< "retrieved run number "<< irun <<std::endl;  

  if(irun>max_since) {
	  // retrieve from last value data record 
	  // always call this method at first run
	  
	  std::map<EcalLogicID, RunDCSMagnetDat> dataset;
	  
	  econn->fetchDataSet(&dataset, &rp);
	  
	  if (!dataset.size()) {
	    throw(std::runtime_error("Zero rows read back"));
	  } else {
	    std::cout<< "retrieved magnet current"<<std::endl;  
	  }
	  
	  
	  float mag_cur=0;
	  
	  std::map<  EcalLogicID, RunDCSMagnetDat >::iterator it;
	  for (it=dataset.begin(); it!=dataset.end(); ++it){
	    
	    RunDCSMagnetDat  a_mag = (*it).second;
	    mag_cur= a_mag.getMagnetCurrent();
	    
	  }
	  
	  
	  std::string file_=m_file_highfield;
    if(tagInfo().size) {
	  if(mag_cur>7000. && magnet_high ) {
	    
	    std::cout << " the magnet is ON and the constants are for magnet ON " << std::endl; 
	    
	  } else if(mag_cur>7000. && !magnet_high ) {
	    something_to_transfer=true;
	    std::cout << " the magnet is ON and the constants are for magnet OFF " << std::endl; 
	    std::cout << " I transfer the ON constants "<< std::endl; 
	    file_=m_file_highfield;
	    
	  } else if(mag_cur<6000. && magnet_high ) {
	    something_to_transfer=true;
	    std::cout << " the magnet is OFF and the constants are for magnet ON "<< std::endl;
	    std::cout << " I transfer the OFF constants "<< std::endl;
	    file_=m_file_lowfield;
	    
	  } else if( mag_cur<6000. && !magnet_high ){
	    
	    std::cout << " the magnet is OFF and the constants are for magnet OFF "<< std::endl;
	    file_=m_file_lowfield;
	    
	  } else {
	    
	    std::cout << " the magnet is in a strange situation I do nothing ... just be patient "<< std::endl;
	    
	  }
    }
    else {
      if(mag_cur>7000.)
	std::cout <<" first payload, the magnet is ON " << std::endl;
      else if( mag_cur<6000.) {
	std::cout <<" first payload, the magnet is OFF " << std::endl;
	file_=m_file_lowfield;
      }
      else
	std::cout << " the magnet is in a strange situation I do nothing ... just be patient "<< std::endl;
    }
	  
    if(something_to_transfer) {
	    
	    std::cout << "Generating popcon record for run " << irun << "..." << std::flush;
	    std::cout << "going to open file "<<file_ << std::flush;
	    
	    
	    EcalCondHeader   header;
	    EcalADCToGeVConstant * payload = new EcalADCToGeVConstant;
	    EcalADCToGeVXMLTranslator::readXML(file_,header,*payload);
	    
	    
	    Time_t snc= (Time_t) irun ;
	    
	    popcon::PopConSourceHandler<EcalADCToGeVConstant>::m_to_transfer.push_back(
											  std::make_pair(payload,snc));
	    
	    ss << "Run=" << irun << "_Magnet_changed_"<<std::endl; 
	    m_userTextLog = ss.str()+";";
	    
    } else {
	    std::cout << "Run " << irun << " nothing sent to the DB"<< std::endl;
	  
	    ss<< "Run=" << irun << "_Magnet_NOT_changed_"<<std::endl; 
	    m_userTextLog = ss.str()+";";
    }
	
    delete econn;
  }  // irun > max_since
  else {
	    std::cout << "Run " << irun << " nothing sent to the DB"<< std::endl;
	    ss<< "Run=" << irun << "_no_new_runs_"<<std::endl; 
	    m_userTextLog = ss.str()+";";


  }

	std::cout << "Ecal - > end of getNewObjects -----------\n";
}

