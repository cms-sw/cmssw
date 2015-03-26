#include "CondTools/Ecal/interface/EcalIntercalibHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"


#include<iostream>

popcon::EcalIntercalibHandler::EcalIntercalibHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalIntercalibHandler")) {

	std::cout << "EcalIntercalib Source handler constructor\n" << std::endl;
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
	m_value_highfield= ps.getUntrackedParameter< double >("Value_Bon");
	//	m_value_highfield= 0.75585;



        std::cout << m_sid<<"/"<<m_user<<"/"<<m_location<<"/"<<m_gentag   << std::endl;


}

popcon::EcalIntercalibHandler::~EcalIntercalibHandler()
{
}


void popcon::EcalIntercalibHandler::getNewObjects()
{

  std::cout << "------- Ecal - > getNewObjects\n";

  std::ostringstream ss; 
  ss<<"ECAL ";

  unsigned int max_since=0;
  max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
  std::cout << "max_since : "  << max_since << std::endl;
  bool something_to_transfer = false;
  bool magnet_high = true; 
  if(tagInfo().size) {
	Ref ped_db = lastPayload();
	
	// we parse the last record in the DB and check if it is low or high field 

	std::cout << "retrieved last payload "  << std::endl;


	EcalIntercalibConstant the_cal = 0. ; // relies on it being a float.
	                                      // instead should perhaps
	                                      // protect the next if when
                                              // the EEDetId isn't valid?

	int iX=50;
	int iY=5;
	int iZ=-1;


	float the_value_high=(float)m_value_highfield;
	std::cout << "The value for high field at EE x/y/z= 50/5/-1 is " << the_value_high << std::endl;

	if (EEDetId::validDetId(iX,iY,iZ))
	  {
	    EEDetId eedetidpos(iX,iY,iZ);
	    
	    EcalIntercalibConstants::const_iterator it =ped_db->find(eedetidpos.rawId());
	    

	    the_cal = (*it);

	  }
	
	if(the_cal!= the_value_high) magnet_high=false;
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

    if(something_to_transfer){
	    
	    std::cout << "Generating popcon record for run " << irun << "..." << std::flush;
	    std::cout << "going to open file "<<file_ << std::flush;
	    
	    
	    EcalCondHeader   header;
	    EcalIntercalibConstants * payload = new EcalIntercalibConstants;
	    EcalFloatCondObjectContainerXMLTranslator::readXML(file_,header,*payload);
	    
	    
	    Time_t snc= (Time_t) irun ;
	    
	    popcon::PopConSourceHandler<EcalIntercalibConstants>::m_to_transfer.push_back(
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
