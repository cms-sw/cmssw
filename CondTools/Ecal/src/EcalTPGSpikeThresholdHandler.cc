#include "CondTools/Ecal/interface/EcalTPGSpikeThresholdHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSpikeInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSpikeDat.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"



#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"



#include<iostream>
#include<fstream>

#include <time.h>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGSpikeThresholdHandler::EcalTPGSpikeThresholdHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGSpikeThresholdHandler")) {

	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "EcalTPGSpike Source handler constructor";
  	m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
  	m_lastRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
  	m_sid= ps.getParameter<std::string>("OnlineDBSID");
  	m_user= ps.getParameter<std::string>("OnlineDBUser");
  	m_pass= ps.getParameter<std::string>("OnlineDBPassword");
  	m_locationsource= ps.getParameter<std::string>("LocationSource");
 	m_location=ps.getParameter<std::string>("Location");
  	m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtype=ps.getParameter<std::string>("RunType");

  	edm::LogInfo("EcalTPGSpikeThresholdHandler") << m_sid<<"/"<<m_user<<"/"<<m_location<<"/"<<m_gentag;
}

popcon::EcalTPGSpikeThresholdHandler::~EcalTPGSpikeThresholdHandler()
{
}


void popcon::EcalTPGSpikeThresholdHandler::getNewObjects()
{


	using namespace edm;
	using namespace std;

	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "Started GetNewObjects!!!";

	//check whats already inside of database
	if (tagInfo().size){
  	//check whats already inside of database
    	  std::cout << "got offlineInfo = " << std::endl;
	  std::cout << "tag name = " << tagInfo().name << std::endl;
	  std::cout << "size = " << tagInfo().size <<  std::endl;
    	} else {
    	  std::cout << " First object for this tag " << std::endl;
    	}

	unsigned int max_since=0;
	max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "max_since : "  << max_since;
	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "Done.";
	
	if (!econn)
	{
	  std::cout << " connection parameters " <<m_sid <<"/"<<m_user<<std::endl;
	  //	    cerr << e.what() << std::endl;
	  throw cms::Exception("OMDS not available");
	} 

	
	LocationDef my_locdef;
	my_locdef.setLocation(m_location); 

	RunTypeDef my_rundef;
	my_rundef.setRunType(m_runtype); 

	RunTag  my_runtag;
	my_runtag.setLocationDef( my_locdef );
	my_runtag.setRunTypeDef(  my_rundef );
	my_runtag.setGeneralTag(m_gentag); 


	readFromFile("last_tpg_spikeThreshold_settings.txt");


 	unsigned int min_run=m_i_run_number+1;

	if(m_firstRun<m_i_run_number) {
	  min_run=m_i_run_number+1;
	} else {
	  min_run=m_firstRun;
	}
	

	if(min_run<max_since) {
	  min_run=  max_since+1; // we have to add 1 to the last transferred one
	} 
	
	std::cout<<"m_i_run_number"<< m_i_run_number <<"m_firstRun "<<m_firstRun<< "max_since " <<max_since<< std::endl;

	unsigned int max_run=m_lastRun;
	edm::LogInfo("EcalTPGSpikeThresholdHandler") << "min_run= " << min_run << " max_run= " << max_run;

        RunList my_list;
	my_list=econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
        //my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	size_t num_runs=run_vec.size();
	
	std::cout <<"number of runs is : "<< num_runs<< std::endl;
        
    	std::string str="";
        
	unsigned int irun;
	if(num_runs>0){
	    
	  for(size_t kr=0; kr<run_vec.size(); kr++){

	    irun=static_cast<unsigned int>(run_vec[kr].getRunNumber());

	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" run= "<<irun<<std::endl;

	    // retrieve the data :
	    std::map<EcalLogicID, RunTPGConfigDat> dataset;
	    econn->fetchDataSet(&dataset, &run_vec[kr]);
	    
	    std::string the_config_tag="";
	    int the_config_version=0;
	    
	    std::map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;
	    
	    int nr=0;        
	    for( it=dataset.begin(); it!=dataset.end(); it++ )
	    {
	      ++nr;
	      //EcalLogicID ecalid  = it->first;
	    
	      RunTPGConfigDat  dat = it->second;
	      the_config_tag=dat.getConfigTag();
	      the_config_version=dat.getVersion();
	    }
	    
	      
	    // it is all the same for all SM... get the last one 


	    std::cout<<" run= "<<irun<<" tag "<<the_config_tag<<" version="<<the_config_version <<std::endl;

	    // here we should check if it is the same as previous run.


	    if((the_config_tag != m_i_tag || the_config_version != m_i_version ) && nr>0 ) {
	      std::cout<<"the tag is different from last transferred run ... retrieving last config set from DB"<<std::endl;

	      FEConfigMainInfo fe_main_info;
	      fe_main_info.setConfigTag(the_config_tag);
	      fe_main_info.setVersion(the_config_version);

	      try{ 
		std::cout << " before fetch config set" << std::endl;	    
		econn-> fetchConfigSet(&fe_main_info);
		std::cout << " after fetch config set" << std::endl;	
		
		
            	// now get TPGSpikeThreshold
            	int spikeId=fe_main_info.getSpiId();
	    
	    	if( spikeId != m_i_spikeTh ) {
	    
	    	  FEConfigSpikeInfo fe_spike_info;
	    	  fe_spike_info.setId(spikeId);
	    	  econn-> fetchConfigSet(&fe_spike_info);
	    	  std::map<EcalLogicID, FEConfigSpikeDat> dataset_TpgSpike;
	    	  econn->fetchDataSet(&dataset_TpgSpike, &fe_spike_info);

            	  EcalTPGSpike *lut=new EcalTPGSpike();
            	  typedef std::map<EcalLogicID, FEConfigSpikeDat>::const_iterator CIfelut;
	    	  EcalLogicID ecid_xt;
	    	  FEConfigSpikeDat  rd_spike;
	    	  int itowers=0;
	
	    	  for (CIfelut p = dataset_TpgSpike.begin(); p != dataset_TpgSpike.end(); p++) 
	    	  {
	      	    ecid_xt = p->first;
	            rd_spike  = p->second;
	  
	      	    std::string ecid_name=ecid_xt.getName();
	      
	      	    if(ecid_name=="EB_trigger_tower") {
	      	      // SM number
              	      int smid=ecid_xt.getID1();
              	      // TT number
	      	      int towerid=ecid_xt.getID2();


		      int tow_eta=(towerid-1)/4;
		      int tow_phi=((towerid-1)-tow_eta*4);

		      int axt=(tow_eta*5)*20 + tow_phi*5 +1 ;

		      EBDetId id(smid, axt, EBDetId::SMCRYSTALMODE ) ;
		      const EcalTrigTowerDetId towid= id.tower();
	  
                      //std::cout << " TTid =  " << towid.rawId() << " SpikeTh = " <<rd_spike.getSpikeThreshold() << std::endl;
		       
	      	      lut->setValue(towid.rawId(), rd_spike.getSpikeThreshold());
	      	      ++itowers;
	    	    }
		    ++itowers;
		  }
	
                  std::cout <<" The number of towers is = " << itowers << std::endl;
		   
		  Time_t snc= (Time_t) irun;
	      	      
		  m_to_transfer.push_back(std::make_pair((EcalTPGSpike*)lut,snc));
	
	          m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;
		  m_i_spikeTh=spikeId;
		  
		  writeFile("last_tpg_spikeThreshold_settings.txt");

		} else {

		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;

		  writeFile("last_tpg_spikeThreshold_settings.txt");

		  std::cout<< " even if the tag/version is not the same, the lutGroup id is the same -> no transfer needed "<< std::endl; 

		}

	      }       
	      
	      catch (std::exception &e) { 
		std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" <<the_config_tag
			  <<" version="<<the_config_version<< std::endl;
		std::cout << e.what() << std::endl;
		m_i_run_number=irun;

	      }
	      std::cout<<" **************** "<<std::endl;
	      
	    } else if(nr==0) {
	      m_i_run_number=irun;
	      std::cout<< " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed "<< std::endl; 
	      std::cout<<" **************** "<<std::endl;
	    } else {
	      m_i_run_number=irun;
	      m_i_tag=the_config_tag;
	      m_i_version=the_config_version;
	      std::cout<< " the tag/version is the same -> no transfer needed "<< std::endl; 
	      std::cout<<" **************** "<<std::endl;
	      writeFile("last_tpg_spikeThreshold_settings.txt");
	    }
       }
      }
	  
      delete econn;
      
      edm::LogInfo("EcalTPGSpikeThresholdHandler") << "Ecal - > end of getNewObjects -----------";
}

void  popcon::EcalTPGSpikeThresholdHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_spikeTh=0; 

  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGSpikeThresholdHandler")<<"*** Can not open file: "<<inputFile;
    return;
  }

  char line[256];
    
  std::ostringstream str;

  fgets(line,255,inpFile);
  m_i_tag=to_string(line);
  str << "gen tag " << m_i_tag << std::endl ;  // should I use this? 

  fgets(line,255,inpFile);
  m_i_version=atoi(line);
  str << "version= " << m_i_version << std::endl ;  

  fgets(line,255,inpFile);
  m_i_run_number=atoi(line);
  str << "run_number= " << m_i_run_number << std::endl ;  

  fgets(line,255,inpFile);
  m_i_spikeTh=atoi(line);
  str << "spikeTh_config= " << m_i_spikeTh << std::endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGSpikeThresholdHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  std::ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<std::endl;
  myfile << m_i_version <<std::endl;
  myfile << m_i_run_number <<std::endl;
  myfile << m_i_spikeTh <<std::endl;

  myfile.close();

}
