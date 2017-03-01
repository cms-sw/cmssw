#include "CondTools/Ecal/interface/EcalTPGPedestalsHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"

#include<iostream>
#include<fstream>


#include <time.h>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>



popcon::EcalTPGPedestalsHandler::EcalTPGPedestalsHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGPedestalsHandler")) {

	edm::LogInfo("EcalTPGPedestalsHandler") << "EcalTPGPedestals Source handler constructor";
        m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
        m_lastRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtype=ps.getParameter<std::string>("RunType");

	edm::LogInfo("EcalTPGPedestalsHandler")<< m_sid<<"/"<<m_user<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGPedestalsHandler::~EcalTPGPedestalsHandler()
{
}


void popcon::EcalTPGPedestalsHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGPedestalsHandler") << "Started getNewObjects";
	
	//check whats already inside of database
	if (tagInfo().size){
  	//check whats already inside of database
    	std::cout << "got offlineInfo = " << std::endl;
	std::cout << "tag name = " << tagInfo().name << std::endl;
	std::cout << "size = " << tagInfo().size <<  std::endl;
    	} else {
    	std::cout << " First object for this tag " << std::endl;
    	}
		 	
	unsigned int max_since =0;
	max_since=static_cast<unsigned int>(tagInfo().lastInterval.first); 
    	edm::LogInfo("EcalTPGPedestalsHandler") << "max_since = " << max_since;    
	edm::LogInfo("EcalTPGPedestalsHandler")<< "Retrieved last payload ";

        // here we retrieve all the runs after the last from online DB 
    	edm::LogInfo("EcalTPGPedestalsHandler")<< "Retrieving run list from ONLINE DB ... " << std::endl;

    	edm::LogInfo("EcalTPGPedestalsHandler") << "Making connection..." << std::flush;
    	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    	edm::LogInfo("EcalTPGPedestalsHandler") << "Done." << std::endl;
        
	if (!econn)
	{
	  std::cout << " Connection parameters " <<m_sid <<"/"<<m_user<<std::endl;
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


	readFromFile("last_tpg_ped_settings.txt");


 	unsigned int min_run=m_i_run_number+1;

	if(m_firstRun < m_i_run_number) {
	  min_run=m_i_run_number+1;
	} else {
	  min_run=m_firstRun;
	}
	if(min_run<max_since) {
	  min_run=  max_since+1; // we have to add 1 to the last transferred one
	} 

	std::cout<<"m_i_run_number"<< m_i_run_number <<"m_firstRun "<<m_firstRun<< "max_since " <<max_since<< std::endl;

	unsigned int max_run=m_lastRun;
	edm::LogInfo("EcalTPGPedestalsHandler") <<"min_run= " << min_run << " max_run = " << max_run;
	RunList my_list; 
	my_list=econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
	//	my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef); 
       
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	size_t num_runs=run_vec.size();
	
	std::cout <<"number of runs is : "<< num_runs<< std::endl;
        
	unsigned int irun=0;
		
	if(num_runs>0){ 
	
	  for(size_t kr=0; kr<run_vec.size(); kr++){
	    irun=static_cast<unsigned int>(run_vec[kr].getRunNumber());

	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" run= "<<irun<<std::endl;

	   
            // retrieve the data :
            std::map<EcalLogicID, RunTPGConfigDat> dataset;
            econn->fetchDataSet(&dataset, &run_vec[kr]);

            std::string the_config_tag="";
	    int the_config_version=0;

            std::map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;

            int nr=0;
            for ( it=dataset.begin(); it!=dataset.end(); it++ )
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

	      // now get TPGPedestals
		int pedId=fe_main_info.getPedId();
		
		if( pedId != m_i_ped ) {
		  
		  FEConfigPedInfo fe_ped_info;
		  fe_ped_info.setId(pedId);
		  econn-> fetchConfigSet(&fe_ped_info);
		  std::map<EcalLogicID, FEConfigPedDat> dataset_TpgPed;
		  econn->fetchDataSet(&dataset_TpgPed, &fe_ped_info);
		  
		  // NB new 
		  EcalTPGPedestals* peds = new EcalTPGPedestals;
		  typedef std::map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
		  EcalLogicID ecid_xt;
		  FEConfigPedDat  rd_ped;
		  int icells=0;
		  for (CIfeped p = dataset_TpgPed.begin(); p != dataset_TpgPed.end(); p++) 
		    {
		      ecid_xt = p->first;
		      rd_ped  = p->second;
		    
		      std::string ecid_name=ecid_xt.getName();
		    
		    // EB data
		    if (ecid_name=="EB_crystal_number") {
		      if(icells<10) std::cout << " copy the EB data " << " icells = " << icells << std::endl;
		      int sm_num=ecid_xt.getID1();
		      int xt_num=ecid_xt.getID2();
		      
		      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
		      EcalTPGPedestals::Item item;
		      item.mean_x1  =(unsigned int)rd_ped.getPedMeanG1() ;
		      item.mean_x6  =(unsigned int)rd_ped.getPedMeanG6();
		      item.mean_x12 =(unsigned int)rd_ped.getPedMeanG12();
		      
		      peds->insert(std::make_pair(ebdetid.rawId(),item));
		      ++icells;
		    }else if (ecid_name=="EE_crystal_number"){
		      
		      // EE data
		      int z=ecid_xt.getID1();
		      int x=ecid_xt.getID2();
		      int y=ecid_xt.getID3();
		      EEDetId eedetid(x,y,z,EEDetId::XYMODE);
		      EcalTPGPedestals::Item item;
		      item.mean_x1  =(unsigned int)rd_ped.getPedMeanG1();
		      item.mean_x6  =(unsigned int)rd_ped.getPedMeanG6();
		      item.mean_x12 =(unsigned int)rd_ped.getPedMeanG12();
		      
		      peds->insert(std::make_pair(eedetid.rawId(),item));
		      ++icells;
		    }
		  }
		
		  
		  Time_t snc= (Time_t) irun; 
		  m_to_transfer.push_back(std::make_pair((EcalTPGPedestals*)peds,snc));
		  
		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;
		  m_i_ped=pedId;
		  
		  writeFile("last_tpg_ped_settings.txt");

		} else {

		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;

		  writeFile("last_tpg_ped_settings.txt");

		  std::cout<< " even if the tag/version is not the same, the pedestals id is the same -> no transfer needed "<< std::endl; 

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
	      writeFile("last_tpg_ped_settings.txt");
	    }

	  }
	}
	

  	delete econn;
	
	edm::LogInfo("EcalTPGPedestalsHandler")  << "Ecal - > end of getNewObjects -----------";        

}



void  popcon::EcalTPGPedestalsHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_ped=0; 

  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGPedestalsHandler")<<"*** Can not open file: "<<inputFile;
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
  m_i_ped=atoi(line);
  str << "ped_config= " << m_i_ped << std::endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGPedestalsHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  std::ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<std::endl;
  myfile << m_i_version <<std::endl;
  myfile << m_i_run_number <<std::endl;
  myfile << m_i_ped <<std::endl;

  myfile.close();

}


