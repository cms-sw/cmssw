#include "CondTools/Ecal/interface/EcalTPGBadTTHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadTTInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

#include<iostream>
#include<fstream>


#include <time.h>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGBadTTHandler::EcalTPGBadTTHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGBadTTHandler")) {

        edm::LogInfo("EcalTPGBadTTHandler") << "EcalTPGTowerStatus Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtype=ps.getParameter<std::string>("RunType");

        edm::LogInfo("EcalTPGBadTTHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGBadTTHandler::~EcalTPGBadTTHandler()
{
}


void popcon::EcalTPGBadTTHandler::getNewObjects()
{
    	edm::LogInfo("EcalTPGBadTTHandler") << "Started GetNewObjects!!!";

    	int max_since=0;
    	max_since=(int)tagInfo().lastInterval.first;
    	edm::LogInfo("EcalTPGBadTTHandler") << "max_since : "  << max_since;
    	edm::LogInfo("EcalTPGBadTTHandler") << "retrieved last payload ";

    	// here we retrieve all the runs after the last from online DB 
    	edm::LogInfo("EcalTPGBadTTHandler") << "Retrieving run list from ONLINE DB ... ";

    	edm::LogInfo("EcalTPGBadTTHandler") << "Making connection...";
    	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    	edm::LogInfo("EcalTPGBadTTHandler") << "Done.";
        
    	if (!econn)
    	{
      	  cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
      	  throw cms::Exception("OMDS not available");
    	} 
      
    	LocationDef my_locdef;
    	my_locdef.setLocation(m_location); 

    	RunTypeDef my_rundef;
    	my_rundef.setRunType(m_runtype); 

	RunTag  my_runtag;
	my_runtag.setLocationDef( my_locdef );
	my_runtag.setRunTypeDef( my_rundef );
	my_runtag.setGeneralTag(m_gentag); 

	readFromFile("last_tpg_badTT_settings.txt");


 	int min_run=m_i_run_number+1;

	if(m_firstRun<(unsigned int)m_i_run_number) {
	  min_run=(int) m_i_run_number+1;
	} else {
	  min_run=(int)m_firstRun;
	}
	if(min_run<max_since) {
	  min_run=  max_since+1; // we have to add 1 to the last transferred one
	} 

	std::cout<<"m_i_run_number"<< m_i_run_number <<"m_firstRun "<<m_firstRun<< "max_since " <<max_since<< endl;

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGBadTTHandler") << "min_run= " << min_run << "max_run= " << max_run;

	RunList my_list; 
	my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef); 
	      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int num_runs=run_vec.size();

	std::cout <<"number of runs is : "<< num_runs<< endl;
       
    	std::string str="";
    
	unsigned long irun=0;
	if(num_runs>0){
	 for(int kr=0; kr<(int)run_vec.size(); kr++){
	  
	    irun=(unsigned long) run_vec[kr].getRunNumber();

            // retrieve the data :
            map<EcalLogicID, RunTPGConfigDat> dataset;
            econn->fetchDataSet(&dataset, &run_vec[kr]);
            
	    std::string the_config_tag="";
            int the_config_version=0;
	    
	    map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;
	
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


	    // here we should check if it is the same as previous run.


	    if((the_config_tag != m_i_tag || the_config_version != m_i_version ) && nr>0 ) {
	      std::cout<<" run= "<<irun<<" tag "<<the_config_tag<<" version="<<the_config_version <<std::endl;
	      std::cout<<"the tag is different from last transferred run ... retrieving last config set from DB"<<endl;

	      FEConfigMainInfo fe_main_info;
	      fe_main_info.setConfigTag(the_config_tag);
	      fe_main_info.setVersion(the_config_version);

	      try{ 

		econn-> fetchConfigSet(&fe_main_info);
	
            	// now get TPGTowerStatus
           	int badttId=fe_main_info.getBttId();
	
	   	if( badttId != m_i_badTT ) {
	
	     	  FEConfigBadTTInfo fe_badTT_info;
	     	  fe_badTT_info.setId(badttId);
	     
	     	  econn-> fetchConfigSet(&fe_badTT_info);
	     
	     	  std::vector< FEConfigBadTTDat > dataset_TpgBadTT;

	     	  econn->fetchConfigDataSet(&dataset_TpgBadTT, &fe_badTT_info);
	     
	     	  EcalTPGTowerStatus* towerStatus = new EcalTPGTowerStatus;
	     	  typedef std::vector<FEConfigBadTTDat>::const_iterator CIfeped;
	     	  EcalLogicID ecid_xt;
	     	  FEConfigBadTTDat  rd_badTT;
	     	  int icells=0;
	     
	     	  for (CIfeped p = dataset_TpgBadTT.begin(); p != dataset_TpgBadTT.end(); p++) {
	       	    rd_badTT  = *p;
	  
		    int tcc_num=rd_badTT.getTCCId();
		    int tt_num=rd_badTT.getTTId();
		 
		    std::cout<< " tcc/tt"<< tcc_num<<"/"<<tt_num<< endl;

		    int ebTTDetId=tcc_num*100+tt_num;
		 
		    towerStatus->setValue(ebTTDetId,rd_badTT.getStatus());
		 
		    ++icells;
	          }
	     

	     	  edm::LogInfo("EcalTPGBadTTHandler") << "Finished badTT reading.";
	     
	     	  Time_t snc= (Time_t) irun ;                      
	     
	          m_to_transfer.push_back(std::make_pair((EcalTPGTowerStatus*)towerStatus,snc));
	     
	          m_i_run_number=irun;
	          m_i_tag=the_config_tag;
	          m_i_version=the_config_version;
	          m_i_badTT=badttId;
	     
	          writeFile("last_tpg_badTT_settings.txt");
	     
	        } else {
	     
	          m_i_run_number=irun;
	          m_i_tag=the_config_tag;
	          m_i_version=the_config_version;
	     
	          writeFile("last_tpg_badTT_settings.txt");
	     
	          //  std::cout<< " even if the tag/version is not the same, the badTT id is the same -> no transfer needed "<< std::endl; 
	     
	        }
	   
	      }       
	      
	      catch (std::exception &e) { 
		std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" <<the_config_tag
			  <<" version="<<the_config_version<< std::endl;
		cout << e.what() << endl;
		m_i_run_number=irun;
		
	      }
	      
	      
	    } else if(nr==0) {
	      m_i_run_number=irun;
	      //	      std::cout<< " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed "<< std::endl; 
	    } else {
	      m_i_run_number=irun;
	      m_i_tag=the_config_tag;
	      m_i_version=the_config_version;

	      writeFile("last_tpg_badTT_settings.txt");
	    }	
	 }
        }
	
	delete econn;

  	edm::LogInfo("EcalTPGBadTTHandler") << "Ecal - > end of getNewObjects -----------";        
}


void  popcon::EcalTPGBadTTHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_badTT=0; 

  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGBadTTHandler")<<"*** Can not open file: "<<inputFile;
  }

  char line[256];
    
  std::ostringstream str;

  fgets(line,255,inpFile);
  m_i_tag=to_string(line);
  str << "gen tag " << m_i_tag << endl ;  // should I use this? 

  fgets(line,255,inpFile);
  m_i_version=atoi(line);
  str << "version= " << m_i_version << endl ;  

  fgets(line,255,inpFile);
  m_i_run_number=atoi(line);
  str << "run_number= " << m_i_run_number << endl ;  

  fgets(line,255,inpFile);
  m_i_badTT=atoi(line);
  str << "badTT_config= " << m_i_badTT << endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGBadTTHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<endl;
  myfile << m_i_version <<endl;
  myfile << m_i_run_number <<endl;
  myfile << m_i_badTT <<endl;

  myfile.close();

}
