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
        m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
        m_lastRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtype=ps.getParameter<std::string>("RunType");

        edm::LogInfo("EcalTPGBadTTHandler") << m_sid<<"/"<<m_user<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGBadTTHandler::~EcalTPGBadTTHandler()
{
}


void popcon::EcalTPGBadTTHandler::getNewObjects()
{
    	edm::LogInfo("EcalTPGBadTTHandler") << "Started GetNewObjects!!!";

    	unsigned int max_since=0;
    	max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
    	edm::LogInfo("EcalTPGBadTTHandler") << "max_since : "  << max_since;
    	edm::LogInfo("EcalTPGBadTTHandler") << "retrieved last payload ";

    	// here we retrieve all the runs after the last from online DB 
    	edm::LogInfo("EcalTPGBadTTHandler") << "Retrieving run list from ONLINE DB ... ";

    	edm::LogInfo("EcalTPGBadTTHandler") << "Making connection...";
    	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    	edm::LogInfo("EcalTPGBadTTHandler") << "Done.";
        
    	if (!econn)
    	{
      	  std::cout << " connection parameters " <<m_sid <<"/"<<m_user<<std::endl;
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
	edm::LogInfo("EcalTPGBadTTHandler") << "min_run= " << min_run << "max_run= " << max_run;

	RunList my_list; 
	//	my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef); 
	my_list=econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
    
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	size_t num_runs=run_vec.size();

	std::cout <<"number of runs is : "<< num_runs<< std::endl;
       
    	std::string str="";
    
	unsigned int irun=0;
	if(num_runs>0){


	  // going to query the ecal logic id 
	    std::vector<EcalLogicID> my_TTEcalLogicId_EE;
	    my_TTEcalLogicId_EE = econn->getEcalLogicIDSetOrdered( "EE_trigger_tower",
						    1, 200,
						    1, 70,
						    EcalLogicID::NULLID,EcalLogicID::NULLID,
						    "EE_offline_towerid",12 );
	    std::cout <<" GOT the logic ID for the EE trigger towers "<< std::endl;



	 for(size_t kr=0; kr<run_vec.size(); kr++){
	  
	    irun=static_cast<unsigned int>(run_vec[kr].getRunNumber());

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


	    // here we should check if it is the same as previous run.


	    if((the_config_tag != m_i_tag || the_config_version != m_i_version ) && nr>0 ) {
	      std::cout<<" run= "<<irun<<" tag "<<the_config_tag<<" version="<<the_config_version <<std::endl;
	      std::cout<<"the tag is different from last transferred run ... retrieving last config set from DB"<<std::endl;

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

		  // reset the map 
		  // EB
		  for(int ism=1; ism<=36; ism++) {
		    for(int ito=1; ito<=68; ito++) {
		      int tow_eta=(ito-1)/4;
		      int tow_phi=((ito-1)-tow_eta*4);
		      int axt=(tow_eta*5)*20 + tow_phi*5 +1 ;
		      EBDetId id(ism, axt, EBDetId::SMCRYSTALMODE ) ;
		      const EcalTrigTowerDetId towid= id.tower();
		      int tower_status=0;
		      towerStatus->setValue(towid.rawId(),tower_status);
		    }
		  }
		  //EE
		  for (size_t itower=0; itower<my_TTEcalLogicId_EE.size(); itower++) {
		    int towid =my_TTEcalLogicId_EE[itower].getLogicID();
		    int tower_status=0;
		    towerStatus->setValue(towid,tower_status);
		  }

		  // now put at 1 those that are bad 
	     	  int icells=0;
	     	  for (CIfeped p = dataset_TpgBadTT.begin(); p != dataset_TpgBadTT.end(); p++) {
	       	    rd_badTT  = *p;
	  
		    int tcc_num=rd_badTT.getTCCId();
		    int tt_num=rd_badTT.getTTId();
		 
		    std::cout<< " tcc/tt"<< tcc_num<<"/"<<tt_num<< std::endl;

	      	    if(tcc_num>36 && tcc_num<=72) {
	      	      // SM number
              	      int smid=tcc_num-54;
		      if(tcc_num<55) smid=tcc_num-18;
              	      // TT number
	      	      int towerid=tt_num;

		      int tow_eta=(towerid-1)/4;
		      int tow_phi=((towerid-1)-tow_eta*4);
		      int axt=(tow_eta*5)*20 + tow_phi*5 +1 ;

		      EBDetId id(smid, axt, EBDetId::SMCRYSTALMODE ) ;
		      const EcalTrigTowerDetId towid= id.tower();
		      towerStatus->setValue(towid.rawId(),rd_badTT.getStatus());
  
		      ++icells;
	    	    }  else  {
	      	      // EE data

	      	      // TCC number
      	      	      int tccid=tcc_num;
      	      	      // TT number
	      	      int towerid=tt_num;

		      bool set_the_tower=false;
		      int towid;
		      for (size_t itower=0; itower<my_TTEcalLogicId_EE.size(); itower++) {

			if(!set_the_tower){
			  
			  if(my_TTEcalLogicId_EE[itower].getID1()==tccid && my_TTEcalLogicId_EE[itower].getID2()==towerid){
			    towid =my_TTEcalLogicId_EE[itower].getLogicID();
			    set_the_tower=true;
			    break;
			  }
			}
			
		      }
		      
		      if(set_the_tower){

			towerStatus->setValue(towid,rd_badTT.getStatus());
  

		      } else {
			std::cout <<" these may be the additional towers TCC/TT "
				  << tccid<<"/"<<towerid<<std::endl;
		      }
	      	      
		      ++icells;

	  	    }
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
		std::cout << e.what() << std::endl;
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
  m_i_badTT=atoi(line);
  str << "badTT_config= " << m_i_badTT << std::endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGBadTTHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  std::ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<std::endl;
  myfile << m_i_version <<std::endl;
  myfile << m_i_run_number <<std::endl;
  myfile << m_i_badTT <<std::endl;

  myfile.close();

}
