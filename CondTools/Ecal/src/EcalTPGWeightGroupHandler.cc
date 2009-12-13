#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGWeightGroupHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingDat.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

#include<iostream>
#include<fstream>


#include <time.h>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>


popcon::EcalTPGWeightGroupHandler::EcalTPGWeightGroupHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGWeightGroupHandler")) {

        edm::LogInfo("EcalTPGWeightGroupHandler") << "EcalTPGWeightGroup Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtype=ps.getParameter<std::string>("RunType");

        edm::LogInfo("EcalTPGWeightGroupHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;
}

popcon::EcalTPGWeightGroupHandler::~EcalTPGWeightGroupHandler()
{
}

void popcon::EcalTPGWeightGroupHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGWeightGroupHandler") << "Started GetNewObjects!!!";

	//check whats already inside of database
	if (tagInfo().size){
  	//check whats already inside of database
    	std::cout << "got offlineInfo = " << std::endl;
	std::cout << "tag name = " << tagInfo().name << std::endl;
	std::cout << "size = " << tagInfo().size <<  std::endl;
    	} else {
    	std::cout << " First object for this tag " << std::endl;
    	}

	int max_since=0;
	max_since=(int)tagInfo().lastInterval.first;
	edm::LogInfo("EcalTPGWeightGroupHandler") << "max_since : "  << max_since;
	
	edm::LogInfo("EcalTPGWeightGroupHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 

	edm::LogInfo("EcalTPGWeightGroupHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGWeightGroupHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGWeightGroupHandler") << "Done.";
	
	if (!econn)
	  {
	    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
	    //	    cerr << e.what() << endl;
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

	readFromFile("last_tpg_weightGroup_settings.txt");

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
	edm::LogInfo("EcalTPGWeightGroupHandler") << "min_run= " << min_run << " max_run= " << max_run;

	RunList my_list;
        my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int num_runs=run_vec.size();

	std::cout <<"number of runs is : "<< num_runs<< endl;

	//makeStripId();
	
	unsigned long irun=0;
	if(num_runs>0){

	  for(int kr=0; kr<(int)run_vec.size(); kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();

	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" run= "<<irun<<std::endl;

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


	    std::cout<<" run= "<<irun<<" tag "<<the_config_tag<<" version="<<the_config_version <<std::endl;

	    // here we should check if it is the same as previous run.


	    if((the_config_tag != m_i_tag || the_config_version != m_i_version ) && nr>0 ) {
	      std::cout<<"the tag is different from last transferred run ... retrieving last config set from DB"<<endl;

	      FEConfigMainInfo fe_main_info;
	      fe_main_info.setConfigTag(the_config_tag);
	      fe_main_info.setVersion(the_config_version);

	      try{ 
		std::cout << " before fetch config set" << std::endl;	    
		econn-> fetchConfigSet(&fe_main_info);
		std::cout << " after fetch config set" << std::endl;	    


	    	// now get TPGWeightGroup
	    	int wId=fe_main_info.getWeiId();
	    
	    	if( wId != m_i_weightGroup ) {
	    
	    	  FEConfigWeightInfo fe_w_info;
	    	  fe_w_info.setId(wId);
	    	  econn-> fetchConfigSet(&fe_w_info);
	    	  map<EcalLogicID, FEConfigWeightDat> dataset_TpgW;
	    	  econn->fetchDataSet(&dataset_TpgW, &fe_w_info);


	    	  EcalTPGWeightGroup* weightG = new EcalTPGWeightGroup;
	    	  typedef map<EcalLogicID, FEConfigWeightDat>::const_iterator CIfesli;
	    	  EcalLogicID ecid_xt;
	    	  int weightGroup;
	    	  int icells=0;
	    
	    	  std::map<std::string,int> map;
	    	  std::string str;

	    	  for (CIfesli p = dataset_TpgW.begin(); p != dataset_TpgW.end(); p++) {
	      	    ecid_xt = p->first;
	      	    weightGroup  = p->second.getWeightGroupId();
	      
	      	    std::string ecid_name=ecid_xt.getName();
	      
	      	    // EB data
	      	    if(ecid_name=="EB_VFE") {
	              // SM num (1,36)
	              int id1=ecid_xt.getID1();
	              // TT num (1,68)
	              int id2=ecid_xt.getID2();
	              // strip num (1,5)
	              int id3=ecid_xt.getID3();
	      
	              char identSMTTST[10];			      
	              sprintf(identSMTTST,"%d%d%d", id1, id2, id3);
	              std::string S="";
		      S.insert(0,identSMTTST);
		
		      unsigned int stripEBId = 0;
		      stripEBId = atoi(S.c_str());
		      
	              weightG->setValue(stripEBId,weightGroup);
	              ++icells;
	      	    }
	       	    else if (ecid_name=="EE_trigger_strip"){
               	      // EE data to add
	       	      int id1=ecid_xt.getID1();
	              int id2=ecid_xt.getID2();
	              int id3=ecid_xt.getID3();	

		      char ch[10];
		      sprintf(ch,"%d%d%d", id1, id2, id3);
		
		      std::string S ="";
		      S.insert(0,ch);
		       
		      unsigned int stripEEId = atoi(S.c_str());		   

	       	      weightG->setValue(stripEEId,weightGroup);
	              ++icells;    
	      	    }
	          } 
	    
	          Time_t snc= (Time_t) irun ;
	      	      
	          m_to_transfer.push_back(std::make_pair((EcalTPGWeightGroup*)weightG,snc));
	    
	    	  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;
		  m_i_weightGroup=wId;
		  
		  writeFile("last_tpg_weightGroup_settings.txt");

		} else {

		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;

		  writeFile("last_tpg_weightGroup_settings.txt");

		  std::cout<< " even if the tag/version is not the same, the weight group id is the same -> no transfer needed "<< std::endl; 

		}

	      }       
	      
	      catch (std::exception &e) { 
		std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" <<the_config_tag
			  <<" version="<<the_config_version<< std::endl;
		cout << e.what() << endl;
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
	      writeFile("last_tpg_weightGroup_settings.txt");
	    }

	  }
	}
	  
	delete econn;

	edm::LogInfo("EcalTPGWeightGroupHandler") << "Ecal - > end of getNewObjects -----------";
	
}

void  popcon::EcalTPGWeightGroupHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_weightGroup=0; 

  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGWeightGroupHandler")<<"*** Can not open file: "<<inputFile;
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
  m_i_weightGroup=atoi(line);
  str << "weightGroup_config= " << m_i_weightGroup << endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGWeightGroupHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<endl;
  myfile << m_i_version <<endl;
  myfile << m_i_run_number <<endl;
  myfile << m_i_weightGroup <<endl;

  myfile.close();

}
