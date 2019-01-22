#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGWeightIdMapHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>
#include<fstream>


#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>


popcon::EcalTPGWeightIdMapHandler::EcalTPGWeightIdMapHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGWeightIdMapHandler")) {

        edm::LogInfo("EcalTPGWeightIdMapHandler") << "EcalTPGWeightIdMap Source handler constructor";
        m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
        m_lastRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtype=ps.getParameter<std::string>("RunType");
	m_file_type = ps.getParameter<std::string>("fileType");           // xml/txt
	m_file_name = ps.getParameter<std::string>("fileName");

        edm::LogInfo("EcalTPGWeightIdMapHandler") << m_sid<<"/"<<m_user<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGWeightIdMapHandler::~EcalTPGWeightIdMapHandler() {
}


void popcon::EcalTPGWeightIdMapHandler::getNewObjects() {
  if(m_file_type == "txt") {
    readtxtFile();
  }
  else if(m_file_type == "xml") {
    readxmlFile();
  }
  else {
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Started GetNewObjects!!!";

	//check whats already inside of database
	if (tagInfo().size){
  	//check whats already inside of database
	  edm::LogInfo("got offlineInfo = ");
	  edm::LogInfo("tag name = ") << tagInfo().name;
	  edm::LogInfo("size = ") << tagInfo().size;
    	} else {
	  edm::LogInfo(" First object for this tag ");
    	}

	unsigned int max_since=0;
	max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "max_since : "  << max_since;
	Ref weightIdMap_db = lastPayload();
	
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Done.";
	
	if (!econn)
	  {
	    edm::LogInfo(" connection parameters ") <<m_sid <<"/"<<m_user;
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

	readFromFile("last_tpg_weightIdMap_settings.txt");


 	unsigned int min_run=m_i_run_number+1;

	if(m_firstRun<m_i_run_number) {
	  min_run=m_i_run_number+1;
	} else {
	  min_run=m_firstRun;
	}
	
	if(min_run<max_since) {
	  min_run=max_since+1; // we have to add 1 to the last transferred one
	} 

	edm::LogInfo("m_i_run_number")<< m_i_run_number <<"m_firstRun "<<m_firstRun<< "max_since " <<max_since;

	unsigned int max_run=m_lastRun;
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "min_run= " << min_run << "max_run= " << max_run;

        RunList my_list;
	my_list=econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
	//        my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	size_t num_runs=run_vec.size();
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "number of Mon runs is : "<< num_runs;

	unsigned int irun;
	if(num_runs>0){
	  for(size_t kr=0; kr<run_vec.size(); kr++){

	    irun=static_cast<unsigned int>(run_vec[kr].getRunNumber());

	    edm::LogInfo(" run= ")<<irun;
	  
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


			 edm::LogInfo(" run= ")<<irun<<" tag "<<the_config_tag<<" version="<<the_config_version;

	    // here we should check if it is the same as previous run.


	    if((the_config_tag != m_i_tag || the_config_version != m_i_version ) && nr>0 ) {
	      edm::LogInfo("the tag is different from last transferred run ... retrieving last config set from DB");

	      FEConfigMainInfo fe_main_info;
	      fe_main_info.setConfigTag(the_config_tag);
	      fe_main_info.setVersion(the_config_version);

	      try{ 
		econn-> fetchConfigSet(&fe_main_info);


        	// now get TPGWeightIdMap
        	int weightId=fe_main_info.getWeiId();
		
		if( weightId != m_i_weightIdMap ) {
		
		  FEConfigWeightInfo fe_weight_info;
	 	  fe_weight_info.setId(weightId);
		  econn-> fetchConfigSet(&fe_weight_info);
       		  std::map<EcalLogicID, FEConfigWeightGroupDat> dataset_TpgWeight;
		  econn->fetchDataSet(&dataset_TpgWeight, &fe_weight_info);
		  edm::LogInfo("EcalTPGWeightIdMapHandler") << "Got object!";
		  EcalTPGWeightIdMap* weightMap = new EcalTPGWeightIdMap;
		  typedef std::map<EcalLogicID, FEConfigWeightGroupDat>::const_iterator CIfeweight;
		  EcalLogicID ecid_xt;
		  FEConfigWeightGroupDat  rd_w;

		  int igroups=0;
		  for (CIfeweight p = dataset_TpgWeight.begin(); p != dataset_TpgWeight.end(); p++) {
		
		  rd_w =  p->second;
	  	  // EB and EE data 
	  	  EcalTPGWeights w;		
	  	  unsigned int weight0 = static_cast<unsigned int>(rd_w.getWeight4());
	  	  unsigned int weight1 = static_cast<unsigned int>(rd_w.getWeight3());
	  	  unsigned int weight2 = static_cast<unsigned int>(rd_w.getWeight2());
	  	  unsigned int weight3 = static_cast<unsigned int>(rd_w.getWeight1()- 0x80);
	  	  unsigned int weight4 = static_cast<unsigned int>(rd_w.getWeight0());
	  
        	  w.setValues(weight0,weight1,weight2,weight3,weight4);
        	  weightMap->setValue(rd_w.getWeightGroupId(),w);
	  
	  	  ++igroups;
		}
	
		edm::LogInfo("EcalTPGWeightIdMapHandler") << "found " << igroups << "Weight groups";

 		Time_t snc= (Time_t) irun; 	      
 		m_to_transfer.push_back(std::make_pair((EcalTPGWeightIdMap*)weightMap,snc));
	   		  
		m_i_run_number=irun;
		m_i_tag=the_config_tag;
		m_i_version=the_config_version;
		m_i_weightIdMap=weightId;
		  
		writeFile("last_tpg_weightIdMap_settings.txt");

		} else {

		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;

		  writeFile("last_tpg_weightIdMap_settings.txt");

		  edm::LogInfo(" even if the tag/version is not the same, the weightIdMap id is the same -> no transfer needed "); 

		}

	      }       catch (std::exception &e) { 
		edm::LogInfo("ERROR: THIS CONFIG DOES NOT EXIST: tag=") <<the_config_tag
			  <<" version="<<the_config_version;
			     edm::LogInfo("error")<<e.what();
		m_i_run_number=irun;

	      }
	      
	    } else if(nr==0) {
	      m_i_run_number=irun;
	      edm::LogInfo( " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor,  no transfer needed "); 
	    } else {
	      m_i_run_number=irun;
	      m_i_tag=the_config_tag;
	      m_i_version=the_config_version;
	      edm::LogInfo(" the tag/version is the same, no transfer needed "); 
	      writeFile("last_tpg_weightIdMap_settings.txt");
	    }
	    
	  }
	}
	  
	delete econn;
  }  // usual way
  edm::LogInfo("EcalTPGWeightIdMapHandler") << "Ecal - > end of getNewObjects -----------";
}
void  popcon::EcalTPGWeightIdMapHandler::readtxtFile() {
  edm::LogInfo(" reading the input file ") << m_file_name;
  std::ifstream fInput;
  fInput.open(m_file_name);
  if(!fInput.is_open()) {
    edm::LogInfo("ERROR : cannot open file ") << m_file_name;
    exit (1);
  }
  unsigned int wloc[5];
  EcalTPGWeights w;		
  EcalTPGWeightIdMap* weightMap = new EcalTPGWeightIdMap;
  int igroups = 0;
  for (int ifirst = 0; ifirst < 2; ifirst++) {
    for (int isecond = 0; isecond < 5; isecond++)
      fInput >> wloc[isecond];
    w.setValues(wloc[0], wloc[1], wloc[2], wloc[3], wloc[4]);
    weightMap->setValue(igroups, w);
    igroups++;
  }
  edm::LogInfo("EcalTPGWeightIdMapHandler") << "found " << igroups << "Weight groups";
  try{ 
    Time_t snc= (Time_t) m_firstRun; 	      
    m_to_transfer.push_back(std::make_pair((EcalTPGWeightIdMap*)weightMap, snc));
  } catch (std::exception &e) { 
    edm::LogInfo("EcalTPGWeightIdMapHandler::readtxtFile error : ") << e.what() << std::endl;
  }
}

void  popcon::EcalTPGWeightIdMapHandler::readxmlFile() {
  edm::LogInfo(" reading the input file ") << m_file_name;
  std::ifstream fxml;
  fxml.open(m_file_name);
  if(!fxml.is_open()) {
    edm::LogInfo("ERROR : cannot open file ") << m_file_name;
    exit (1);
  }
  std::string dummyLine, bid;
  unsigned int wloc[5];
  EcalTPGWeights w;		
  EcalTPGWeightIdMap* weightMap = new EcalTPGWeightIdMap;
  int ngroups, igroups;
  edm::LogInfo("EcalTPGWeightIdMapHandler") << "found " << igroups << "Weight groups";
  for(int i = 0; i < 5; i++) std::getline(fxml, dummyLine);   // skip first lines
  fxml >> bid;
  std::string stt = bid.substr(7, 1);
  std::istringstream sc(stt);
  sc >> ngroups;
  if(ngroups != 2) {
    edm::LogInfo(" line : ") << bid << " ngroups " << ngroups;
    exit(-1);
  }
  for(int i = 0; i < 3; i++) std::getline(fxml, dummyLine);
  fxml >> bid;
  stt = bid.substr(7, 1);
  std::istringstream sg1(stt);
  sg1 >> igroups;
  if(igroups != 0) {
    edm::LogInfo(" group 1: ") << bid << " igroups " << igroups;
    exit(-1);
  }
  for(int i = 0; i < 2; i++) std::getline(fxml, dummyLine);
  for(int i = 0; i < 5; i++) {
    fxml >> bid;
    std::size_t found = bid.find("</");
    stt = bid.substr(5, found - 5);
    std::istringstream w(stt);
    w >> wloc[i];
  }
  w.setValues(wloc[0], wloc[1], wloc[2], wloc[3], wloc[4]);
  weightMap->setValue(igroups, w);
  // second group
  for(int i = 0; i < 4; i++) std::getline(fxml, dummyLine);
  fxml >> bid;
  stt = bid.substr(7, 1);
  std::istringstream sg2(stt);
  sg2 >> igroups;
  if(igroups != 1) {
    edm::LogInfo(" group 2 : ") << bid << " igroups " << igroups;
    exit(-1);
  }
  for(int i = 0; i < 2; i++) std::getline(fxml, dummyLine);
  for(int i = 0; i < 5; i++) {
    fxml >> bid;
    std::size_t found = bid.find("</");
    stt = bid.substr(5, found - 5);
    std::istringstream w(stt);
    w >> wloc[i];
  }
  w.setValues(wloc[0], wloc[1], wloc[2], wloc[3], wloc[4]);
  weightMap->setValue(igroups, w);
  try{ 
    Time_t snc= (Time_t) m_firstRun; 	      
    m_to_transfer.push_back(std::make_pair((EcalTPGWeightIdMap*)weightMap, snc));
  } catch (std::exception &e) { 
    edm::LogInfo("EcalTPGWeightIdMapHandler::readxmlFile error : ") << e.what();
  }
}

void  popcon::EcalTPGWeightIdMapHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_weightIdMap=0; 

  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGWeightIdMapHandler")<<"*** Can not open file: "<<inputFile;
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
  m_i_weightIdMap=atoi(line);
  str << "weightIdMap_config= " << m_i_weightIdMap << std::endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGWeightIdMapHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  std::ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<std::endl;
  myfile << m_i_version <<std::endl;
  myfile << m_i_run_number <<std::endl;
  myfile << m_i_weightIdMap <<std::endl;

  myfile.close();

}
