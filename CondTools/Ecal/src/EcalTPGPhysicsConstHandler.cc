#include "CondTools/Ecal/interface/EcalTPGPhysicsConstHandler.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigParamDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

popcon::EcalTPGPhysicsConstHandler::EcalTPGPhysicsConstHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGPhysicsConstHandler")) {

        edm::LogInfo("EcalTPGPhysicsConstHandler") << "EcalTPGPhysicsConst Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGPhysicsConstHandler")<< m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGPhysicsConstHandler::~EcalTPGPhysicsConstHandler()
{
}


void popcon::EcalTPGPhysicsConstHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Started GetNewObjects!!!";

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
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "max_since : "  << max_since;
	Ref physC_db = lastPayload();
	
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Done.";
	
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


        readFromFile("last_tpg_physC_settings.txt");

 	int min_run=m_i_run_number+1;

	if(m_firstRun<(unsigned int)m_i_run_number) {
	  min_run=(int) m_i_run_number+1;
	} else {
	  min_run=(int)m_firstRun;
	}
	
	std::cout<<"m_i_run_number"<< m_i_run_number <<"m_firstRun "<<m_firstRun<< "max_since " <<max_since<< endl;

	if(min_run<(unsigned int)max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} 

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "min_run= " << min_run << "max_run= " << max_run;
	
        RunList my_list;
	my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);

      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int num_runs=run_vec.size();

	std::cout <<"number of runs is : "<< num_runs<< endl;

	unsigned long irun;
	if(num_runs>0){
	
	  for(int kr=0; kr<run_vec.size(); kr++){

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
	   
            // now get TPGPhysicsConst
            int linId=fe_main_info.getLinId();
	    
	    if( linId != m_i_physC ) {
	    
	    FEConfigLinInfo fe_phys_info;
	    fe_phys_info.setId(linId);
	    econn-> fetchConfigSet(&fe_phys_info);
	    map<EcalLogicID, FEConfigParamDat> dataset_TpgPhysics;
	    econn->fetchDataSet(&dataset_TpgPhysics, &fe_phys_info);

	    EcalTPGPhysicsConst* physC = new EcalTPGPhysicsConst;
            typedef map<EcalLogicID, FEConfigParamDat>::const_iterator CIfelin;
	    EcalLogicID ecid_xt;
	    FEConfigParamDat  rd_phys;
	    int icells=0;
	    for (CIfelin p = dataset_TpgPhysics.begin(); p != dataset_TpgPhysics.end(); p++) 
	    {
	      ecid_xt = p->first;
	      rd_phys  = p->second;
	  
	      std::string ecid_name=ecid_xt.getName();    
	      // Ecal barrel detector
	      if(ecid_name=="EB") {
	      //int sm_num=ecid_xt.getID1();
	      //int xt_num=ecid_xt.getID2();
	  
	      // I am not sure, may be have to use EcalLogicId but it is empty (see in ..)
	      DetId eb(DetId::Ecal, EcalBarrel);
	  
	      EcalTPGPhysicsConst::Item item;
	      item.EtSat=rd_phys.getETSat();
	      item.ttf_threshold_Low=rd_phys.getTTThreshlow();
	      item.ttf_threshold_High=rd_phys.getTTThreshhigh();
	      item.FG_lowThreshold=rd_phys.getFGlowthresh();
	      item.FG_highThreshold=rd_phys.getFGhighthresh();
	      item.FG_lowRatio=rd_phys.getFGlowratio();
	      item.FG_highRatio= rd_phys.getFGhighratio();	  
	  
	      physC->setValue(eb.rawId(),item);
	  
	      ++icells;
	    }
	    else if (ecid_name=="EE") {
	      // Ecan endcap detector
	  
	      //int sm_num=ecid_xt.getID1();
	      //int xt_num=ecid_xt.getID2();
	  
	      // I am not sure, may be have to use EcalLogicId but it is empty (see in ..)
	      DetId ee(DetId::Ecal, EcalEndcap);
	  
	      EcalTPGPhysicsConst::Item item;
	      item.EtSat=rd_phys.getETSat();
	      item.ttf_threshold_Low=rd_phys.getTTThreshlow();
	      item.ttf_threshold_High=rd_phys.getTTThreshhigh();
	      item.FG_lowThreshold=rd_phys.getFGlowthresh();
	      item.FG_highThreshold=rd_phys.getFGhighthresh();
	      // the last two is empty for the EE
	      item.FG_lowRatio=rd_phys.getFGlowratio();
	      item.FG_highRatio= rd_phys.getFGhighratio();	  
	  
	      physC->setValue(ee.rawId(),item);
	   
	      ++icells;	  
	  }
	}
		
 	      Time_t snc= (Time_t) irun ;
	      	      
 	      m_to_transfer.push_back(std::make_pair((EcalTPGPhysicsConst*)physC,snc));
	      
	      
	          m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;
		  m_i_physC=linId;
		  
		  writeFile("last_tpg_physC_settings.txt");

		} else {

		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;

		  writeFile("last_tpg_physC_settings.txt");

		  std::cout<< " even if the tag/version is not the same, the physics constants id is the same -> no transfer needed "<< std::endl; 

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
	      writeFile("last_tpg_physC_settings.txt");
	    }
    }
  }
	  
	delete econn;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Ecal - > end of getNewObjects -----------";
	
}


void  popcon::EcalTPGPhysicsConstHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_physC=0; 

  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGPhysicsConstHandler")<<"*** Can not open file: "<<inputFile;
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
  m_i_physC=atoi(line);
  str << "physC_config= " << m_i_physC << endl ;  

    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGPhysicsConstHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<endl;
  myfile << m_i_version <<endl;
  myfile << m_i_run_number <<endl;
  myfile << m_i_physC <<endl;

  myfile.close();

}
