#include "CondTools/Ecal/interface/EcalTPGFineGrainEBIdMapHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

popcon::EcalTPGFineGrainEBIdMapHandler::EcalTPGFineGrainEBIdMapHandler(const edm::ParameterSet & ps)
    :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGFineGrainEBIdMapHandler")) {
       
        edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "EcalTPGFineGrainEBIdMap Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;
}

popcon::EcalTPGFineGrainEBIdMapHandler::~EcalTPGFineGrainEBIdMapHandler()
{
}


void popcon::EcalTPGFineGrainEBIdMapHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Started GetNewObjects!!!";

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
	max_since=(int)tagInfo().lastInterval.first;
	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "max_since : "  << max_since;
	Ref ped_db = lastPayload();
	
	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "retrieved last payload ";

	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") <<"WOW: we just retrieved the last valid record from DB ";


	// here we retrieve all the runs after the last from online DB 

	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Done.";
	
	if (!econn)
	  {
	    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
	    //	    cerr << e.what() << endl;
	    throw cms::Exception("OMDS not available");
	  } 

	
	LocationDef my_locdef;
	my_locdef.setLocation(m_location); 

	RunTypeDef my_rundef;
	my_rundef.setRunType("PEDESTAL"); 

	RunTag  my_runtag;
	my_runtag.setLocationDef( my_locdef );
	my_runtag.setRunTypeDef(  my_rundef );
	//    my_runtag.setGeneralTag( generalTag );  <- this is normally "GLOBAL"
	my_runtag.setGeneralTag(m_gentag); 

	int min_run=0;
	if(m_firstRun<max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "min_run= " << min_run << "max_run= " << max_run;

	RunList my_list; 
        //my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") <<"number of Mon runs is : "<< mon_runs;

	unsigned long irun;
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Fetching run by tag";

	    // retrieve the data :
	    map<EcalLogicID, RunTPGConfigDat> dataset;
	    econn->fetchDataSet(&dataset, &run_vec[kr]);
	    std::string the_config_tag="";
	    map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;
	    FEConfigMainInfo fe_main_info;
	    int nr=0;
            for( it=dataset.begin(); it!=dataset.end(); it++ )
	    {
	      ++nr;
	      EcalLogicID ecalid  = it->first;
	      RunTPGConfigDat  dat = it->second;
	      std::string the_config_tag=dat.getConfigTag();
	      edm::LogInfo("EcalTPGFineGrainEBIdMapHandler")<<"config_tag "<<the_config_tag;
	      fe_main_info.setConfigTag(the_config_tag);
	      econn-> fetchConfigSet(&fe_main_info);
	    }
	    edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "got " << nr << "d objects in dataset.";

            // now get TPGFineGrainEBIdMap
            int fgrId=fe_main_info.getFgrId();
	    FEConfigFgrInfo fe_fgr_info;
	    fe_fgr_info.setId(fgrId);
	    econn-> fetchConfigSet(&fe_fgr_info);
       	    map<EcalLogicID, FEConfigFgrGroupDat> dataset_TpgFineGrainEB;
	    econn->fetchDataSet(&dataset_TpgFineGrainEB, &fe_fgr_info);
	    edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Got object!";
	    EcalTPGFineGrainEBIdMap *fgrMap = new EcalTPGFineGrainEBIdMap;
	    typedef map<EcalLogicID, FEConfigFgrGroupDat>::const_iterator CIfefgr;
	    EcalLogicID ecid_xt;
	    FEConfigFgrGroupDat  rd_fgr;

	    int igroups=0;	
	    for (CIfefgr p = dataset_TpgFineGrainEB.begin(); p != dataset_TpgFineGrainEB.end(); p++) 
	    {
	      ecid_xt = p->first;
	      rd_fgr  = p->second;
	  	  
	      std::string ecid_name=ecid_xt.getName();
	  
	      EcalTPGFineGrainConstEB f;
	      unsigned int ThrL = (unsigned int)rd_fgr.getThreshLow();
	      unsigned int ThrH = (unsigned int)rd_fgr.getThreshHigh();
	      unsigned int RatioL = (unsigned int)rd_fgr.getRatioLow();
	      unsigned int RatioH = (unsigned int)rd_fgr.getRatioHigh();
	      unsigned int LutConfId = (unsigned int)rd_fgr.getLUTConfId();
	  
	      f.setValues(ThrL,ThrH,RatioL,RatioH,LutConfId);
              fgrMap->setValue(rd_fgr.getFgrGroupId(),f);
	      ++igroups;
	    }
	
	    edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "found " << igroups << "Weight groups.";

 	    Time_t snc= (Time_t) irun ;	      	      
 	    m_to_transfer.push_back(std::make_pair((EcalTPGFineGrainEBIdMap*)fgrMap,snc));
	  }
	}
	  
	delete econn;
	edm::LogInfo("EcalTPGFineGrainEBIdMapHandler") << "Ecal - > end of getNewObjects -----------";
}

