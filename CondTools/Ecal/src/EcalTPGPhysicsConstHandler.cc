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
	Ref ped_db = lastPayload();
	
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
	my_rundef.setRunType("PEDESTAL"); 

	RunTag  my_runtag;
	my_runtag.setLocationDef( my_locdef );
	my_runtag.setRunTypeDef(  my_rundef );
	my_runtag.setGeneralTag(m_gentag); 

	int min_run=0;
	if(m_firstRun<(unsigned int)max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "min_run= " << min_run << "max_run= " << max_run;
	
        RunList my_list;
	//my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGPhysicsConstHandler") <<"number of Mon runs is : "<< mon_runs;

	unsigned long irun;
	if(mon_runs>0){
	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGPhysicsConstHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGPhysicsConstHandler")<< "Fetching run by tag";

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
	      edm::LogInfo("EcalTPGPhysicsConstHandler") <<"config_tag "<<the_config_tag;
	      fe_main_info.setConfigTag(the_config_tag);
	      econn-> fetchConfigSet(&fe_main_info);

	    }
	    edm::LogInfo("EcalTPGPhysicsConstHandler") << "got " << nr << " objects in dataset";

            // now get TPGPhysicsConst
            int linId=fe_main_info.getLinId();
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
    }
  }
	  
	delete econn;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Ecal - > end of getNewObjects -----------";
	
}

