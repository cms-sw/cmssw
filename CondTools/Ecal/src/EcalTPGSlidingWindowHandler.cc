#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGSlidingWindowHandler.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingDat.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include<iostream>

popcon::EcalTPGSlidingWindowHandler::EcalTPGSlidingWindowHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGSlidingWindowHandler")) {

        edm::LogInfo("EcalTPGSlidingWindowHandler") << "EcalTPGSlidingWindow Source handler constructor";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGSlidingWindowHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;


}

popcon::EcalTPGSlidingWindowHandler::~EcalTPGSlidingWindowHandler()
{
}

void popcon::EcalTPGSlidingWindowHandler::getNewObjects()
{
	edm::LogInfo("EcalTPGSlidingWindowHandler") << "Started GetNewObjects!!!";

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
	edm::LogInfo("EcalTPGSlidingWindowHandler") << "max_since : "  << max_since;
	edm::LogInfo("EcalTPGSlidingWindowHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 

	edm::LogInfo("EcalTPGSlidingWindowHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGSlidingWindowHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGSlidingWindowHandler") << "Done.";
	
	if (!econn)
	{
	    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass;
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
	if((int)m_firstRun<max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGSlidingWindowHandler") << "min_run= " << min_run << "max_run= " << max_run;

        RunList my_list;
        //my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run, my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGSlidingWindowHandler") <<"number of runs is : "<< mon_runs;
            	
        unsigned long irun=0;
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGSlidingWindowHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGSlidingWindowHandler") << "Fetching run by tag";

	    // retrieve the data :
	    std::map<EcalLogicID, RunTPGConfigDat> dataset;
	    econn->fetchDataSet(&dataset, &run_vec[kr]);
	    std::string the_config_tag="";
	    std::map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;
	    FEConfigMainInfo fe_main_info;
	    
	    
	    int nr=0;
	    for( it=dataset.begin(); it!=dataset.end(); it++ )
	      {
		++nr;
		EcalLogicID ecalid  = it->first;
		RunTPGConfigDat  dat = it->second;
		std::string the_config_tag=dat.getConfigTag();
		edm::LogInfo("EcalTPGSlidingWindowHandler") <<"config_tag "<<the_config_tag;
		fe_main_info.setConfigTag(the_config_tag);
		econn-> fetchConfigSet(&fe_main_info);

	      }
	    edm::LogInfo("EcalTPGSlidingWindowHandler") << "got " << nr << "objects in dataset.";


	    // now get TPGSlidingWindow
	    int sliId=fe_main_info.getSliId();
	    FEConfigSlidingInfo fe_sli_info;
	    fe_sli_info.setId(sliId);
	    econn-> fetchConfigSet(&fe_sli_info);
	    std::map<EcalLogicID, FEConfigSlidingDat> dataset_TpgSli;
	    econn->fetchDataSet(&dataset_TpgSli, &fe_sli_info);

	    EcalTPGSlidingWindow * sliW = new EcalTPGSlidingWindow;
	    typedef std::map<EcalLogicID, FEConfigSlidingDat>::const_iterator CIfesli;
	    EcalLogicID ecid_xt;
	    FEConfigSlidingDat rd_sli;
	    unsigned int  rd_slid;
	    int icells=0;
	    	    
	    for (CIfesli p = dataset_TpgSli.begin(); p != dataset_TpgSli.end(); p++) {
	      ecid_xt = p->first;
	      rd_sli  = p->second;
	      
	      std::string ecid_name=ecid_xt.getName();
	      
	      // EB data
	      if (ecid_name=="EB_VFE") {
	        int id1=ecid_xt.getID1();
	        int id2=ecid_xt.getID2();
	        int id3=ecid_xt.getID3();
	    
	        rd_slid = (unsigned int)rd_sli.getSliding();
	       	  
		char ch[10];
		sprintf(ch,"%d%d%d", id1, id2, id3); 
		
		std::string S="";
		S.insert(0,ch);
		
		unsigned int stripEBId = 0;
		stripEBId = atoi(S.c_str());
		   		
	        sliW->setValue((unsigned int)stripEBId, (unsigned int)rd_sli.getSliding());
	        ++icells;
	      	}
	      	else if (ecid_name=="EE_trigger_strip"){
	        
		// EE data
		int id1=ecid_xt.getID1();
	        int id2=ecid_xt.getID2();
	        int id3=ecid_xt.getID3();	
		
		rd_slid = (unsigned int)rd_sli.getSliding();
	       	
		char ch[10];
		sprintf(ch,"%d%d%d", id1, id2, id3);
		
		std::string S ="";
		S.insert(0,ch);
		       
		unsigned int stripEEId = atoi(S.c_str());		   
		
	        sliW->setValue(stripEEId, (unsigned int)rd_sli.getSliding());
	        ++icells;
	       }
	      }//for over the data
	    
	       
	    edm::LogInfo("EcalTPGSlidingWindowHandler") << "found " << icells << "strips.";

	    Time_t snc= (Time_t) irun; 	      
	    m_to_transfer.push_back(std::make_pair((EcalTPGSlidingWindow*)sliW,snc));

	  }//while over the runs
	}//if
	  
	delete econn;

	edm::LogInfo("EcalTPGSlidingWindowHandler") << "Ecal - > end of getNewObjects -----------";
	
}

