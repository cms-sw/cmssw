#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGFineGrainStripEEHandler.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include<iostream>

popcon::EcalTPGFineGrainStripEEHandler::EcalTPGFineGrainStripEEHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGFineGrainStripEEHandler")) {

        edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "EcalTPGFineGrainStripEEHandler Source handler constructor";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGFineGrainStripEEHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;


}

popcon::EcalTPGFineGrainStripEEHandler::~EcalTPGFineGrainStripEEHandler()
{
}

void popcon::EcalTPGFineGrainStripEEHandler::getNewObjects()
{
	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Started GetNewObjects!!!";

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
	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "max_since : "  << max_since;
	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 

	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Done.";
	
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
	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "min_run= " << min_run << "max_run= " << max_run;

        RunList my_list;
        //my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run, my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGFineGrainStripEEHandler") <<"number of runs is : "<< mon_runs;
            	
        unsigned long irun=0;
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Fetching run by tag";

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
		edm::LogInfo("EcalTPGFineGrainStripEEHandler") <<"config_tag "<<the_config_tag;
		fe_main_info.setConfigTag(the_config_tag);
		econn-> fetchConfigSet(&fe_main_info);

	      }
	    edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "got " << nr << "objects in dataset.";


	    // now get TPGFineGrainStripEE
	    int fgrId=fe_main_info.getFgrId();
	    FEConfigFgrInfo fe_fgr_info;
	    fe_fgr_info.setId(fgrId);
	    econn-> fetchConfigSet(&fe_fgr_info);
	    std::map<EcalLogicID, FEConfigFgrGroupDat> dataset_TpgFineGrainStripEE;
	    econn->fetchDataSet(&dataset_TpgFineGrainStripEE, &fe_fgr_info);

	    EcalTPGFineGrainStripEE * fgrStripEE = new EcalTPGFineGrainStripEE;
	    typedef std::map<EcalLogicID, FEConfigFgrGroupDat>::const_iterator CIfefgr;
	    EcalLogicID ecid_xt;
	    FEConfigFgrGroupDat  rd_fgr;

	    int icells=0;
	    	    
	    for (CIfefgr p = dataset_TpgFineGrainStripEE.begin(); p != dataset_TpgFineGrainStripEE.end(); p++) {
	      
	      ecid_xt = p->first;
	      rd_fgr  = p->second;
	      
	      std::string ecid_name=ecid_xt.getName();
	      
	      // EE data
	      if (ecid_name=="EE_trigger_strip"){
	        
		// EE data
		// TCC
		int id1=ecid_xt.getID1();
	        // TT
		int id2=ecid_xt.getID2();
	        // Strip
		int id3=ecid_xt.getID3();	
			       	
		char ch[10];
		sprintf(ch,"%d%d%d", id1, id2, id3);
		
		std::string S ="";
		S.insert(0,ch);
		
		// local strip identifier       
		unsigned int stripEEId = atoi(S.c_str());		   
		
		EcalTPGFineGrainStripEE::Item item;
		// check what data should be set into item
		item.threshold = (unsigned int)rd_fgr.getThreshLow();
	      	/*item.threshold? = (unsigned int)rd_fgr.getThreshHigh();
	      	item.? = (unsigned int)rd_fgr.getRatioLow();
	      	item.? = (unsigned int)rd_fgr.getRatioHigh();
	      	*/
		item.lut = (unsigned int)rd_fgr.getLUTValue();
		
	        fgrStripEE->setValue(stripEEId,item);
	        ++icells;
	       }
	      }//for over the data
	    
	       
	    edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "found " << icells << "strips.";

	    Time_t snc= (Time_t) irun; 	      
	    m_to_transfer.push_back(std::make_pair((EcalTPGFineGrainStripEE*)fgrStripEE,snc));

	  }//while over the runs
	}//if
	  
	delete econn;

	edm::LogInfo("EcalTPGFineGrainStripEEHandler") << "Ecal - > end of getNewObjects -----------";
	
}

