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
	edm::LogInfo("EcalTPGWeightGroupHandler") << "min_run= " << min_run << " max_run= " << max_run;

	RunList my_list;
        //my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGWeightGroupHandler") <<"number of Mon runs is : "<< mon_runs;

	//makeStripId();
	
	unsigned long irun=0;
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGWeightGroupHandler") << "\nHere is the run number: "<< run_vec[kr].getRunNumber();
	  
	    edm::LogInfo("EcalTPGWeightGroupHandler") << "Fetching run by tag";

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
		edm::LogInfo("EcalTPGWeightGroupHandler")<<"config_tag "<<the_config_tag;
		fe_main_info.setConfigTag(the_config_tag);
		econn->fetchConfigSet(&fe_main_info);

	      }
	    edm::LogInfo("EcalTPGWeightGroupHandler") << "got " << nr << "d objects in dataset.";


	    // now get TPGWeightGroup
	    int wId=fe_main_info.getWeiId();
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
	    
	    edm::LogInfo("EcalTPGWeightGroupHandler") << "found " << icells << "strips.";

	    Time_t snc= (Time_t) irun ;
	      	      
	    m_to_transfer.push_back(std::make_pair((EcalTPGWeightGroup*)weightG,snc));
	  }
	}
	  
	delete econn;

	edm::LogInfo("EcalTPGWeightGroupHandler") << "Ecal - > end of getNewObjects -----------";
	
}

