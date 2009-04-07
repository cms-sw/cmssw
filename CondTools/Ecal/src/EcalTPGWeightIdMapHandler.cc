#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGWeightIdMapHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

popcon::EcalTPGWeightIdMapHandler::EcalTPGWeightIdMapHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGWeightIdMapHandler")) {

        edm::LogInfo("EcalTPGWeightIdMapHandler") << "EcalTPGWeightIdMap Source handler constructor";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGWeightIdMapHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;


}

popcon::EcalTPGWeightIdMapHandler::~EcalTPGWeightIdMapHandler()
{
}


void popcon::EcalTPGWeightIdMapHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Started GetNewObjects!!!";

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
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "max_since : "  << max_since;
	Ref ped_db = lastPayload();
	
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Done.";
	
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
	if(m_firstRun<max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "min_run= " << min_run << "max_run= " << max_run;

        RunList my_list;
        //my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGWeightIdMapHandler") << "number of Mon runs is : "<< mon_runs;

	unsigned long irun;
	if(mon_runs>0){
	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGWeightIdMapHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	  
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
	      	  edm::LogInfo("EcalTPGWeightIdMapHandler")<<"config_tag "<<the_config_tag;
	      	  fe_main_info.setConfigTag(the_config_tag);
	     	  econn-> fetchConfigSet(&fe_main_info);
	  	}
		
		edm::LogInfo("EcalTPGWeightIdMapHandler") <<"got "<< nr <<"objects in dataset";


        	// now get TPGWeightIdMap
        	int weightId=fe_main_info.getWeiId();
		FEConfigWeightInfo fe_weight_info;
	 	fe_weight_info.setId(weightId);
		econn-> fetchConfigSet(&fe_weight_info);
       		map<EcalLogicID, FEConfigWeightGroupDat> dataset_TpgWeight;
		econn->fetchDataSet(&dataset_TpgWeight, &fe_weight_info);
		edm::LogInfo("EcalTPGWeightIdMapHandler") << "Got object!";
		EcalTPGWeightIdMap* weightMap = new EcalTPGWeightIdMap;
		typedef map<EcalLogicID, FEConfigWeightGroupDat>::const_iterator CIfeweight;
		EcalLogicID ecid_xt;
		FEConfigWeightGroupDat  rd_w;

		int igroups=0;
		for (CIfeweight p = dataset_TpgWeight.begin(); p != dataset_TpgWeight.end(); p++) {
	
	  	// EB and EE data 
	  	EcalTPGWeights w;		
	  	unsigned int weight0 = (unsigned int)rd_w.getWeight0();
	  	unsigned int weight1 = (unsigned int)rd_w.getWeight1();
	  	unsigned int weight2 = (unsigned int)rd_w.getWeight2();
	  	unsigned int weight3 = (unsigned int)rd_w.getWeight3();
	  	unsigned int weight4 = (unsigned int)rd_w.getWeight4();
	  
        	w.setValues(weight0,weight1,weight2,weight3,weight4);
        	weightMap->setValue(rd_w.getWeightGroupId(),w);
	  
	  	++igroups;
		}
	
		edm::LogInfo("EcalTPGWeightIdMapHandler") << "found " << igroups << "Weight groups";

 		Time_t snc= (Time_t) irun; 	      
 		m_to_transfer.push_back(std::make_pair((EcalTPGWeightIdMap*)weightMap,snc));
	   
	  }
	}
	  
	delete econn;

	edm::LogInfo("EcalTPGWeightIdMapHandler") << "Ecal - > end of getNewObjects -----------";
	
}

