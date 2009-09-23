#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGLutIdMapHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
//#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
//#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
//#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
//#include "Geometry/EcalMapping/interface/EcalMapingRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

popcon::EcalTPGLutIdMapHandler::EcalTPGLutIdMapHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGLutIdMapHandler")) {

        edm::LogInfo("EcalTPGLutIdMapHandler") << "EcalTPGLutIdMap Source handler constructor";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGLutIdMapHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;


}

popcon::EcalTPGLutIdMapHandler::~EcalTPGLutIdMapHandler()
{
}


void popcon::EcalTPGLutIdMapHandler::getNewObjects()
{
	edm::LogInfo("EcalTPGLutIdMapHandler") << "Started GetNewObjects!!!";

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
	edm::LogInfo("EcalTPGLutIdMapHandler") << "max_since : "  << max_since;
	edm::LogInfo("EcalTPGLutIdMapHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
	edm::LogInfo("EcalTPGLutIdMapHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGLutIdMapHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGLutIdMapHandler") << "Done.";
	
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
	edm::LogInfo("EcalTPGLutIdMapHandler") << "min_run=  " << min_run << "max_run= " << max_run;

	RunList my_list;
    	//my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGLutIdMapHandler") <<"number of Mon runs is : ";
	
	unsigned long irun;
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGLutIdMapHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	  

		edm::LogInfo("EcalTPGLutIdMapHandler") << "Fetching run by tag";

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
	      	  std::cout<<"config_tag "<<the_config_tag<<std::endl;
	      	  fe_main_info.setConfigTag(the_config_tag);
	      	  econn-> fetchConfigSet(&fe_main_info);
	    	}
		edm::LogInfo("EcalTPGLutIdMapHandler") << "got " << nr << "objects in dataset";


        	// now get TPGLutIdMap
        	int lutId=fe_main_info.getLutId();
		FEConfigLUTInfo fe_lut_info;
		fe_lut_info.setId(lutId);
		econn-> fetchConfigSet(&fe_lut_info);
       		map<EcalLogicID, FEConfigLUTGroupDat> dataset_TpgLut;
	
		econn->fetchDataSet(&dataset_TpgLut, &fe_lut_info);
		edm::LogInfo("EcalTPGLutIdMapHandler") << "Got object!";
	
		EcalTPGLutIdMap* lutMap = new EcalTPGLutIdMap;
	
		typedef map<EcalLogicID, FEConfigLUTGroupDat>::const_iterator CIfelut;
		EcalLogicID ecid_xt;
		FEConfigLUTGroupDat  rd_lut;
		int igroups=0;
		unsigned int lutArray[1024] ;
	
		for (CIfelut p = dataset_TpgLut.begin(); p != dataset_TpgLut.end(); p++) 
		{
	  	  ecid_xt = p->first;
	  	  rd_lut  = p->second;

		  std::string ecid_name=ecid_xt.getName();
 	  
	  	for (int ilut=0;ilut<1024;++ilut) {
	    	lutArray[ilut]=rd_lut.getLUTValue(ilut);
	  	}
	  
	  	EcalTPGLut mylut;
        	mylut.setLut(lutArray);	
	  	lutMap->setValue(rd_lut.getLUTGroupId(),mylut);
	  	++igroups;
	  
		}
	
		edm::LogInfo("EcalTPGLutIdMapHandler") << "found " << igroups << "Lut groups.";

 	    	Time_t snc= (Time_t) irun ;
 	    	m_to_transfer.push_back(std::make_pair((EcalTPGLutIdMap*)lutMap,snc));
	  }//run
	}//if
	  
	delete econn;

	edm::LogInfo("EcalTPGLutIdMapHandler") << "Ecal - > end of getNewObjects -----------";
	
}

