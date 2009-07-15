#include "CondTools/Ecal/interface/EcalTPGBadTTHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadTTInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

#include<iostream>

popcon::EcalTPGBadTTHandler::EcalTPGBadTTHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGBadTTHandler")) {

        edm::LogInfo("EcalTPGBadTTHandler") << "EcalTPGTowerStatus Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGBadTTHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGBadTTHandler::~EcalTPGBadTTHandler()
{
}


void popcon::EcalTPGBadTTHandler::getNewObjects()
{
    	edm::LogInfo("EcalTPGBadTTHandler") << "Started GetNewObjects!!!";

    	int max_since=0;
    	max_since=(int)tagInfo().lastInterval.first;
    	edm::LogInfo("EcalTPGBadTTHandler") << "max_since : "  << max_since;
    	edm::LogInfo("EcalTPGBadTTHandler") << "retrieved last payload ";

    	// here we retrieve all the runs after the last from online DB 
    	edm::LogInfo("EcalTPGBadTTHandler") << "Retrieving run list from ONLINE DB ... ";

    	edm::LogInfo("EcalTPGBadTTHandler") << "Making connection...";
    	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    	edm::LogInfo("EcalTPGBadTTHandler") << "Done.";
        
    	if (!econn)
    	{
      	  cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
      	  throw cms::Exception("OMDS not available");
    	} 
      
    	LocationDef my_locdef;
    	my_locdef.setLocation(m_location); 

    	RunTypeDef my_rundef;
    	my_rundef.setRunType("PEDESTAL"); 

	RunTag  my_runtag;
	my_runtag.setLocationDef( my_locdef );
	my_runtag.setRunTypeDef( my_rundef );
	my_runtag.setGeneralTag(m_gentag); 

	int min_run=0;
	if(m_firstRun<(unsigned int)max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGBadTTHandler") << "min_run= " << min_run << "max_run= " << max_run;

	RunList my_list; 
	//my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef); 
	my_list=econn->fetchRunList(my_runtag);
	      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGBadTTHandler") << "number of runs is : "<< mon_runs;
       
    	std::string str="";
    
	unsigned long irun=0;
	if(mon_runs>0){
	 for(int kr=0; kr<mon_runs; kr++){
	  
	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGBadTTHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGBadTTHandler") << "Fetching run by tag";

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
              edm::LogInfo("EcalTPGBadTTHandler") <<"config_tag "<<the_config_tag;
              fe_main_info.setConfigTag(the_config_tag);
              econn-> fetchConfigSet(&fe_main_info);
            }
        
            edm::LogInfo("EcalTPGBadTTHandler") << "got " << nr << "objects in dataset.";
	
            // now get TPGTowerStatus
           int badttId=fe_main_info.getBttId();
	
           FEConfigBadTTInfo fe_badTT_info;
           fe_badTT_info.setId(badttId);
	
           econn-> fetchConfigSet(&fe_badTT_info);
           std::vector<FEConfigBadTTDat> dataset_TpgBadTT;
	   //econn->fetchDataSet(&dataset_TpgBadTT, &fe_badTT_info);
           // try the method from EcalCondDNInterface 
	   // with 
	   econn->fetchConfigDataSet(&dataset_TpgBadTT, &fe_badTT_info);

	   EcalTPGTowerStatus* towerStatus = new EcalTPGTowerStatus;
           //typedef map<EcalLogicID, FEConfigBadTTDat>::const_iterator CIfeped;
	   typedef std::vector<FEConfigBadTTDat>::const_iterator CIfeped;
           EcalLogicID ecid_xt;
	   FEConfigBadTTDat  rd_badTT;
           int icells=0;

           for (CIfeped p = dataset_TpgBadTT.begin(); p != dataset_TpgBadTT.end(); p++) {
             rd_badTT  = *p;
	     std::string ecid_name=ecid_xt.getName();
	      
	     if(ecid_name=="EB_trigger_tower") {
 	  
	       int tcc_num=rd_badTT.getTCCId();
	       int tt_num=rd_badTT.getTTId();
	        
	       char identTTEB[10];
    	       sprintf(identTTEB,"%d%d", tcc_num, tt_num);
	       str.assign(identTTEB);
	       std::string S="";
	       S.insert(0,identTTEB);
	       
	       int ebTTDetId = 0; 
	       ebTTDetId = atoi(S.c_str());
	       
	       towerStatus->setValue(ebTTDetId,rd_badTT.getStatus());
	    
	       ++icells;
	     }
	     else if (ecid_name=="EE_trigger_tower"){
	       // Check
	       // EE data
	    
	       int tcc_num=rd_badTT.getTCCId();
	       int tt_num=rd_badTT.getTTId();

	       char identTTEE[10];
	       sprintf(identTTEE,"%d%d", tcc_num, tt_num);
	       str.assign(identTTEE);
		
	       std::string S="";
	       S.insert(0,identTTEE);
	       		        
	       int eeTTDetId = 0; 
	       eeTTDetId = atoi(S.c_str());
                 
	       towerStatus->setValue(eeTTDetId,rd_badTT.getStatus());
	    
	      ++icells;
		
	    }
          }


        edm::LogInfo("EcalTPGBadTTHandler") << "Finished pedestal reading.";
	
	Time_t snc= (Time_t) irun ;                      

	m_to_transfer.push_back(std::make_pair((EcalTPGTowerStatus*)towerStatus,snc));
		
          }
        }
	
	delete econn;

  	edm::LogInfo("EcalTPGBadTTHandler") << "Ecal - > end of getNewObjects -----------";        
}

