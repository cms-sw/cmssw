#include "CondTools/Ecal/interface/EcalTPGPedestalsHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"

#include<iostream>

popcon::EcalTPGPedestalsHandler::EcalTPGPedestalsHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGPedestalsHandler")) {

	edm::LogInfo("EcalTPGPedestalsHandler") << "EcalTPGPedestals Source handler constructor";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

	edm::LogInfo("EcalTPGPedestalsHandler")<< m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGPedestalsHandler::~EcalTPGPedestalsHandler()
{
}


void popcon::EcalTPGPedestalsHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGPedestalsHandler") << "Started getNewObjects";
	
	//check whats already inside of database
	if (tagInfo().size){
  	//check whats already inside of database
    	std::cout << "got offlineInfo = " << std::endl;
	std::cout << "tag name = " << tagInfo().name << std::endl;
	std::cout << "size = " << tagInfo().size <<  std::endl;
    	} else {
    	std::cout << " First object for this tag " << std::endl;
    	}
		 	
	int max_since =0;
	max_since=(int)tagInfo().lastInterval.first; 
    	edm::LogInfo("EcalTPGPedestalsHandler") << "max_since = " << max_since;    
	edm::LogInfo("EcalTPGPedestalsHandler")<< "Retrieved last payload ";

        // here we retrieve all the runs after the last from online DB 
    	edm::LogInfo("EcalTPGPedestalsHandler")<< "Retrieving run list from ONLINE DB ... " << endl;

    	edm::LogInfo("EcalTPGPedestalsHandler") << "Making connection..." << flush;
    	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    	edm::LogInfo("EcalTPGPedestalsHandler") << "Done." << endl;
        
	if (!econn)
	{
	  cout << " Connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
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
	edm::LogInfo("EcalTPGPedestalsHandler") <<"min_run= " << min_run << " max_run = " << max_run;
	RunList my_list; 
	//my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef); 
    	my_list=econn->fetchRunList(my_runtag); 
       
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	
	edm::LogInfo("EcalTPGPedestalsHandler") <<"number of runs is : "<< mon_runs<< endl;
        
	unsigned long irun=0;
		
	if(mon_runs>0){ 
	
	  for(int kr=0; kr<mon_runs; kr++){
	  
	    irun=(unsigned long) run_vec[kr].getRunNumber();
	   
	    edm::LogInfo("EcalTPGPedestalsHandler")  << "Here is the run number: "<< run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGPedestalsHandler")  << "Fetching run by tag";

            // retrieve the data :
            map<EcalLogicID, RunTPGConfigDat> dataset;
            econn->fetchDataSet(&dataset, &run_vec[kr]);
            std::string the_config_tag="";
            map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;
            FEConfigMainInfo fe_main_info;
	
            int nr=0;
            for ( it=dataset.begin(); it!=dataset.end(); it++ )
            {
              ++nr;
              //EcalLogicID ecalid  = it->first;
           
              RunTPGConfigDat  dat = it->second;
              std::string the_config_tag=dat.getConfigTag();
              edm::LogInfo("EcalTPGPedestalsHandler")<<"config_tag "<<the_config_tag<<std::endl;
              fe_main_info.setConfigTag(the_config_tag);
              econn-> fetchConfigSet(&fe_main_info);	    
            }
        
	    edm::LogInfo("EcalTPGPedestalsHandler")<<"Got " << nr << "objects in the Online dataset.";
	
    	    // now get TPGPedestals
	    int pedId=fe_main_info.getPedId();
	    FEConfigPedInfo fe_ped_info;
    	    fe_ped_info.setId(pedId);
     	    econn-> fetchConfigSet(&fe_ped_info);
     	    map<EcalLogicID, FEConfigPedDat> dataset_TpgPed;
     	    econn->fetchDataSet(&dataset_TpgPed, &fe_ped_info);

	    // NB new 
	    EcalTPGPedestals* peds = new EcalTPGPedestals;
            typedef map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
            EcalLogicID ecid_xt;
            FEConfigPedDat  rd_ped;
    	    int icells=0;
    	    for (CIfeped p = dataset_TpgPed.begin(); p != dataset_TpgPed.end(); p++) 
	    {
	      ecid_xt = p->first;
              rd_ped  = p->second;
	  
	      std::string ecid_name=ecid_xt.getName();
	  
	      // EB data
	      if (ecid_name=="EB_crystal_number") {
	        int sm_num=ecid_xt.getID1();
	        int xt_num=ecid_xt.getID2();
	    
	        EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
	        EcalTPGPedestals::Item item;
	        item.mean_x1  =(unsigned int)rd_ped.getPedMeanG1() ;
	        item.mean_x6  =(unsigned int)rd_ped.getPedMeanG6();
	        item.mean_x12 =(unsigned int)rd_ped.getPedMeanG12();
	    
	        peds->insert(std::make_pair(ebdetid.rawId(),item));
	        ++icells;
	  	} else if (ecid_name=="EE_crystal_number"){
	  
	  	// EE data
	        int z=ecid_xt.getID1();
	        int x=ecid_xt.getID2();
	        int y=ecid_xt.getID3();
	        EEDetId eedetid(x,y,z,EEDetId::SCCRYSTALMODE);
	        EcalTPGPedestals::Item item;
	        item.mean_x1  =(unsigned int)rd_ped.getPedMeanG1();
	        item.mean_x6  =(unsigned int)rd_ped.getPedMeanG6();
	        item.mean_x12 =(unsigned int)rd_ped.getPedMeanG12();
	    
	        peds->insert(std::make_pair(eedetid.rawId(),item));
	        ++icells;
	  	}
	     }

	edm::LogInfo("EcalTPGPedestalsHandler") << "Finished pedestal reading.";
	
	Time_t snc= (Time_t) irun; 
	m_to_transfer.push_back(std::make_pair((EcalTPGPedestals*)peds,snc));
	
          }
        }
	
  	delete econn;
	
	edm::LogInfo("EcalTPGPedestalsHandler")  << "Ecal - > end of getNewObjects -----------";        
}

