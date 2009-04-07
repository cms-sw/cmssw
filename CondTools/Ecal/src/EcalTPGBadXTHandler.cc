#include "CondTools/Ecal/interface/EcalTPGBadXTHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadXTInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include<iostream>

popcon::EcalTPGBadXTHandler::EcalTPGBadXTHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGBadXTHandler")) {

        edm::LogInfo("EcalTPGBadXTHandler") << "EcalTPGBadXT Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

        edm::LogInfo("EcalTPGBadXTHandler") << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;


}

popcon::EcalTPGBadXTHandler::~EcalTPGBadXTHandler()
{
}


void popcon::EcalTPGBadXTHandler::getNewObjects()
{
        edm::LogInfo("EcalTPGBadXTHandler") << "Started GetNewObjects!!!";

        
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
        edm::LogInfo("EcalTPGBadXTHandler") << "max_since : "  << max_since;
        edm::LogInfo("EcalTPGBadXTHandler")  << "retrieved last payload ";

        // here we retrieve all the runs after the last from online DB 
        edm::LogInfo("EcalTPGBadXTHandler") << "Retrieving run list from ONLINE DB ... ";

        edm::LogInfo("EcalTPGBadXTHandler") << "Making connection...";
        econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
        edm::LogInfo("EcalTPGBadXTHandler")<< "Done.";
        
        if (!econn)
          {
            cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
            //      cerr << e.what() << endl;
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
	edm::LogInfo("EcalTPGBadXTHandler") << "min_run= " << min_run << "max_run= " << max_run;

	RunList my_list; 
	//my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef); 
        my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGBadXTHandler") <<"number of runs is : ";

	unsigned long irun=0;
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){
	    cout << "here we are in run "<<kr<<endl;
	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGBadXTHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();
	  
	    edm::LogInfo("EcalTPGBadXTHandler") << "Fetching run by tag";

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
	    
              //int id1 = ecalid.getID1();
              RunTPGConfigDat  dat = it->second;
              std::string the_config_tag=dat.getConfigTag();
              edm::LogInfo("EcalTPGBadXTHandler")<<"config_tag "<<the_config_tag;
              fe_main_info.setConfigTag(the_config_tag);
              econn-> fetchConfigSet(&fe_main_info);	    
            }
	    
            edm::LogInfo("EcalTPGBadXTHandler") << "got " << nr << "objects in dataset.";
	
	
            // now get TPGBadXT
            int badxtId=fe_main_info.getBxtId();
            FEConfigBadXTInfo fe_badXt_info;
            fe_badXt_info.setId(badxtId);
            econn-> fetchConfigSet(&fe_badXt_info);
            std::vector<FEConfigBadXTDat> dataset_TpgBadXT;
	    econn->fetchConfigDataSet(&dataset_TpgBadXT, &fe_badXt_info);

            // NB new 
	    EcalTPGCrystalStatus* badXt = new EcalTPGCrystalStatus;
            //typedef map<EcalLogicID, FEConfigBadXTDat>::const_iterator CIfeped;
            typedef std::vector<FEConfigBadXTDat>::const_iterator CIfeped;
            EcalLogicID ecid_xt;
	    FEConfigBadXTDat  rd_badXt;
            int icells=0;
	
            for (CIfeped p = dataset_TpgBadXT.begin(); p != dataset_TpgBadXT.end(); p++) {
            rd_badXt = *p;	  
	    std::string ecid_name=ecid_xt.getName();
	  
	    // EB data	    
	    if (ecid_name=="EB_crystal_number") {
	      // get SM id
	      int sm_num=rd_badXt.getSMId();
	      // get crystal id
	      int xt_num=rd_badXt.getXTId();
	    
	      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
            	    
	      badXt->setValue(ebdetid.rawId(),rd_badXt.getStatus());	
	      ++icells;
	    }
	      else if (ecid_name=="EE_crystal_number"){
	      // EE data
	      int z=ecid_xt.getID1();
	      int x=ecid_xt.getID2();
	      int y=ecid_xt.getID3();
	      EEDetId eedetid(x,y,z,EEDetId::SCCRYSTALMODE);
	   	
	      badXt->setValue(eedetid.rawId(),rd_badXt.getStatus());	
	    
	      ++icells;
	    }
          }//end for over data
	  
          edm::LogInfo("EcalTPGBadXTHandler") << "Finished pedestal reading";
	
	  Time_t snc= (Time_t) irun ;                      
	  m_to_transfer.push_back(std::make_pair((EcalTPGCrystalStatus*)badXt,snc));
		
          }//end for over kr (nr of runs)
        }//end if
	
        delete econn;

        edm::LogInfo("EcalTPGBadXTHandler")<< "Ecal - > end of getNewObjects -----------";        
}

