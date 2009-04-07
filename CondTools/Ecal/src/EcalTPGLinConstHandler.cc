#include "CondTools/Ecal/interface/EcalTPGLinConstHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

popcon::EcalTPGLinConstHandler::EcalTPGLinConstHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGLinConstHandler")) {

        edm::LogInfo("EcalTPGLinConstHandler") << "EcalTPGLinConst Source handler constructor";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");

	edm::LogInfo("EcalTPGLinConstHandler") << m_sid <<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;   
}

popcon::EcalTPGLinConstHandler::~EcalTPGLinConstHandler()
{
}


void popcon::EcalTPGLinConstHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGLinConstHandler") << "Started getNewObjects";
        
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
	edm::LogInfo("EcalTPGLinConstHandler") << "max_since = " << max_since;	
	edm::LogInfo("EcalTPGLinConstHandler")<< "Retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
    	edm::LogInfo("EcalTPGLinConstHandler")<< "Retrieving run list from ONLINE DB ... " << endl;

    	edm::LogInfo("EcalTPGLinConstHandler") << "Making connection..." << flush;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    	edm::LogInfo("EcalTPGLinConstHandler") << "Done." << endl;
	
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
	if(m_firstRun<(unsigned int)max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGLinConstHandler") << "min_run=  " << min_run << "max_run = " << max_run;

    	RunList my_list;
    	//my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef);
	my_list=econn->fetchRunList(my_runtag);
      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int mon_runs=run_vec.size();
	edm::LogInfo("EcalTPGLinConstHandler") <<"number of runs is : "<< mon_runs;
        
	unsigned long irun=0; 
	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();
	    edm::LogInfo("EcalTPGLinConstHandler") << "Here is the run number: "<< run_vec[kr].getRunNumber();	  
	    edm::LogInfo("EcalTPGLinConstHandler") << "Fetching run by tag";

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
	    	edm::LogInfo("EcalTPGLinConstHandler")<<"config_tag "<<the_config_tag<<std::endl;
	    	fe_main_info.setConfigTag(the_config_tag);
	    	econn-> fetchConfigSet(&fe_main_info);
	      }
	      edm::LogInfo("EcalTPGLinConstHandler") << "Got " << nr << "objects in dataset.";

              // now get TPGLinConst
              int linId=fe_main_info.getLinId();
	      FEConfigLinInfo fe_lin_info;
	      fe_lin_info.setId(linId);
	      econn-> fetchConfigSet(&fe_lin_info);
	      map<EcalLogicID, FEConfigLinDat> dataset_TpgLin;
	      econn->fetchDataSet(&dataset_TpgLin, &fe_lin_info);

	      EcalTPGLinearizationConst *linC = new EcalTPGLinearizationConst;
              typedef map<EcalLogicID, FEConfigLinDat>::const_iterator CIfelin;
	      EcalLogicID ecid_xt;
	      FEConfigLinDat  rd_lin;
	      int icells=0;
	
	      for (CIfelin p = dataset_TpgLin.begin(); p != dataset_TpgLin.end(); p++) 
	      {
	        ecid_xt = p->first;
	  	rd_lin  = p->second;
	  	std::string ecid_name=ecid_xt.getName();
	  
	  	//EB data
	  	if (ecid_name=="EB_crystal_number") 
		{
	  	  int sm_num=ecid_xt.getID1();
	          int xt_num=ecid_xt.getID2();
	  	  EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);

          	  EcalTPGLinearizationConst::Item item;
	  	  item.mult_x1   =rd_lin.getMultX1() ;
	  	  item.mult_x6   =rd_lin.getMultX6();
	  	  item.mult_x12  =rd_lin.getMultX12();
	  	  item.shift_x1  =rd_lin.getShift1() ;
	  	  item.shift_x6  =rd_lin.getShift6();
	  	  item.shift_x12 =rd_lin.getShift12();
          	  
	  	  linC->insert(std::make_pair(ebdetid.rawId(),item));
	  	  ++icells;
		} 
		else 
		{
		//EE data
	  	  int z=ecid_xt.getID1();
	  	  int x=ecid_xt.getID2();
	  	  int y=ecid_xt.getID3();
	  	  EEDetId eedetid(x,y,z,EEDetId::SCCRYSTALMODE);
	  
	  	  EcalTPGLinearizationConst::Item item;
	  
	  	  item.mult_x1   =rd_lin.getMultX1() ;
	  	  item.mult_x6   =rd_lin.getMultX6();
	  	  item.mult_x12  =rd_lin.getMultX12();
	  	  item.shift_x1  =rd_lin.getShift1() ;
	  	  item.shift_x6  =rd_lin.getShift6();
	  	  item.shift_x12 =rd_lin.getShift12();

	  	  linC->insert(std::make_pair(eedetid.rawId(),item));
	  	  ++icells;
		}
	  } 
	
	  edm::LogInfo("EcalTPGLinConstHandler") << "found " << icells << "crystals.";

 	  Time_t snc= (Time_t) irun ;		
	  m_to_transfer.push_back(std::make_pair((EcalTPGLinearizationConst*)linC,snc));
	      
	  }// while over all runs
	}//if condition on mon_run
	  
	delete econn;
	edm::LogInfo("EcalTPGLinConstHandler") << "Ecal - > end of getNewObjects -----------";	
}

