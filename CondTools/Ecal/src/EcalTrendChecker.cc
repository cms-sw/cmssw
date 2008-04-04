#include "FWCore/ServiceRegistry/interface/Service.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"



#include "CondTools/Ecal/interface/EcalTrendChecker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/MonRunList.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"

#include <map>
#include <vector>

#include <iostream>
#include <string>
#include <sstream>
#include <time.h>





using namespace std;

EcalTrendChecker::EcalTrendChecker(const edm::ParameterSet& iConfig) 
{

  m_firstRun=(unsigned long)atoi( iConfig.getParameter<std::string>("firstRun").c_str());
  m_lastRun=(unsigned long)atoi( iConfig.getParameter<std::string>("lastRun").c_str());

  sid= iConfig.getParameter<std::string>("OnlineDBSID");
  user= iConfig.getParameter<std::string>("OnlineDBUser");
  pass= iConfig.getParameter<std::string>("OnlineDBPassword");
  

}


EcalTrendChecker::~EcalTrendChecker()
{
  
}

void EcalTrendChecker::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
{


  
  
  try {
    cout << "Making connection to Online DB..." << flush;
    econn = new EcalCondDBInterface( sid, user, pass );
    cout << "Done." << endl;
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
    throw cms::Exception("OMDS not available");
  } 
  
      cout << "Retrieving run list from ONLINE DB ... " << endl;
      
      
      // these are the online conditions DB classes 
      RunList my_runlist ;
      RunTag  my_runtag;
      LocationDef my_locdef;
      RunTypeDef my_rundef;
      
      my_locdef.setLocation("P5_Co");
      my_rundef.setRunType("PEDESTAL");
      my_runtag.setLocationDef(my_locdef);
      my_runtag.setRunTypeDef(my_rundef);
      my_runtag.setGeneralTag("PEDESTAL");
      
      // other methods that may be useful 
      /*
        my_runlist=econn->fetchRunList(my_runtag);
	std::vector<RunIOV> run_vec=  my_runlist.getRuns();
	cout <<"number of runs is : "<< run_vec.size()<< endl;
	int nruns=run_vec.size();
	if(nruns>0){
	cout << "here is first run : "<< run_vec[0].getRunNumber() << endl;
	cout << "here is the run Start for first run ="<< run_vec[0].getRunStart()<< endl;
	cout << "here is the run End for first run =" <<run_vec[0].getRunEnd()<< endl;
	}
	
      */
      
      int sm_num=0;  
      
      // here we retrieve the Monitoring run records
      
      MonVersionDef monverdef;
      monverdef.setMonitoringVersion("test01");
      
      MonRunTag mon_tag;
      mon_tag.setGeneralTag("CMSSW");
      mon_tag.setMonVersionDef(monverdef);
      MonRunList mon_list;
      mon_list.setMonRunTag(mon_tag);
      mon_list.setRunTag(my_runtag);
      //    mon_list=econn->fetchMonRunList(my_runtag, mon_tag);
      int min_run=(int)m_firstRun;
      int max_run=(int)m_lastRun;

      // this retrieves the whole Mon Run list

      mon_list=econn->fetchMonRunList(my_runtag, mon_tag,min_run,max_run );
      
      std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();
      cout <<"number of Mon runs is : "<< mon_run_vec.size()<< endl;
      int mon_runs=mon_run_vec.size();
      if(mon_runs>0){

	for(int kr=0; kr<mon_runs; kr++){
	
	  unsigned long irun=(unsigned long) mon_run_vec[kr].getRunIOV().getRunNumber();
	  
	  cout << "here is first sub run : "<< mon_run_vec[kr].getSubRunNumber() << endl;
	  cout << "here is the run number: "<< mon_run_vec[kr].getRunIOV().getRunNumber() << endl;
	  
	  cout <<" retrieve the data for a given run"<< endl;
	  
	  // retrieve the data for a given run
	  RunIOV runiov_prime = mon_run_vec[kr].getRunIOV();
	  
	  // retrieve the pedestals from OMDS for this run 
	  map<EcalLogicID, MonPedestalsDat> dataset_mon;
	  econn->fetchDataSet(&dataset_mon, &mon_run_vec[kr]);

	    
	  cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_mon.size() << endl;
	     

	  
	  typedef map<EcalLogicID, MonPedestalsDat>::const_iterator CImon;
	  EcalLogicID ecid_xt;
	  MonPedestalsDat  rd_ped;
	  
	  int iEta=0;
	  int iPhi=0;

	  for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
	    ecid_xt = p->first;
	    rd_ped  = p->second;
	    sm_num=ecid_xt.getID1();
	    int xt_num=ecid_xt.getID2()-1; // careful here !!! we number the channels from 0 to 1699
	    
	    iEta=(xt_num/20)+1;
	    iPhi=20-(xt_num-(iEta-1)*20);
	    
	    float mean_x1  =rd_ped.getPedMeanG1() ;
	    float rms_x1   =rd_ped.getPedRMSG1();
	    float mean_x6  = rd_ped.getPedMeanG6();
	    float rms_x6   =rd_ped.getPedRMSG6() ;
	    float mean_x12 =rd_ped.getPedMeanG12();
	    float rms_x12  =rd_ped.getPedRMSG12();
	    // here you can do something with the values ... 
	    cout << mean_x1<< rms_x1<< mean_x6<< rms_x6<< mean_x12<<rms_x12<<endl;
	  }
	    

	}
	
      }
      
    

  delete econn;
  
  
}
