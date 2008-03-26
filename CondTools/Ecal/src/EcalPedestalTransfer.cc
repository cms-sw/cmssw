#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "CondTools/Ecal/interface/EcalPedestalTransfer.h"

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

EcalPedestalTransfer::EcalPedestalTransfer(const edm::ParameterSet& iConfig) :
  m_timetype(iConfig.getParameter<std::string>("timetype")),
  m_cacheIDs(),
  m_records()
{

  std::string container;
  std::string tag;
  std::string record;

  m_firstRun=(unsigned long)atoi( iConfig.getParameter<std::string>("firstRun").c_str());
  m_lastRun=(unsigned long)atoi( iConfig.getParameter<std::string>("lastRun").c_str());


  sid= iConfig.getParameter<std::string>("OnlineDBSID");
  user= iConfig.getParameter<std::string>("OnlineDBUser");
  pass= iConfig.getParameter<std::string>("OnlineDBPassword");
  

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for(Parameters::iterator i = toCopy.begin(); i != toCopy.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert( std::make_pair(container, 0) );
    m_records.insert( std::make_pair(container, record) );
  }
  
}


EcalPedestalTransfer::~EcalPedestalTransfer()
{
  
}

void EcalPedestalTransfer::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
{

  // check if offline DB connection is active
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  
  
  try {
    cout << "Making connection..." << flush;
    econn = new EcalCondDBInterface( sid, user, pass );
    cout << "Done." << endl;
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
    throw cms::Exception("OMDS not available");
  } 
  
    
  // loop on offline DB conditions to be transferred as from config file 
  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;
    
    std::string recordName = m_records[container];
    
    
    // for the moment it only works for the ECAL Pedestals     
    if (container == "EcalPedestals") {
      
      const EcalPedestals* obj;

      
      
    
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
      mon_list=econn->fetchMonRunList(my_runtag, mon_tag,min_run,max_run );
      
      std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();
      cout <<"number of Mon runs is : "<< mon_run_vec.size()<< endl;
      int mon_runs=mon_run_vec.size();
      if(mon_runs>0){



	// this is the offline object 
	EcalPedestals* peds = new EcalPedestals();
	EcalPedestals::Item item;
	


	//	if( !dbOutput->isNewTagRequest(recordName)) {
	  
	  // get from offline DB the last valid pedestal set 
	  edm::ESHandle<EcalPedestals> handle;
	  evtSetup.get<EcalPedestalsRcd>().get(handle);
	  const EcalPedestalsMap& pedMap= handle.product()->getMap(); // map of pedestals

	  EcalPedestalsMapIterator pedIter; // pedestal iterator
	  EcalPedestals::Item aped; // pedestal object for a single xtal
	  
	  
	    for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	      if(iEta==0) continue;
	      for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
		
		EBDetId ebdetid(iEta,iPhi);
		
		pedIter = pedMap.find(ebdetid);
		
		if( pedIter != pedMap.end() ) {
		  aped = (*pedIter);
		} else {
		  edm::LogError("EcalPedestalTransfer") 
		    << "error!! could not find pedestals for channel: Eta/Phi" << iEta <<"/"<< iPhi  << "\n ";
		  continue;
		}

		item.mean_x1  = aped.mean_x1;
		item.rms_x1   = aped.rms_x1;
		item.mean_x6  = aped.mean_x6;
		item.rms_x6   =  aped.rms_x6;
		item.mean_x12 = aped.mean_x12;
		item.rms_x12  =  aped.rms_x12;

		peds->insert(std::make_pair(ebdetid,item));
	      }
	    }

	  cout <<"WOW: we just retrieved the last valid record from DB "<< endl;

	  /*	} else {
	  // it does not exist we create a default one 

	  cout << "this should never happen "  << endl; 

	  } */


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
	    
	    item.mean_x1  =rd_ped.getPedMeanG1() ;
	    item.rms_x1   =rd_ped.getPedRMSG1();
	    item.mean_x6  = rd_ped.getPedMeanG6();
	    item.rms_x6   =rd_ped.getPedRMSG6() ;
	    item.mean_x12 =rd_ped.getPedMeanG12();
	    item.rms_x12  =rd_ped.getPedRMSG12();
	    
	    EBDetId ebdetid(iEta,iPhi);
	    peds->insert(std::make_pair(ebdetid.rawId(),item));
	    
	  }
	    
	  cout <<"Done creating the peds object : pedestal size is: "<< peds->getMap().size() << endl;
	 
	  cout << "Starting offline DB Transaction for run " << irun << "..." << flush;
	  
	  if( dbOutput->isNewTagRequest(recordName)) {
	    // create new
	    std::cout<<"First One "<<std::endl;
	    dbOutput->createNewIOV<const EcalPedestals>( peds, dbOutput->endOfTime() ,recordName);
	  } else {
	    // append
	    std::cout<<"Old One "<<std::endl;
	    dbOutput->appendSinceTime<const EcalPedestals>( peds, irun , recordName);
	  }
	  
	  cout << "... offline DB Transaction done OK for run " << irun << " with tag " << recordName << endl;

	}
	
      }
      
    } else {
      cout << "it does not work yet for " << container << "..." << flush;
      
    }
    
  }


  delete econn;
  
  
}
