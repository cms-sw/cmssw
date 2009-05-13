#include "CondTools/Ecal/interface/EcalLaserHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"


#include<iostream>



popcon::EcalLaserHandler::EcalLaserHandler(const edm::ParameterSet & ps) 
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalLaserHandler")) {

	std::cout << "EcalLaser Source handler constructor\n" << std::endl;

	m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
	m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
	m_sid= ps.getParameter<std::string>("OnlineDBSID");
	m_user= ps.getParameter<std::string>("OnlineDBUser");
	m_pass= ps.getParameter<std::string>("OnlineDBPassword");
	m_locationsource= ps.getParameter<std::string>("LocationSource");
	m_location=ps.getParameter<std::string>("Location");
	m_gentag=ps.getParameter<std::string>("GenTag");

	std::cout << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag   << std::endl;
}

popcon::EcalLaserHandler::~EcalLaserHandler()
{
}

void popcon::EcalLaserHandler::getNewObjects()
{
  std::cerr << "------- " << m_name 
	    << " - > getNewObjects" << std::endl;

  std::cout << "------- Ecal - > getNewObjects\n";


  unsigned long long max_since= 1;
  string payloadtoken = "";
  max_since=tagInfo().lastInterval.first;
  std::cout << "max_since : "  << max_since << endl;

  payloadtoken = tagInfo().lastPayloadToken;
  std::cout << " payloadtoken : " << payloadtoken << endl;
  Ref payload= lastPayload();

  std::cout << "retrieved last payload "  << endl;
  

  // we will copy the last valid record to a temporary object apdpns_temp
  EcalLaserAPDPNRatios* apdpns_temp = new EcalLaserAPDPNRatios();
  
  // let's get from offline DB the last valid apdpn object to set apdpn_db
  
  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;


  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = payload->getLaserMap(); 
  std::cout << "payload->getLaserMap():  OK " << endl;
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = payload->getTimeMap(); 
  std::cout << "payload->getTimeMap():  OK " << endl;
  std::cout << "going to access objects in the last payload "  << endl;


  // loop through barrel
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      if (EBDetId::validDetId(iEta,iPhi))
	{
	  EBDetId ebdetid(iEta,iPhi);
	  unsigned int hiee = ebdetid.hashedIndex();    
	  apdpnpair = laserRatiosMap[ebdetid];
	  
	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp; 
	  apdpnpair_temp.p1 = apdpnpair.p1;
	  apdpnpair_temp.p2 = apdpnpair.p2;
	  apdpns_temp->setValue(ebdetid, apdpnpair_temp);
	  if (hiee%1000 == 0 )std::cout <<"hiee = "<< hiee << "    p1 = " << apdpnpair.p1  <<"    p2 = " << apdpnpair.p2  <<endl;
	}
    }
  }
  
  std::cout << "going to access Endcap objects in the last payload "  << endl;

  // loop through ecal endcap      
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      try
	{
	  EEDetId eedetidpos(iX,iY,1);
	  int hi = eedetidpos.hashedIndex();
		
	  if (laserRatiosMap.find(eedetidpos)!=laserRatiosMap.end()){
	    apdpnpair = laserRatiosMap[eedetidpos];
	    EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp; 
	    apdpnpair_temp.p1 = apdpnpair.p1;
	    apdpnpair_temp.p2 = apdpnpair.p2;
	    apdpns_temp->setValue(eedetidpos, apdpnpair_temp);
	    
	  } else {
	    edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << endl;     
	  }
	  
	  EEDetId eedetidneg(iX,iY,-1);
	  hi = eedetidneg.hashedIndex();

	  if (laserRatiosMap.find(eedetidneg)!=laserRatiosMap.end()){
	    apdpnpair = laserRatiosMap[eedetidneg];
	    EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp; 
	    apdpnpair_temp.p1 = apdpnpair.p1;
	    apdpnpair_temp.p2 = apdpnpair.p2;
	    apdpns_temp->setValue(eedetidneg, apdpnpair_temp);
	    
	  } else {
	    edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << endl;     
	  }
	  
	}
      catch (cms::Exception &e) {  }
    }
  }

  unsigned long long start_time_old=0;
  edm::Timestamp t_min= edm::Timestamp(start_time_old);


  //loop through light modules

  for (int i=0; i<92; i++){
    timestamp = laserTimeMap[i];
    
    EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
    timestamp_temp.t1 = timestamp.t1;
    timestamp_temp.t2 = timestamp.t2;
    
    apdpns_temp->setTime(i,timestamp_temp);

    if(t_min<timestamp.t2) t_min=timestamp.t2;
  }
  
  cout <<"WOW: we just retrieved the last valid record from DB "<< endl;
  cout <<"this is the t_min "<< t_min.value()<< endl;




  // here we retrieve all the runs after the last from online DB 
  
  cout << "Retrieving run list from ONLINE DB ... " << endl;
  try {
    cout << "Making connection..." << flush;
    econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    cout << "Done." << endl;
  } catch (runtime_error &e) {
    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
    cerr << e.what() << endl;
    throw cms::Exception("OMDS not available");
  } 
  
  // these are the online conditions DB classes 
  RunList my_runlist ;
  RunTag  my_runtag;
  LocationDef my_locdef;
  RunTypeDef my_rundef;
  
  
  //FC:  provo a commentare qui sotto e a caricare in memoria any run type 
  // my_locdef.setLocation(m_location);
  // my_rundef.setRunType("LASER");
  // my_runtag.setLocationDef(my_locdef);
  // my_runtag.setRunTypeDef(my_rundef);
  // my_runtag.setGeneralTag(m_gentag);
  
  
  // here we retrieve the laser Monitoring Farm  run records
  
  LMFRunTag lmf_tag;
  lmf_tag.setGeneralTag("default");
  LMFRunList lmf_list;
  lmf_list.setLMFRunTag(lmf_tag);
  lmf_list.setRunTag(my_runtag);
  //    mon_list=econn->fetchMonRunList(my_runtag, mon_tag);

  uint64_t t_min_val= (uint64_t)t_min.value();


  lmf_list=econn->fetchLMFRunList(my_runtag, lmf_tag,t_min_val, m_lastRun );
  
  std::vector<LMFRunIOV> lmf_run_vec=  lmf_list.getRuns();
  int lmf_runs=lmf_run_vec.size();
  cout <<"number of LMF runs is : "<< lmf_runs<< endl;

  if(lmf_runs>0){
    
    Time_t snc=lmf_run_vec[0].getSubRunStart().microsTime();
    
    std::vector<int> nsubruns;
    nsubruns.reserve(lmf_runs);
    std::vector<int> last_lmr;
    last_lmr.reserve(lmf_runs);



    for(int kt=0; kt<lmf_runs; kt++){
      last_lmr[kt]=0;
      nsubruns[kt]=lmf_run_vec[kt].getSubRunNumber();
      std::cout<< "nsubr="<<nsubruns[kt]<<endl;
      
      // FC qui bisogna controllare il subrun number 
      // NN00LL0T
      // NN = numero di sequenza 
      // LL = numero di Laser Monitoring Region
      // T = laser type 
      
      int i=lmf_run_vec[kt].getSequenceNumber();
      int i_lmr=lmf_run_vec[kt].getLMRNumber();
      
      if(last_lmr[i-1]<i_lmr) last_lmr[i-1]=i_lmr;
    }


    int old_seq=0;
    for(int kr=0; kr<lmf_runs; kr++){
      
      int i=lmf_run_vec[kr].getSequenceNumber();
      int i_lmr=lmf_run_vec[kr].getLMRNumber();
      Tm time_lmf_subrun=lmf_run_vec[kr].getSubRunStart();
      unsigned long long time_lmf_subrun_micro=time_lmf_subrun.microsTime();
      
      unsigned long long irun=(unsigned long long) lmf_run_vec[kr].getRunIOV().getRunNumber();
      cout << "here is the run number: "<< lmf_run_vec[kr].getRunIOV().getRunNumber() <<"."<<i<<"/"<<nsubruns[kr]<< endl;
      
      if(old_seq!=i) {
	old_seq=i;
	// setting right since for the first subrun 
	snc = time_lmf_subrun_micro;
      }
      
	    
      // retrieve the APDPNs from OMDS for this run 
      
      map<EcalLogicID, LMFLaserPrimDat > dataset_lmf;
      econn->fetchDataSet(&dataset_lmf, &lmf_run_vec[kr]);
      cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_lmf.size() << " records"<< endl;
      
      
      typedef map<EcalLogicID, LMFLaserPrimDat>::const_iterator CIlmf;
      EcalLogicID ecid_xt;
      
      LMFLaserPrimDat rd_apdnorm;
      const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = apdpns_temp->getLaserMap();




      
      for (CIlmf p = dataset_lmf.begin(); p != dataset_lmf.end(); p++) {
	ecid_xt = p->first;
	rd_apdnorm = p->second;
	
	if(ecid_xt.getName()=="EB_crystal_number"){
	  int sm_num=ecid_xt.getID1();
	  int xt_num=ecid_xt.getID2(); 
	  EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
	  unsigned int hiee = ebdetid.hashedIndex();
	  apdpnpair = laserRatiosMap[ebdetid];
	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
	  // p1 in a new object should be equal to the p2 in the previos object
	  apdpnpair_temp.p1 = apdpnpair.p2;
	  apdpnpair_temp.p2 = rd_apdnorm.getAPDOverPNMean();
	  // here we change in the apdpns_temp object only the values that are available in the online DB 
	  // otherwise we keep the old value 
	  apdpns_temp->setValue(ebdetid, apdpnpair_temp);
	} else { 
	  // endcaps case
	  int iz=ecid_xt.getID1();
	  int ix=ecid_xt.getID2(); 
	  int iy=ecid_xt.getID2(); 
	  EEDetId eedetid(ix,iy,iz);
	  unsigned int hiee = eedetid.hashedIndex();
	  apdpnpair = laserRatiosMap[eedetid];
	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
	  // p1 in a new object should be equal to the p2 in the previos object
	  apdpnpair_temp.p1 = apdpnpair.p2;
	  apdpnpair_temp.p2 = rd_apdnorm.getAPDOverPNMean();
	  // here we change in the apdpns_temp object only the values that are available in the online DB 
	  // otherwise we keep the old value 
	  apdpns_temp->setValue(eedetid, apdpnpair_temp);
	}
      }    

      std::cout << " setting timestamp for LMR "<< i_lmr << " to "<<time_lmf_subrun_micro << endl; 
      const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = apdpns_temp->getTimeMap();
      timestamp = laserTimeMap[i_lmr-1];
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
      timestamp_temp.t1 = timestamp.t2;
      timestamp_temp.t2 = edm::Timestamp(time_lmf_subrun_micro);
      apdpns_temp->setTime( (i_lmr-1) , timestamp_temp);
      

    

      if(i_lmr==last_lmr[i-1]){

	// this is the last subrun of a run we save to offline DB 
	
	cout << "Generating popcon record for run " << irun << "..." << flush;
	
	
	// now I copy apdpns_temp in apdpns_popcon and I ship apdpns_popcon to popcon
	// if I use always the same apdpns_temp I always overwrite
	// so I really have to create new objects for each new run
	// popcon deletes everything for me 
	
	EcalLaserAPDPNRatios* apdpns_popcon = new EcalLaserAPDPNRatios();
	
	
	
	
	
	for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	  if(iEta==0) continue;
	  for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	    // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
	    if (EBDetId::validDetId(iEta,iPhi))
	      {
		EBDetId ebdetid(iEta,iPhi);
		unsigned int hiee = ebdetid.hashedIndex();
		
		
		// here I copy the last valid value in the laser object
		if (laserRatiosMap.find(ebdetid)!=laserRatiosMap.end()){
		  
		  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = apdpns_temp->getLaserMap();
		  apdpnpair = laserRatiosMap[ebdetid];
		  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
		  apdpnpair_temp.p1 = apdpnpair.p1;
		  apdpnpair_temp.p2 = apdpnpair.p2;
		  
		  apdpns_popcon->setValue(ebdetid, apdpnpair_temp);
		} else {
		  edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << endl;
		}
		
	      }
	  }
	}
	// loop through ecal endcap      
	for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	  for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
	    try
	      {
		EEDetId eedetidpos(iX,iY,1);
		int hi = eedetidpos.hashedIndex();
		
		if (laserRatiosMap.find(eedetidpos)!=laserRatiosMap.end()){
		  apdpnpair = laserRatiosMap[eedetidpos];
		  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp; 
		  apdpnpair_temp.p1 = apdpnpair.p1;
		  apdpnpair_temp.p2 = apdpnpair.p2;
		  apdpns_popcon->setValue(eedetidpos, apdpnpair_temp);
		  
		} else {
		  edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << endl;     
		}
		
		EEDetId eedetidneg(iX,iY,-1);
		hi = eedetidneg.hashedIndex();
		
		if (laserRatiosMap.find(eedetidneg)!=laserRatiosMap.end()){
		  apdpnpair = laserRatiosMap[eedetidneg];
		  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp; 
		  apdpnpair_temp.p1 = apdpnpair.p1;
		  apdpnpair_temp.p2 = apdpnpair.p2;
		  apdpns_popcon->setValue(eedetidneg, apdpnpair_temp);
		  
		} else {
		  edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << endl;     
		}
		
	      }
	    catch (cms::Exception &e) {  }
	  }
	}
	
	
	for (int j=0; j<92; j++){
	  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = apdpns_temp->getTimeMap();
	  timestamp = laserTimeMap[j];
	  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
	  timestamp_temp.t1 = timestamp.t1;
	  timestamp_temp.t2 = timestamp.t2;
	  apdpns_popcon->setTime(j,timestamp_temp);
	}
	
	//	snc=i;
	m_to_transfer.push_back(std::make_pair(apdpns_popcon,snc));
	
	std::cout <<" snc : "<< snc << endl;
	
      }
      
      
      
      
    }
    
  }  
    
    
  delete econn;
  delete apdpns_temp;  // this is the only one that popcon does not delete 
  std::cout << "Ecal - > end of getNewObjects -----------\n";
	
	
}
