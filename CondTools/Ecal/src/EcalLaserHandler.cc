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
bool popcon::EcalLaserHandler::checkAPDPN(float x, float old_x)
{
  bool result=true;
  if(x<=0 || x>20) result=false;
  if(old_x!=1.000 && abs(x-old_x)/old_x>100.00 ) result=false; 
  return result;
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
	  apdpnpair_temp.p3 = apdpnpair.p3;
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
	    apdpnpair_temp.p3 = apdpnpair.p3;
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
	    apdpnpair_temp.p3 = apdpnpair.p3;
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
    timestamp_temp.t3 = timestamp.t3;
    
    apdpns_temp->setTime(i,timestamp_temp);

    if(t_min<timestamp.t3   ) t_min=timestamp.t3;
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

  //  uint64_t t_min_val= (uint64_t)t_min.value();
  uint64_t t_min_val= (t_min.value()>>32 )*1000000;
  

  lmf_list=econn->fetchLMFRunList(my_runtag, lmf_tag,t_min_val,m_firstRun,  m_lastRun );
  
  std::vector<LMFRunIOV> lmf_run_vec=  lmf_list.getRuns();
  int lmf_runs=lmf_run_vec.size();
  cout <<"number of LMF runs is : "<< lmf_runs<< endl;

  if(lmf_runs>0){
    
    Time_t snc=lmf_run_vec[0].getSubRunStart().microsTime();
    Time_t snc_old=lmf_run_vec[0].getSubRunStart().microsTime();
    
    std::vector<int> nsubruns;
    nsubruns.reserve(lmf_runs);
    std::vector<int> last_lmr;
    last_lmr.reserve(lmf_runs);


    // get the Ecal Logic Ids of the ECAL LMR and their crystals
    vector<EcalLogicID> crystals_EB  = econn->getEcalLogicIDSetOrdered( "EB_crystal_number",
									1,36,1,1700,
									EcalLogicID::NULLID,EcalLogicID::NULLID,
									"ECAL_LMR", 4 );
    vector<EcalLogicID> crystals_EE  = econn->getEcalLogicIDSetOrdered( "EE_crystal_number",
									-1,1,-120,120,
									-120,120,
									"ECAL_LMR", 4 );
    /*
    std::map<int ,vector<EcalLogicID> > crystals_by_LMR;

    for(int ilmr=0; ilmr<72; ilmr++){
      vector<EcalLogicID> temp;
      for (int ixt=0; ixt<61200; ixt++){
	if(crystals_EB[ixt].getLogicID()%100 == ilmr+1){
	  temp.push_back(crystals_EB[ixt]);
	}
      }
      std::pair<int ,EcalLogicID> tp=make_pair(ilmr+1,temp) ;
      crystals_by_LMR.insert(tp);
    }
    for(int ilmr=72; ilmr<92; ilmr++){
      vector<EcalLogicID> temp;
      for (int ixt=0; ixt<61200; ixt++){
	if(crystals_EE[ixt].getLogicID()%100 == ilmr+1){
	  temp.push_back(crystals_EE[ixt]);
	}
      }

      std::pair<int ,EcalLogicID> tp=make_pair(ilmr+1,temp) ;
      crystals_by_LMR.insert(tp);
    }
    // done with Ecal Logic Id and LMR

    */

    for(int kt=0; kt<lmf_runs; kt++){
      last_lmr[kt]=0;
    }


    int old_seq=0;
    unsigned long long irun_old=0;
    int last_lmr_x=0;
    int run_saved=0;


    for(int kt=0; kt<lmf_runs; kt++){
      nsubruns[kt]=lmf_run_vec[kt].getSubRunNumber();
      //  std::cout<< "nsubr="<<nsubruns[kt]<<endl;
      int iseq=lmf_run_vec[kt].getSequenceNumber();
      int i_lmr=lmf_run_vec[kt].getLMRNumber();
      unsigned long long irun=(unsigned long long) lmf_run_vec[kt].getRunIOV().getRunNumber();

      if(old_seq!=iseq || irun!=irun_old) {
	// we enter here at the first sub run of a sequence
	// we can do some initialization
	old_seq=iseq;
	irun_old=irun;
	if(kt !=0) {
	  last_lmr[kt-1]=last_lmr_x;
	}
	last_lmr_x=0;
      }
      if(last_lmr_x<= i_lmr) last_lmr_x=i_lmr;
    }

    last_lmr[lmf_runs-1]=last_lmr_x;



    std::vector<int> updated_channels;
    updated_channels.reserve(75848);
    std::vector<int> updated_lmr;
    updated_lmr.reserve(92);

    old_seq=0;
    irun_old=0;
    for(int kr=0; kr<lmf_runs; kr++){
      
      int i=lmf_run_vec[kr].getSequenceNumber();
      int iseq=lmf_run_vec[kr].getSequenceNumber();
      int i_lmr=lmf_run_vec[kr].getLMRNumber();
      Tm time_lmf_subrun=lmf_run_vec[kr].getSubRunStart();
      unsigned long long time_lmf_subrun_micro=time_lmf_subrun.microsTime();
      

      time_lmf_subrun_micro=time_lmf_subrun_micro/1000000;
      time_lmf_subrun_micro=time_lmf_subrun_micro << 32;



      unsigned long long irun=(unsigned long long) lmf_run_vec[kr].getRunIOV().getRunNumber();
      
      if(old_seq!=iseq || irun!=irun_old) {
	// we enter here at the first sub run of a sequence 
	// we can do some initialization
	old_seq=iseq;
	irun_old=irun;
	// setting right since for the first subrun 

	snc_old = snc;

	snc = time_lmf_subrun_micro;

	for(int id=0; id<75848; id++){
	  updated_channels[id] =0;
	}
	for(int id=0; id<92; id++){
	  updated_lmr[id] =0;
	}
      }
      
	    
      // retrieve the APDPNs from OMDS for this run 
      
      map<EcalLogicID, LMFLaserPrimDat > dataset_lmf;
      econn->fetchDataSet(&dataset_lmf, &lmf_run_vec[kr]);
      // cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_lmf.size() << " records"<< endl;
      
      
      typedef map<EcalLogicID, LMFLaserPrimDat>::const_iterator CIlmf;
      EcalLogicID ecid_xt;
      
      LMFLaserPrimDat rd_apdnorm;
      const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = apdpns_temp->getLaserMap();




      int ich=0;
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
	  float x=rd_apdnorm.getAPDOverPNMean();
	  float old_x=apdpnpair.p3;
	  if(checkAPDPN(x,old_x)){
	    apdpnpair_temp.p1 = apdpnpair.p2;
	    apdpnpair_temp.p2 = apdpnpair.p3;
	    apdpnpair_temp.p3 = x;
	  // here we change in the apdpns_temp object only the values that are available in the online DB 
	  // otherwise we keep the old value 
	    apdpns_temp->setValue(ebdetid, apdpnpair_temp);
	    updated_channels[hiee]=1;
	    ich++;
	    //    if (ich<2) std::cout<< "updating channel "<< x<<endl;
	  } else {
	    // FC here we must decide what to do.
	    apdpnpair_temp.p1 = apdpnpair.p2;
	    apdpnpair_temp.p2 = apdpnpair.p3;
	    apdpnpair_temp.p3 = apdpnpair.p3;
            apdpns_temp->setValue(ebdetid, apdpnpair_temp);
	    updated_channels[hiee]=2; // 2 means channel was bad and we propagate the old value 
	    //std::cout<< "NOT updating channel "<< x<<endl;
	  }

	} else { 
	  // endcaps case
	  int iz=ecid_xt.getID1();
	  int ix=ecid_xt.getID2(); 
	  int iy=ecid_xt.getID3(); 
	  EEDetId eedetid(ix,iy,iz);

	  apdpnpair = laserRatiosMap[eedetid];
	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
	  // p1 in a new object should be equal to the p2 in the previos object
	  apdpnpair_temp.p1 = apdpnpair.p2;
	  apdpnpair_temp.p2 = apdpnpair.p3;
	  apdpnpair_temp.p3 = rd_apdnorm.getAPDOverPNMean();
	  // here we change in the apdpns_temp object only the values that are available in the online DB 
	  // otherwise we keep the old value 
	  apdpns_temp->setValue(eedetid, apdpnpair_temp);
	  
	  unsigned int hiee = eedetid.hashedIndex();
	  updated_channels[hiee+61200]=1;
	}
      }    

      if(i_lmr==1 || i_lmr == last_lmr[kr] ) std::cout << "timestamp for run"<<lmf_run_vec[kr].getRunIOV().getRunNumber() <<"."<<iseq<<"/"<<last_lmr[kr] <<" LMR "<< i_lmr << " to "<<time_lmf_subrun_micro << endl; 
      const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = apdpns_temp->getTimeMap();
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp = laserTimeMap[i_lmr-1];
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
      timestamp_temp.t1 = timestamp.t2;
      timestamp_temp.t2 = timestamp.t3;
      timestamp_temp.t3 = edm::Timestamp(time_lmf_subrun_micro);
      apdpns_temp->setTime( (i_lmr-1) , timestamp_temp);
      updated_lmr[i_lmr-1]=1;
      // missing a part that checks for each LMR the 
      // corresponding crystals and eventually 
      // extends the values to the new interval

      //      vector<EcalLogicID> this_group=  crystals_by_LMR.find(i_lmr);
      
    

      if(i_lmr==last_lmr[kr]){

	// this is the last subrun of a run we save to offline DB 
	
	cout << "Generating popcon record for run " << irun << "..." << endl;
	
	
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
		  apdpnpair_temp.p3 = apdpnpair.p3;
		  
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
		  apdpnpair_temp.p3 = apdpnpair.p3;
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
		  apdpnpair_temp.p3 = apdpnpair.p3;
		  apdpns_popcon->setValue(eedetidneg, apdpnpair_temp);
		  
		} else {
		  edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << endl;     
		}
		
	      }
	    catch (cms::Exception &e) {  }
	  }
	}
	



	unsigned long long start_time_late=0ULL;
	Time_t t_late= start_time_late;
	unsigned long long start_time_early=18446744073709551615ULL;
	Time_t t_early= start_time_early;
	
	
	for (int j=0; j<92; j++){
	  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = apdpns_temp->getTimeMap();
	  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestampx = laserTimeMap[j];
	  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
	  timestamp_temp.t1 = timestampx.t1;
	  timestamp_temp.t2 = timestampx.t2;
	  timestamp_temp.t3 = timestampx.t3;
	  apdpns_popcon->setTime(j,timestamp_temp);

	  if(timestamp_temp.t2.value()>t_late    ) t_late=timestamp_temp.t2.value();
    
	}


	uint64_t t_late_val= (t_late >>32 ); // timestamp in seconds

	for (int j=0; j<92; j++){
	  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = apdpns_popcon->getTimeMap();
	  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp = laserTimeMap[j];

	  uint64_t t_val= (timestamp_temp.t2.value()>>32 ); // timestamp in seconds
	  
	  if(timestamp_temp.t2.value()<t_early && updated_lmr[j]==1 && (t_late_val-t_val)<1400 ) 
				  t_early=timestamp_temp.t2.value();
    
	}

	

	std::cout <<" t_early : "<< t_early << endl;
	std::cout <<" snc_old : "<< snc_old << endl;
	std::cout <<" t_late= : "<< t_late << endl;


	if(t_early<=1) t_early=2;

	if(run_saved!=0) {
	  t_early=snc_old; 
	}

	std::cout <<" chosen : "<< t_early << endl;

	m_to_transfer.push_back(std::make_pair(apdpns_popcon,t_early));
	run_saved++; 
	
      }
      
      
      
      
    }
    
  }  
    
    
  delete econn;
  delete apdpns_temp;  // this is the only one that popcon does not delete 
  std::cout << "Ecal - > end of getNewObjects -----------\n";
	
	
}
