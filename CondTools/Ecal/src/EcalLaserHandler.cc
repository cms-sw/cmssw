#include "CondTools/Ecal/interface/EcalLaserHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>

popcon::EcalLaserHandler::EcalLaserHandler(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalLaserHandler")) {
  
  std::cout << "EcalLaser Source handler constructor\n" << std::endl;
  
  m_firstRun=static_cast<unsigned long>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun=static_cast<unsigned long>(atoi(ps.getParameter<std::string>("lastRun").c_str()));
  m_sid= ps.getParameter<std::string>("OnlineDBSID");
  m_user= ps.getParameter<std::string>("OnlineDBUser");
  m_pass= ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource= ps.getParameter<std::string>("LocationSource");
  m_location=ps.getParameter<std::string>("Location");
  m_gentag=ps.getParameter<std::string>("GenTag");
  m_debug=ps.getParameter<bool>("debug");
  
  std::cout << "Starting O2O process on DB: " << m_sid
	    << " User: "<< m_user << "Location: " << m_location 
	    << " Tag: " << m_gentag << std::endl;
}

popcon::EcalLaserHandler::~EcalLaserHandler()
{
  // do nothing
}

/*
bool popcon::EcalLaserHandler::checkAPDPN(float x, float old_x)
{
  bool result=true;
  if(x<=0 || x>20) result=false;
  if((old_x!=1.000 && old_x!=0) && std::abs(x-old_x)/old_x>100.00) result=false; 
  return result;
}
*/

void popcon::EcalLaserHandler::getNewObjects()
{
  std::cerr << "------- " << m_name 
	    << " ---> getNewObjects" << std::endl;
  
  std::cout << "------- Ecal -> getNewObjects\n";
  
  
  unsigned long long max_since= 1;
  std::string payloadtoken = "";
  
  // here popcon tells us which is the last since of the last object in the 
  // offline DB
  max_since=tagInfo().lastInterval.first;
  std::cout << "Last Object in Offline DB has SINCE = "  << max_since 
	    << std::endl;
  
  // connect to the database 
  try {
    std::cout << "Making connection..." << std::flush;
    econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
    std::cout << "Done." << std::endl;
  } catch (std::runtime_error &e) {
    std::cout << " connection parameters " << m_sid << "/" << m_user;
    if (m_debug) {
      std::cout << "/" << m_pass <<std::endl;
    } else {
      std::cout << "/**********" <<std::endl;
    }
    std::cerr << e.what() << std::endl;
    throw cms::Exception("OMDS not available");
  } 

  // retrieve the lists of logic_ids, to build the detids
  std::vector<EcalLogicID> crystals_EB  = 
    econn->getEcalLogicIDSetOrdered( "EB_crystal_angle",
				     -85,85,1,360,
				     EcalLogicID::NULLID,EcalLogicID::NULLID,
				     "EB_crystal_number", 4 );
  std::vector<EcalLogicID> crystals_EE  = 
    econn->getEcalLogicIDSetOrdered( "EE_crystal_number",
				     -1,1,1,100,
				     1,100,
				     "EE_crystal_number", 4 );
  
  std::vector<EcalLogicID>::const_iterator ieb = crystals_EB.begin();
  std::vector<EcalLogicID>::const_iterator eeb = crystals_EB.end();

  std::cout << "Got list of " << crystals_EB.size() << " crystals in EB" 
	    << std::endl;
  std::cout << "Got list of " << crystals_EE.size() << " crystals in EE" 
	    << std::endl;
  // loop through barrel
  int count = 0;
  // prepare a map to associate EB logic id's to detids
  std::map<int, int> detids;
  while (ieb != eeb) {
    int iEta = ieb->getID1();
    int iPhi = ieb->getID2();
    count++;
    EBDetId ebdetid(iEta,iPhi);
    unsigned int hieb = ebdetid.hashedIndex();    
    detids[ieb->getLogicID()] = hieb;
    ieb++;
  }
  std::cout << "Validated " << count << " logic ID's for EB" << std::endl;
  
  // do the same for EE
  std::cout << "going to access Endcap objects in the last payload "  << 
    std::endl;
  
  std::vector<EcalLogicID>::const_iterator iee = crystals_EE.begin();
  std::vector<EcalLogicID>::const_iterator eee = crystals_EE.end();

  count = 0;
  while (iee != eee) {
    int iSide = iee->getID1();
    int iX    = iee->getID2();
    int iY    = iee->getID3();
    EEDetId eedetidpos(iX,iY,iSide);
    int hi = eedetidpos.hashedIndex();
    detids[iee->getLogicID()] = hi;
    count ++;
    iee++;
  }
  std::cout << "Validated " << count << " logic ID's for EE" << std::endl;

  // get association between ecal logic id and LMR
  std::map<int, int> logicId2Lmr = econn->getEcalLogicID2LmrMap();

  std::cout << "Retrieving corrections from ONLINE DB ... " << std::endl;

  LMFCorrCoefDat data(econn);
  Tm tmin;
  tmin.setToMicrosTime(max_since); // is max_since in the right format? 
  // get all data in the database taken after the last available time in ORCOFF 
  data.fetchAfter(tmin);
  std::cout << "Got data from online DB" << std::endl << std::flush;

  // retrieve a map from the database. Map index is the SEQ_ID. To each sequence
  // we associate another map, whose key is the crystal ID and whose value is a
  // sextuple (p1, p2, p3, t1, t2, t3)
  std::map<int, std::map<int, LMFSextuple> > d = data.getParameters();
  // sice must be equal to the number of different SEQ_ID's found
  std::cout << "Got " << d.size() << " groups of data" << std::endl;
  // iterate over sequences
  std::map<int, std::map<int, LMFSextuple> >::const_iterator iseq = d.begin();
  std::map<int, std::map<int, LMFSextuple> >::const_iterator eseq = d.end();
  while (iseq != eseq) {
    std::cout << "SEQ_ID: " << iseq->first << std::endl;
    std::cout << "Contains " << iseq->second.size() << " crystals" << std::endl;
    // iterate over crystals
    std::map<int, LMFSextuple>::const_iterator is = iseq->second.begin();
    std::map<int, LMFSextuple>::const_iterator es = iseq->second.end();
    int c = 0;
    EcalLaserAPDPNRatios* apdpns_popcon = new EcalLaserAPDPNRatios();               
    while (is != es) {
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair_temp;
      apdpnpair_temp.p1 = is->second.p[0];
      apdpnpair_temp.p2 = is->second.p[1];
      apdpnpair_temp.p3 = is->second.p[2];
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp_temp;
      timestamp_temp.t1 = edm::Timestamp(is->second.t[0].microsTime());
      timestamp_temp.t2 = edm::Timestamp(is->second.t[1].microsTime());
      timestamp_temp.t3 = edm::Timestamp(is->second.t[2].microsTime());
      apdpns_popcon->setValue(detids[is->first], apdpnpair_temp);
      apdpns_popcon->setTime( logicId2Lmr[is->first] , timestamp_temp);
      if (c++ % 1700 == 0) {
	// debug output
	std::cout << "XTAL: " << is->first << " brings the following data: " << std::endl;
	for (int i = 0; i < 3; i++) {
	  std::cout << "      T" << (i + 1) << ": " << is->second.t[i].str();
	  std::cout << " C" << (i + 1) << ": " << is->second.p[i] << std::endl;
	}
      }
      is++;
    }
    Tm t_now;
    t_now.setToCurrentLocalTime();
    
    Time_t t_early = t_now.microsTime();
    m_to_transfer.push_back(std::make_pair(apdpns_popcon,t_early));
    iseq++;
  }
  /*
  std::map<int, std::list<std::vector<float> > >::const_iterator id =
    d.begin();
  std::map<int, std::list<std::vector<float> > >::const_iterator ed =
    d.end();
  std::cout << "Looping on data in online DB" << std::endl << std::flush;
  int c = 1;
  while (id != ed) {
    std::cout << c++ << " ID: " << id->first << std::endl;
    std::list<std::vector<float> > listOfTriplets = id->second;
    std::list<std::vector<float> >::const_iterator ilist = 
      listOfTriplets.begin(); 
    std::list<std::vector<float> >::const_iterator elist = 
      listOfTriplets.end(); 
    EcalLaserAPDPNRatios* apdpns_popcon = new EcalLaserAPDPNRatios();
    while (ilist != elist) {
      std::vector<float> v = *ilist;
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair ctriplet;
      ctriplet.p1 = v[3];
      ctriplet.p2 = v[4];
      ctriplet.p3 = v[5];
      EcalLaserAPDPNRatios::EcalLaserTimeStamp ttriplet;
      ttriplet.t1 = EcalLaserAPDPNRatios::EcalLaserTimeStamp(v[0]);
      ttriplet.t2 = EcalLaserAPDPNRatios::EcalLaserTimeStamp(v[1]);
      ttriplet.t3 = EcalLaserAPDPNRatios::EcalLaserTimeStamp(v[2]);
      int hashedIndex = detids[id->first];
      apdpns_popcon->setValue(hashedIndex, ctriplet);
      // lmrindex deve essere uguale al LMR
      int lmrIndex = 1;
      apdpns_popcon->setTime(lmrIndex, ttriplet);
      ilist++;
      }
      Tm t_now;
      t_now.setToCurrentLocalTime();
      
      Time_t t_early = t_now.microsTime();
    m_to_transfer.push_back(std::make_pair(apdpns_popcon,t_early));
    id++;
  }
	  */
  std::cout << "END OF LOOP" << std::endl << std::flush;
  /*
  
  // these are the online conditions DB classes 
  RunList my_runlist ;
  RunTag  my_runtag;
  LocationDef my_locdef;
  RunTypeDef my_rundef;
    

  my_locdef.setLocation(m_location);
  my_rundef.setRunType("COSMIC");
  my_runtag.setLocationDef(my_locdef);
  my_runtag.setRunTypeDef(my_rundef);
  my_runtag.setGeneralTag(m_gentag);
  
  // here we retrieve the laser Monitoring Farm  run records
  
  LMFRunTag lmf_tag;
  lmf_tag.setGeneralTag("default");
  LMFRunList lmf_list;
  lmf_list.setLMFRunTag(lmf_tag);
  lmf_list.setRunTag(my_runtag);

  //  uint64_t t_min_val= (uint64_t)t_min.value();
  uint64_t t_min_val= (t_min.value()>>32 )*1000000;
  

  lmf_list=econn->fetchLMFRunList(my_runtag, lmf_tag,t_min_val,m_firstRun,  m_lastRun );
  
  std::vector<LMFRunIOV> lmf_run_vec=  lmf_list.getRuns();
  int lmf_runs=lmf_run_vec.size();
  std::cout <<"number of LMF runs is : "<< lmf_runs<< std::endl;

  if(lmf_runs>0){


    // we determine the first since 

    
    LMFRunList onlineSequences ;
    onlineSequences.setLMFRunTag(lmf_tag);
    onlineSequences.setRunTag(my_runtag);
    uint64_t t_min_valq= t_min_val;

    onlineSequences=econn->fetchLMFRunListLastNRunsBefore(my_runtag, lmf_tag,t_min_valq,92 );
  
    std::vector<LMFRunIOV> onlineSeq_vec=  onlineSequences.getRuns();
    int nOnlineSeq= onlineSeq_vec.size();
    
    unsigned long long the_zero_since=0;
    
    int oseq_first=onlineSeq_vec[0].getSequenceNumber();
    unsigned long long orun_first=static_cast<unsigned long long>(onlineSeq_vec[0].getRunIOV().getRunNumber());
      


    int n_taken=0;
    if(nOnlineSeq>0){
      for (int n=0; n< nOnlineSeq ; n++){ 
	if( onlineSeq_vec[n].getSequenceNumber()==oseq_first && 
	    onlineSeq_vec[n].getRunIOV().getRunNumber()== (int)orun_first &&  onlineSeq_vec[n].getSequenceNumber()!=0  ){
	  
	  the_zero_since=onlineSeq_vec[n].getSubRunStart().microsTime();
	  the_zero_since=the_zero_since/1000000;
	  the_zero_since=the_zero_since << 32;
	  n_taken=n;
	  std::cout<<n<< " previous sequence = "<<onlineSeq_vec[n_taken].getSequenceNumber()<< "previous run="
		   <<onlineSeq_vec[n_taken].getRunIOV().getRunNumber()<<" lmr number ="<<onlineSeq_vec[n_taken].getLMRNumber()
		   <<"since="<<the_zero_since<<std::endl;
	}
      } 
      std::cout<< "*** previous sequence = "<<onlineSeq_vec[n_taken].getSequenceNumber()<< "previous run="
	       <<onlineSeq_vec[n_taken].getRunIOV().getRunNumber()<<" lmr number ="<<onlineSeq_vec[n_taken].getLMRNumber()<<std::endl;
    } else {
      the_zero_since=0;
    }

    
    Time_t snc=lmf_run_vec[0].getSubRunStart().microsTime();
    Time_t snc_old=lmf_run_vec[0].getSubRunStart().microsTime();
    
    std::vector<int> nsubruns;
    nsubruns.reserve(lmf_runs);
    std::vector<int> last_lmr;
    last_lmr.reserve(lmf_runs);


    // get the Ecal Logic Ids of the ECAL LMR and their crystals
    std::vector<EcalLogicID> crystals_EB  = econn->getEcalLogicIDSetOrdered( "EB_crystal_number",
									1,36,1,1700,
									EcalLogicID::NULLID,EcalLogicID::NULLID,
									"ECAL_LMR", 4 );
    std::vector<EcalLogicID> crystals_EE  = econn->getEcalLogicIDSetOrdered( "EE_crystal_number",
									-1,1,-120,120,
									-120,120,
									"ECAL_LMR", 4 );
 
    for(int kt=0; kt<lmf_runs; kt++){
      last_lmr[kt]=0;
    }


    int old_seq=0;
    unsigned long long irun_old=0;
    int last_lmr_x=0;
    int run_saved=0;


    for(int kt=0; kt<lmf_runs; kt++){
      nsubruns[kt]=lmf_run_vec[kt].getSubRunNumber();
      //  std::cout<< "nsubr="<<nsubruns[kt]<<std::endl;
      int iseq=lmf_run_vec[kt].getSequenceNumber();
      int i_lmr=lmf_run_vec[kt].getLMRNumber();
      unsigned long long irun=static_cast<unsigned long long>(lmf_run_vec[kt].getRunIOV().getRunNumber());

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
      
      //      int i=lmf_run_vec[kr].getSequenceNumber();
      int iseq=lmf_run_vec[kr].getSequenceNumber();
      int i_lmr=lmf_run_vec[kr].getLMRNumber();
      Tm time_lmf_subrun=lmf_run_vec[kr].getSubRunStart();
      unsigned long long time_lmf_subrun_micro=time_lmf_subrun.microsTime();
      

      time_lmf_subrun_micro=time_lmf_subrun_micro/1000000;
      time_lmf_subrun_micro=time_lmf_subrun_micro << 32;

      Time_t t_early_old= 0;


      unsigned long long irun=static_cast<unsigned long long>(lmf_run_vec[kr].getRunIOV().getRunNumber());
      
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
      
      std::map<EcalLogicID, LMFLaserPrimDat > dataset_lmf;
      econn->fetchDataSet(&dataset_lmf, &lmf_run_vec[kr]);
      // std::cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_lmf.size() << " records"<< std::endl;
      
      
      typedef std::map<EcalLogicID, LMFLaserPrimDat>::const_iterator CIlmf;
      EcalLogicID ecid_xt;
      
      LMFLaserPrimDat rd_apdnorm;
      const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = apdpns_temp->getLaserMap();




      int ich=0;
      int ich_bad=0;

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
	    if (alot_of_printout & (ich<2)) std::cout<< "updating channel "<< x<<std::endl;
	  } else {
	    // FC here we must decide what to do.
	    ich_bad++;
	    apdpnpair_temp.p1 = apdpnpair.p2;
	    apdpnpair_temp.p2 = apdpnpair.p3;
	    apdpnpair_temp.p3 = apdpnpair.p3;
            apdpns_temp->setValue(ebdetid, apdpnpair_temp);
	    updated_channels[hiee]=2; // 2 means channel was bad and we propagate the old value 
	    if (alot_of_printout & (ich_bad<10)) std::cout<< "NOT updating channel with APD/PN="<< x <<" "<< old_x <<std::endl;
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
       if (alot_of_printout ) std::cout << "ich="<<ich<<" ich_bad="<<ich_bad<<std::endl;

      if(i_lmr==1 || i_lmr == last_lmr[kr] ) std::cout << "timestamp for run"<<lmf_run_vec[kr].getRunIOV().getRunNumber() <<"."<<iseq<<"/"<<last_lmr[kr] <<" LMR "<< i_lmr << " to "<<time_lmf_subrun_micro << std::endl; 
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

      //      std::vector<EcalLogicID> this_group=  crystals_by_LMR.find(i_lmr);
      
    

      if(i_lmr==last_lmr[kr]){

	// this is the last subrun of a run we save to offline DB 
	
	std::cout << "Generating popcon record for run " << irun << "..." << std::endl;
	
	
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
		//unsigned int hiee = ebdetid.hashedIndex();
		
		
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
		  edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << std::endl;
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
	    edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << std::endl;     
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
		  edm::LogError("EcalLaserHandler") << "error with laserRatiosMap!" << std::endl;     
		}
		
	      }
	      catch (cms::Exception &e) {  
	      std::cout << "Exception in Laser Handler: " << e.toString();
	      }
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

	

	if(t_early<=1) t_early=2;

	if(run_saved != 0) {
	  t_early=snc_old; 
	} else if(the_zero_since!=0) {
	  // use the first time determined from omds 
	  t_early=the_zero_since;
	} else {        m_to_transfer.push_back(std::make_pair(apdpns_popcon,t_early));                                                                                    
	  t_early=2;
	}

	if(t_early==t_early_old) t_early=t_early+1; 

	t_early_old=t_early;

	std::cout <<" chosen : "<< t_early << std::endl;

	m_to_transfer.push_back(std::make_pair(apdpns_popcon,t_early));
	run_saved++; 
	
      }
      
      
      
      
    }
    
  }  
   
  */
    
  delete econn;
  //  delete apdpns_temp;  // this is the only one that popcon does not delete 
  std::cout << "Ecal -> end of getNewObjects -----------\n";
	
	
}
