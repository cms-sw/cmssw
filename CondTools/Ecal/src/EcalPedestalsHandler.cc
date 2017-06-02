#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalPedestalsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string.h>
#include<iostream>
#include <fstream>

#include "TFile.h"
#include "TTree.h"

const Int_t kChannels = 75848, kEBChannels = 61200, kEEChannels = 14648;

popcon::EcalPedestalsHandler::EcalPedestalsHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalPedestalsHandler")) {

	std::cout << "EcalPedestals Source handler constructor\n" << std::endl;
        m_firstRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
        m_lastRun=static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
        m_runtag=ps.getParameter<std::string>("RunTag");
	m_filename = ps.getUntrackedParameter<std::string>("filename","EcalPedestals.txt");
	m_runtype = ps.getUntrackedParameter<int>("RunType",1);

	std::cout << m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag   << std::endl;


}

popcon::EcalPedestalsHandler::~EcalPedestalsHandler() {}


void popcon::EcalPedestalsHandler::getNewObjects() {
  std::cout << "------- Ecal - > getNewObjects\n";
  if(m_locationsource=="H2") {
    getNewObjectsH2();
  } else if(m_locationsource=="P5") {
    getNewObjectsP5();
  }
  else if(m_locationsource=="File") {
    readPedestalFile();
  }
  else if(m_locationsource=="Tree") {
    readPedestalTree();
  }
  else if(m_locationsource=="Timestamp") {
    readPedestalTimestamp();
  }
  else if(m_locationsource=="2017") {
    readPedestal2017();
  }
  else {
    std::cout << " unknown location " << m_locationsource << " give up " << std::endl;
    exit(-1);
  }
}


bool popcon::EcalPedestalsHandler::checkPedestal( EcalPedestals::Item* item ) {
  // true means all is standard and OK
  bool result=true;
  if(item->rms_x12 >3 || item->rms_x12<=0) result=false; 
  if(item->rms_x6 >2 || item->rms_x6<=0) result=false; 
  if(item->rms_x1 >1 || item->rms_x1<=0) result=false; 
  if(item->mean_x12>300 || item->mean_x12<=100) result=false; 
  if(item->mean_x1>300 || item->mean_x1<=100) result=false; 
  if(item->mean_x6>300 || item->mean_x6<=100) result=false; 
  return result; 
}

void popcon::EcalPedestalsHandler::getNewObjectsP5() {
  std::ostringstream ss; 
  ss<<"ECAL ";

  unsigned int max_since=0;
  max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
  std::cout << "max_since : "  << max_since << std::endl;
  Ref ped_db = lastPayload();

  // we copy the last valid record to a temporary object peds
  EcalPedestals* peds = new EcalPedestals();
  std::cout << "retrieved last payload "  << std::endl;

  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(iEta,iPhi)) {
	EBDetId ebdetid(iEta,iPhi,EBDetId::ETAPHIMODE);
	EcalPedestals::const_iterator it =ped_db->find(ebdetid.rawId());
	EcalPedestals::Item aped = (*it);

	// here I copy the last valid value in the peds object
	EcalPedestals::Item item;
	item.mean_x1  = aped.mean_x1;
	item.rms_x1   = aped.rms_x1;
	item.mean_x6  = aped.mean_x6;
	item.rms_x6   = aped.rms_x6;
	item.mean_x12 = aped.mean_x12;
	item.rms_x12  = aped.rms_x12;

	peds->insert(std::make_pair(ebdetid.rawId(),item));
      }
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1)) {
	EEDetId eedetidpos(iX,iY,1);

	EcalPedestals::const_iterator it =ped_db->find(eedetidpos.rawId());
	EcalPedestals::Item aped = (*it);

	//	unsigned int hiee = eedetidpos.hashedIndex();
	//	EcalPedestals::Item aped= ped_db->endcap(hiee);

	// here I copy the last valid value in the peds object
	EcalPedestals::Item item;
	item.mean_x1  = aped.mean_x1;
	item.rms_x1   = aped.rms_x1;
	item.mean_x6  = aped.mean_x6;
	item.rms_x6   = aped.rms_x6;
	item.mean_x12 = aped.mean_x12;
	item.rms_x12  = aped.rms_x12;
	peds->insert(std::make_pair(eedetidpos.rawId(),item));
      }
      if(EEDetId::validDetId(iX,iY,-1)) {
	EEDetId eedetidneg(iX,iY,-1);

	EcalPedestals::const_iterator it =ped_db->find(eedetidneg.rawId());
	EcalPedestals::Item aped = (*it);
	//     unsigned int hiee = eedetidneg.hashedIndex();
	//     EcalPedestals::Item aped= ped_db->endcap(hiee);

	// here I copy the last valid value in the peds object
	EcalPedestals::Item item;
	item.mean_x1  = aped.mean_x1;
	item.rms_x1   = aped.rms_x1;
	item.mean_x6  = aped.mean_x6;
	item.rms_x6   = aped.rms_x6;
	item.mean_x12 = aped.mean_x12;
	item.rms_x12  = aped.rms_x12;
	peds->insert(std::make_pair(eedetidneg.rawId(),item));
      }
    }
  }

  // here we retrieve all the runs after the last from online DB 

  std::cout << "Retrieving run list from ONLINE DB ... " << std::endl;
  econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
  std::cout << "Connection done" << std::endl;
	
  // these are the online conditions DB classes 
  RunList my_runlist ;
  RunTag  my_runtag;
  LocationDef my_locdef;
  RunTypeDef my_rundef;
	
  my_locdef.setLocation(m_location);
  //	my_rundef.setRunType("PEDESTAL");  // 2017
  my_rundef.setRunType(m_runtag);
  my_runtag.setLocationDef(my_locdef);
  my_runtag.setRunTypeDef(my_rundef);
  my_runtag.setGeneralTag(m_gentag);

  // here we retrieve the Monitoring run records
    
  MonVersionDef monverdef;
  monverdef.setMonitoringVersion("test01");
	
  MonRunTag mon_tag;
  //	mon_tag.setGeneralTag("CMSSW");
  mon_tag.setGeneralTag("CMSSW-offline-private");
  mon_tag.setMonVersionDef(monverdef);
  MonRunList mon_list;
  mon_list.setMonRunTag(mon_tag);
  mon_list.setRunTag(my_runtag);
  //    mon_list=econn->fetchMonRunList(my_runtag, mon_tag);
  unsigned int min_run=0;
  if(m_firstRun<max_since) {
    min_run=max_since+1; // we have to add 1 to the last transferred one
  } else {
    min_run=m_firstRun;
  }

  unsigned int max_run=m_lastRun;
  mon_list=econn->fetchMonRunList(my_runtag, mon_tag,min_run,max_run );
    
  std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();
  int mon_runs = mon_run_vec.size();
  std::cout <<"number of Mon runs is : "<< mon_runs<< std::endl;

  if(mon_runs > 0) {
    int krmax = std::min(mon_runs, 30);
    for(int kr = 0; kr < krmax; kr++){
      std::cout << "-kr------:  "<<kr<<std::endl;

      unsigned int irun=static_cast<unsigned int>(mon_run_vec[kr].getRunIOV().getRunNumber());
      std::cout << "retrieve the data for run number: "<< mon_run_vec[kr].getRunIOV().getRunNumber() << std::endl;
      if (mon_run_vec[kr].getSubRunNumber() <=1){ 

	// retrieve the data for a given run
	RunIOV runiov_prime = mon_run_vec[kr].getRunIOV();
	// retrieve the pedestals from OMDS for this run 
	std::map<EcalLogicID, MonPedestalsDat> dataset_mon;
	econn->fetchDataSet(&dataset_mon, &mon_run_vec[kr]);
	std::cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_mon.size() << std::endl;
	int nEB = 0, nEE = 0, nEBbad = 0, nEEbad =0;
	typedef std::map<EcalLogicID, MonPedestalsDat>::const_iterator CImon;
	EcalLogicID ecid_xt;
	MonPedestalsDat  rd_ped;

	// this to validate ...	      
	int nbad=0;
	for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
	  ecid_xt = p->first;
	  rd_ped  = p->second;
	  int sm_num=ecid_xt.getID1();
	  int xt_num=ecid_xt.getID2(); 
	  int yt_num=ecid_xt.getID3(); 

	  EcalPedestals::Item item;
	  item.mean_x1  =rd_ped.getPedMeanG1() ;
	  item.rms_x1   =rd_ped.getPedRMSG1();
	  item.mean_x6  =rd_ped.getPedMeanG6();
	  item.rms_x6   =rd_ped.getPedRMSG6() ;
	  item.mean_x12 =rd_ped.getPedMeanG12();
	  item.rms_x12  =rd_ped.getPedRMSG12();
	  /*	
	  if(irun != 280216 && irun != 280218 && irun != 280259 && irun != 280261 && irun != 280392 && irun != 280394
	     && irun != 280762 && irun != 280765 && irun != 280936 && irun != 280939 && irun != 281756 && irun != 281757) {
	  */
	  if(ecid_xt.getName()=="EB_crystal_number") {
	    nEB++;
	    if(!checkPedestal(&item) ) nEBbad++;
	  }
	  else {
	    nEE++;
	    if(!checkPedestal(&item) ) nEEbad++;
	  }

	    // here we check and count how many bad channels we have 

	  if(!checkPedestal(&item) ){
	    nbad++;
	    if(nbad < 10) std::cout <<"BAD LIST: channel " << sm_num << "/" << xt_num << "/"<< yt_num 
				    <<  "ped/rms "<<item.mean_x12<< "/"<< item.rms_x12 << std::endl;
	  }
	    /*
	  }
	  else {      // only gain 12 in these runs
	    if(ecid_xt.getName()=="EB_crystal_number") {
	      nEB++;
	      if(item.mean_x12 <= 100 || item.mean_x12 > 300 || item.rms_x12 > 3 || item.rms_x12 <= 0) {
		nbad++;
		nEBbad++;
	      }
	    }
	    else {
	      nEE++;
	      if(item.mean_x12 <= 100 || item.mean_x12 > 300 || item.rms_x12 > 3 || item.rms_x12 <= 0) {
		nbad++;
		nEEbad++;
	      }
	    }
	  }    //    only gain 12 in these runs
	    */
	}   // loop over all channels
    
	// ok or bad? 
	// a bad run is for more than 5% bad channels 

	//	      if(nbad<(dataset_mon.size()*0.1)){
	if(nbad<(dataset_mon.size()*0.05) && (nEB > 10200 || nEE > 2460)) {

	  for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
	    ecid_xt = p->first;
	    rd_ped  = p->second;
	    int sm_num=ecid_xt.getID1();
	    int xt_num=ecid_xt.getID2(); 
	    int yt_num=ecid_xt.getID3(); 

	    EcalPedestals::Item item;
	    item.mean_x1  =rd_ped.getPedMeanG1() ;
	    item.rms_x1   =rd_ped.getPedRMSG1();
	    item.mean_x6  =rd_ped.getPedMeanG6();
	    item.rms_x6   =rd_ped.getPedRMSG6() ;
	    item.mean_x12 =rd_ped.getPedMeanG12();
	    item.rms_x12  =rd_ped.getPedRMSG12();

	    if(ecid_xt.getName()=="EB_crystal_number") {
	      // Barrel channel 
	      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
	    
	      // individual objects check
	      if(item.mean_x1==-1 || item.rms_x1 ==-1 || item.mean_x6==-1 || 
		 item.rms_x6==-1 || item.mean_x12==-1 || item.rms_x12==-1 ||
		 item.mean_x1==0 || item.rms_x1 ==0 || item.mean_x6==0 || 
		 item.rms_x6==0 || item.mean_x12==0 || item.rms_x12==0 ) {
		// if one is bad we 
		// retrieve the old valid value  
		unsigned int hieb = ebdetid.hashedIndex();
		EcalPedestals::Item previous_ped= peds->barrel(hieb);
		if(item.mean_x1==-1  || item.mean_x1==0) item.mean_x1 =previous_ped.mean_x1;
		if(item.rms_x1==-1   || item.rms_x1==0)  item.rms_x1  =previous_ped.rms_x1;
		if(item.mean_x6==-1  || item.mean_x6==0) item.mean_x6 =previous_ped.mean_x6;
		if(item.rms_x6==-1   || item.rms_x6==0)  item.rms_x6  =previous_ped.rms_x6;
		if(item.mean_x12==-1 || item.mean_x12==0)item.mean_x12=previous_ped.mean_x12;
		if(item.rms_x12==-1  || item.rms_x12==0) item.rms_x12 =previous_ped.rms_x12;
	      } 
	    
	      // here we change in the peds object only the channels that are available in the online DB 
	      // otherwise we keep the previous value 	  
	      peds->insert(std::make_pair(ebdetid.rawId(),item)); 
	    } else {
	      // Endcap channel 
	      // in this case xt_num is x 
	      // yt_num is y and sm_num is the side +/- 1 
	      if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
		EEDetId eedetid(xt_num,yt_num,sm_num);

		// individual objects check
		if(item.mean_x1==-1 || item.rms_x1 ==-1 || item.mean_x6==-1 || 
		   item.rms_x6==-1 || item.mean_x12==-1 || item.rms_x12==-1 ||
		   item.mean_x1==0 || item.rms_x1 ==0 || item.mean_x6==0 || 
		   item.rms_x6==0 || item.mean_x12==0 || item.rms_x12==0 ) {
		  // if one is bad we 
		  // retrieve the old valid value  
		  unsigned int hiee = eedetid.hashedIndex();
		  EcalPedestals::Item previous_ped= peds->endcap(hiee);
		  if(item.mean_x1==-1  || item.mean_x1==0) item.mean_x1 =previous_ped.mean_x1;
		  if(item.rms_x1==-1   || item.rms_x1==0)  item.rms_x1  =previous_ped.rms_x1;
		  if(item.mean_x6==-1  || item.mean_x6==0) item.mean_x6 =previous_ped.mean_x6;
		  if(item.rms_x6==-1   || item.rms_x6==0)  item.rms_x6  =previous_ped.rms_x6;
		  if(item.mean_x12==-1 || item.mean_x12==0)item.mean_x12=previous_ped.mean_x12;
		  if(item.rms_x12==-1  || item.rms_x12==0) item.rms_x12 =previous_ped.rms_x12;
		}      
		// here we change in the peds object only the channels that are available in the online DB 
		// otherwise we keep the previous value 
		peds->insert(std::make_pair(eedetid.rawId(),item)); 
	      }	    
	    }	
	  }
    
	  std::cout << "Generating popcon record for run " << irun << "..." << std::flush;	
	  // now I copy peds in pedtemp and I ship pedtemp to popcon
	  // if I use always the same peds I always overwrite
	  // so I really have to create new objects for each new run
	  // popcon deletes everything for me 

	  EcalPedestals* pedtemp = new EcalPedestals();
	  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	    if(iEta==0) continue;
	    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
	      if (EBDetId::validDetId(iEta,iPhi)) {
		EBDetId ebdetid(iEta,iPhi);
		unsigned int hiee = ebdetid.hashedIndex();
		EcalPedestals::Item aped= peds->barrel(hiee);

		// here I copy the last valid value in the peds object
		EcalPedestals::Item item;
		item.mean_x1  = aped.mean_x1;
		item.rms_x1   = aped.rms_x1;
		item.mean_x6  = aped.mean_x6;
		item.rms_x6   = aped.rms_x6;
		item.mean_x12 = aped.mean_x12;
		item.rms_x12  = aped.rms_x12;
		// here I copy the last valid value in the pedtemp object
		pedtemp->insert(std::make_pair(ebdetid.rawId(),item));
		if((iEta==-1 || iEta==1) && iPhi==20){
		  float x=aped.mean_x12 ;
		  std::cout<< "channel:" <<iEta<<"/"<<iPhi<< "/" << hiee << " ped mean 12="<< x << std::endl;
		}
	      }
	    }
	  }
	  // endcaps 
	  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
	      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
	      if (EEDetId::validDetId(iX,iY,1)) {
		EEDetId eedetid(iX,iY,1);
		unsigned int hiee = eedetid.hashedIndex();
		EcalPedestals::Item aped= peds->endcap(hiee);
		// here I copy the last valid value in the peds object
		EcalPedestals::Item item;
		item.mean_x1  = aped.mean_x1;
		item.rms_x1   = aped.rms_x1;
		item.mean_x6  = aped.mean_x6;
		item.rms_x6   = aped.rms_x6;
		item.mean_x12 = aped.mean_x12;
		item.rms_x12  = aped.rms_x12;
		// here I copy the last valid value in the pedtemp object
		pedtemp->insert(std::make_pair(eedetid.rawId(),item));
	      }
	      if (EEDetId::validDetId(iX,iY,-1)) {
		EEDetId eedetid(iX,iY,-1);
		unsigned int hiee = eedetid.hashedIndex();
		EcalPedestals::Item aped= peds->endcap(hiee);
		// here I copy the last valid value in the peds object
		EcalPedestals::Item item;
		item.mean_x1  = aped.mean_x1;
		item.rms_x1   = aped.rms_x1;
		item.mean_x6  = aped.mean_x6;
		item.rms_x6   = aped.rms_x6;
		item.mean_x12 = aped.mean_x12;
		item.rms_x12  = aped.rms_x12;
		// here I copy the last valid value in the pedtemp object
		pedtemp->insert(std::make_pair(eedetid.rawId(),item));
	      }
	    }
	  }

	  Time_t snc= (Time_t) irun ;    	      
	  m_to_transfer.push_back(std::make_pair((EcalPedestals*)pedtemp,snc));

	  ss << "Run=" << irun << "_WAS_GOOD_"<<std::endl; 
	  m_userTextLog = ss.str()+";";
	}   // good run : write in DB
	else {
	  std::cout << "Run " << irun << " was BAD !!!! not sent to the DB";
	  if(nbad >= (dataset_mon.size()*0.05))
	    std::cout << " number of bad channels = " << nbad;
	  if(nEB <= 10200)
	    std::cout << " number of EB channels = " << nEB;
	  if(nEE <= 2440)
	    std::cout << " number of EE channels = " << nEE;
	  std::cout << std::endl;
	  ss << "Run=" << irun << "_WAS_BAD_"<<std::endl; 
	  m_userTextLog = ss.str()+";";
	}  //  bad run : do not write in DB
      }   //  SubRunNumber
    }    // loop over runs
	
    delete econn;
    delete peds;  // this is the only one that popcon does not delete 
  }    // runs to analyze ?
  std::cout << "Ecal - > end of getNewObjects -----------\n";
}


void popcon::EcalPedestalsHandler::getNewObjectsH2() {
	unsigned int max_since=0;
	max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
	std::cout << "max_since : "  << max_since << std::endl;
	Ref ped_db = lastPayload();
	
	std::cout << "retrieved last payload "  << std::endl;


	// we copy the last valid record to a temporary object peds
	EcalPedestals* peds = new EcalPedestals();

	// get from offline DB the last valid pedestal set ped_db
	//	edm::ESHandle<EcalPedestals> pedestals;
	//	esetup.get<EcalPedestalsRcd>().get(pedestals);
	

	// only positive endcap side and 
	// x from 86 to 95
	// y from 46 to 55 
	int ixmin=86;	  int ixmax=95;
	int iymin=46;	  int iymax=55;
	for(int iX=ixmin; iX<=ixmax ;++iX) {
	  for(int iY=iymin; iY<=iymax; ++iY) {

	    if (EEDetId::validDetId(iX,iY,1)) {
	      EEDetId eedetidpos(iX,iY,1);
	      unsigned int hiee = eedetidpos.hashedIndex();
	      EcalPedestals::Item aped= ped_db->endcap(hiee);

	      // here I copy the last valid value in the peds object
	      EcalPedestals::Item item;
	      item.mean_x1  = aped.mean_x1;
	      item.rms_x1   = aped.rms_x1;
	      item.mean_x6  = aped.mean_x6;
	      item.rms_x6   = aped.rms_x6;
	      item.mean_x12 = aped.mean_x12;
	      item.rms_x12  = aped.rms_x12;
	      peds->insert(std::make_pair(eedetidpos.rawId(),item));
	      if(iX==ixmin && iY==iymin) std::cout<<"ped12 " << item.mean_x12<< std::endl;  
	      
	    }
	  }
	}
	

	std::cout <<"WOW: we just retrieved the last valid record from DB "<< std::endl;


	// here we retrieve all the runs after the last from online DB 

	std::cout << "Retrieving run list from ONLINE DB ... " << std::endl;
	
	std::cout << "Making connection..." << std::flush;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	std::cout << "Done." << std::endl;

	if (!econn)
	  {
	    std::cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<std::endl;
	    throw cms::Exception("OMDS not available");
	  } 

	// these are the online conditions DB classes 
	RunList my_runlist ;
	RunTag  my_runtag;
	LocationDef my_locdef;
	RunTypeDef my_rundef;
	
	my_locdef.setLocation("H2_07");
	my_rundef.setRunType("PEDESTAL");
	my_runtag.setLocationDef(my_locdef);
	my_runtag.setRunTypeDef(my_rundef);
	my_runtag.setGeneralTag("LOCAL");
	

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
	unsigned int min_run=max_since+1; // we have to add 1 to the last transferred one 
	
	unsigned int max_run=m_lastRun;
	mon_list=econn->fetchMonRunList(my_runtag, mon_tag,min_run,max_run );
      
	std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();
	size_t mon_runs=mon_run_vec.size();
	std::cout <<"number of Mon runs is : "<< mon_runs<< std::endl;

	if(mon_runs>0){

	  for(size_t kr=0; kr<mon_runs; kr++){

	    unsigned int irun=static_cast<unsigned int>(mon_run_vec[kr].getRunIOV().getRunNumber());
	  
	    std::cout << "here is first sub run : "<< mon_run_vec[kr].getSubRunNumber() << std::endl;
	    std::cout << "here is the run number: "<< mon_run_vec[kr].getRunIOV().getRunNumber() << std::endl;
	  
	    std::cout <<" retrieve the data for a given run"<< std::endl;
	  
	    if (mon_run_vec[kr].getSubRunNumber() <=1){ 


	      // retrieve the data for a given run
	      RunIOV runiov_prime = mon_run_vec[kr].getRunIOV();
	      
	      // retrieve the pedestals from OMDS for this run 
	      std::map<EcalLogicID, MonPedestalsDat> dataset_mon;
	      econn->fetchDataSet(&dataset_mon, &mon_run_vec[kr]);
	      std::cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_mon.size() << std::endl;
	      typedef std::map<EcalLogicID, MonPedestalsDat>::const_iterator CImon;
	      EcalLogicID ecid_xt;
	      MonPedestalsDat  rd_ped;
	      
	      //int iEta=0;
	      //int iPhi=0;
	      int ix=0;
	      int iy=0;
	      
	      for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
		ecid_xt = p->first;
		rd_ped  = p->second;
		//int sm_num=ecid_xt.getID1();
		int xt_num=ecid_xt.getID2(); // careful here !!! we number the channels from 1 to 1700
		
		//iEta=(xt_num/20)+1;
		//iPhi=20-(xt_num-(iEta-1)*20);

		ix=95-(xt_num-1)/20;
		iy=46+(xt_num-1)%20;
		

		EcalPedestals::Item item;
		item.mean_x1  =rd_ped.getPedMeanG1() ;
		item.rms_x1   =rd_ped.getPedRMSG1();
		item.mean_x6  =rd_ped.getPedMeanG6();
		item.rms_x6   =rd_ped.getPedRMSG6() ;
		item.mean_x12 =rd_ped.getPedMeanG12();
		item.rms_x12  =rd_ped.getPedRMSG12();
		
		EEDetId eedetidpos(ix,iy,1);
		// EBDetId ebdetid(iEta,iPhi);

		// here we change in the peds object only the values that are available in the online DB 
		// otherwise we keep the old value 
		
		peds->insert(std::make_pair(eedetidpos.rawId(),item));
		if(ix==ixmin && iy==iymin) std::cout<<"ped12 " << item.mean_x12<< std::endl;  
	      }
	    
	      std::cout << "Generating popcon record for run " << irun << "..." << std::flush;


	      // now I copy peds in pedtemp and I ship pedtemp to popcon
	      // if I use always the same peds I always overwrite
	      // so I really have to create new objects for each new run
	      // popcon deletes everything for me 

	      EcalPedestals* pedtemp = new EcalPedestals();
	     

	      for(int iX=ixmin; iX<=ixmax ;++iX) {
		for(int iY=iymin; iY<=iymax; ++iY) {
		  
		  if (EEDetId::validDetId(iX,iY,1))
		    {
		      EEDetId eedetidpos(iX,iY,1);
		      unsigned int hiee = eedetidpos.hashedIndex();
		      EcalPedestals::Item aped = peds->endcap(hiee);

		      EcalPedestals::Item item;
		      item.mean_x1  = aped.mean_x1;
		      item.rms_x1   = aped.rms_x1;
		      item.mean_x6  = aped.mean_x6;
		      item.rms_x6   = aped.rms_x6;
		      item.mean_x12 = aped.mean_x12;
		      item.rms_x12  = aped.rms_x12;
		      // here I copy the last valid value in the pedtemp object
		      pedtemp->insert(std::make_pair(eedetidpos.rawId(),item));
		      if(iX==ixmin && iY==iymin) std::cout<<"ped12 " << item.mean_x12<< std::endl;  
		    }
		}
	      }



	      Time_t snc= (Time_t) irun ;
	      	      
	      m_to_transfer.push_back(std::make_pair((EcalPedestals*)pedtemp,snc));
	      
	      std::cout << "Ecal - > end of getNewObjectsH2 -----------\n";
    
	    }
	  }	  
	}
	delete econn;
	delete peds;  // this is the only one that popcon does not delete 
}

void popcon::EcalPedestalsHandler::readPedestalFile() {
  std::cout << " reading the input file " << m_filename <<  std::endl;
  std::ifstream fInput;
  fInput.open(m_filename);
  if(!fInput.is_open()) {
    std::cout << "ERROR : cannot open file " << m_filename << std::endl;
    exit (1);
  }
  //  string pos, dummyLine;
  int hashedId;
  float EBmean12[kEBChannels], EBrms12[kEBChannels], EBmean6[kEBChannels], EBrms6[kEBChannels], EBmean1[kEBChannels], EBrms1[kEBChannels];
  //*****************  barrel
  //  getline(fInput, dummyLine);   // skip first line
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    fInput >> hashedId >> EBmean12[iChannel] >> EBrms12[iChannel] >> EBmean6[iChannel] >> EBrms6[iChannel] 
	   >> EBmean1[iChannel] >> EBrms1[iChannel];
    if(hashedId != iChannel + 1) {
      std::cout << m_filename << " strange hash " << hashedId << " while iChannel " << iChannel << std::endl;
      exit(-1);
    }
  }

  // ***************** now EE *****************
  float EEmean12[kEEChannels], EErms12[kEEChannels], EEmean6[kEEChannels], EErms6[kEEChannels], EEmean1[kEEChannels], EErms1[kEEChannels];
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    fInput >> hashedId >> EEmean12[iChannel] >> EErms12[iChannel] >> EEmean6[iChannel] >> EErms6[iChannel] 
	   >> EEmean1[iChannel] >> EErms1[iChannel];
    if(hashedId != iChannel + kEBChannels + 1) {
      std::cout << m_filename << " strange hash " << hashedId << " while iChannel " << iChannel << std::endl;
      exit(-1);
    }
  }
  fInput.close();

  EcalPedestals* ped = new EcalPedestals();
  // barrel
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      if (EBDetId::validDetId(iEta,iPhi)) {
	EBDetId ebdetid(iEta,iPhi);
	unsigned int hieb = ebdetid.hashedIndex();
	EcalPedestals::Item item;
	item.mean_x1  = EBmean1[hieb];
	item.rms_x1   = EBrms1[hieb];
	item.mean_x6  = EBmean6[hieb];
	item.rms_x6   = EBrms6[hieb];
	item.mean_x12 = EBmean12[hieb];
	item.rms_x12  = EBrms12[hieb];
	ped->insert(std::make_pair(ebdetid.rawId(),item));
      }   // valid EBId
    }    //  loop over phi
  }     //   loop over eta
  // endcaps 
  //  std::ofstream fout;
  //  fout.open("Pedestal.check");
  for(int iz = -1; iz < 2; iz = iz + 2) {   // z : -1 and +1
    for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
      for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
	if (EEDetId::validDetId(iX, iY, iz)) {
	  EEDetId eedetid(iX, iY, iz);
	  unsigned int hiee = eedetid.hashedIndex();
	  //	  fout << hiee << " mean 12 " << EEmean12[hiee] << std::endl;
	  EcalPedestals::Item item;
	  item.mean_x1  = EEmean1[hiee];
	  item.rms_x1   = EErms1[hiee];
	  item.mean_x6  = EEmean6[hiee];
	  item.rms_x6   = EErms6[hiee];
	  item.mean_x12 = EEmean12[hiee];
	  item.rms_x12  = EErms12[hiee];
	  ped->insert(std::make_pair(eedetid.rawId(),item));
	}  // val EEId
      }   //  loop over y
    }    //   loop over x
  }     //   loop over z
  //  fout.close();

  unsigned int irun = m_firstRun;
  Time_t snc= (Time_t) irun ;      	      
  m_to_transfer.push_back(std::make_pair((EcalPedestals*)ped,snc));
	      
  std::cout << "Ecal - > end of readPedestalFile -----------\n";
}

void popcon::EcalPedestalsHandler::readPedestal2017() {
  bool debug = false;
  std::cout << " reading local Pedestal run (2017 way)! " << m_filename <<  std::endl;

  // First we copy the last valid record to a temporary object peds
  Ref ped_db = lastPayload();
  EcalPedestals* peds = new EcalPedestals();
  std::cout << "retrieved last payload "  << std::endl;

  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(iEta,iPhi)) {
	EBDetId ebdetid(iEta,iPhi,EBDetId::ETAPHIMODE);
	EcalPedestals::const_iterator it = ped_db->find(ebdetid.rawId());
	EcalPedestals::Item aped = (*it);
	// here I copy the last valid value in the peds object
	EcalPedestals::Item item;
	item.mean_x1  = aped.mean_x1;
	item.rms_x1   = aped.rms_x1;
	item.mean_x6  = aped.mean_x6;
	item.rms_x6   = aped.rms_x6;
	item.mean_x12 = aped.mean_x12;
	item.rms_x12  = aped.rms_x12;
	peds->insert(std::make_pair(ebdetid.rawId(),item));
      }
    }
  }
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1)) {
	EEDetId eedetidpos(iX,iY,1);
	EcalPedestals::const_iterator it = ped_db->find(eedetidpos.rawId());
	EcalPedestals::Item aped = (*it);
	// here I copy the last valid value in the peds object
	EcalPedestals::Item item;
	item.mean_x1  = aped.mean_x1;
	item.rms_x1   = aped.rms_x1;
	item.mean_x6  = aped.mean_x6;
	item.rms_x6   = aped.rms_x6;
	item.mean_x12 = aped.mean_x12;
	item.rms_x12  = aped.rms_x12;
	peds->insert(std::make_pair(eedetidpos.rawId(),item));
      }
      if(EEDetId::validDetId(iX,iY,-1)) {
	EEDetId eedetidneg(iX,iY,-1);
	EcalPedestals::const_iterator it = ped_db->find(eedetidneg.rawId());
	EcalPedestals::Item aped = (*it);
	// here I copy the last valid value in the peds object
	EcalPedestals::Item item;
	item.mean_x1  = aped.mean_x1;
	item.rms_x1   = aped.rms_x1;
	item.mean_x6  = aped.mean_x6;
	item.rms_x6   = aped.rms_x6;
	item.mean_x12 = aped.mean_x12;
	item.rms_x12  = aped.rms_x12;
	peds->insert(std::make_pair(eedetidneg.rawId(),item));
      }
    }
  }

  std::cout << " reading the input file " << m_filename <<  std::endl;
  std::ifstream fInput;
  fInput.open(m_filename);
  if(!fInput.is_open()) {
    std::cout << "ERROR : cannot open file " << m_filename << std::endl;
    exit (1);
  }
  //  string pos, dummyLine;
  int hashedId;
  float EBmean12[kEBChannels], EBrms12[kEBChannels], EBmean6[kEBChannels], EBrms6[kEBChannels], EBmean1[kEBChannels], EBrms1[kEBChannels];
  //*****************  barrel
  //  getline(fInput, dummyLine);   // skip first line
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    fInput >> hashedId >> EBmean12[iChannel] >> EBrms12[iChannel] >> EBmean6[iChannel] >> EBrms6[iChannel] 
	   >> EBmean1[iChannel] >> EBrms1[iChannel];
    if(hashedId != iChannel + 1) {
      std::cout << m_filename << " strange hash " << hashedId << " while iChannel " << iChannel << std::endl;
      exit(-1);
    }
  }

  // ***************** now EE *****************
  float EEmean12[kEEChannels], EErms12[kEEChannels], EEmean6[kEEChannels], EErms6[kEEChannels], EEmean1[kEEChannels], EErms1[kEEChannels];
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    fInput >> hashedId >> EEmean12[iChannel] >> EErms12[iChannel] >> EEmean6[iChannel] >> EErms6[iChannel] 
	   >> EEmean1[iChannel] >> EErms1[iChannel];
    if(hashedId != iChannel + kEBChannels + 1) {
      std::cout << m_filename << " strange hash " << hashedId << " while iChannel " << iChannel << std::endl;
      exit(-1);
    }
  }
  fInput.close();

  EcalPedestals* ped = new EcalPedestals();
  // barrel
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      if (EBDetId::validDetId(iEta,iPhi)) {
	EBDetId ebdetid(iEta,iPhi);
	unsigned int hieb = ebdetid.hashedIndex();
	EcalPedestals::Item item;
	EcalPedestals::Item previous_ped = peds->barrel(hieb);
	if(debug & (EBmean12[hieb] == -999. || EBrms12[hieb] == -999. || EBmean6[hieb] == -999. || EBrms6[hieb] == -999. ||
		    EBmean1[hieb] == -999. || EBrms1[hieb] == -999.))
	  std::cout << " bad EB channel eta " << iEta << " phi " << iPhi  << " " << EBmean12[hieb] << " " <<  EBrms12[hieb]
		    << " " << EBmean6[hieb] << " " << EBrms6[hieb] << " " << EBmean1[hieb] << " " << EBrms1[hieb] << std::endl;
	if(EBmean1[hieb] != -999.) item.mean_x1  = EBmean1[hieb];
	else item.mean_x1 = previous_ped.mean_x1;
	if(EBrms1[hieb] != -999.)  item.rms_x1   = EBrms1[hieb];
	else item.rms_x1 = previous_ped.rms_x1;
	if(EBmean6[hieb] != -999.) item.mean_x6  = EBmean6[hieb];
	else item.mean_x6 = previous_ped.mean_x6;
	if(EBrms6[hieb] != -999.) item.rms_x6   = EBrms6[hieb];
	else item.rms_x6  = previous_ped.rms_x6;
	if(EBmean12[hieb] != -999.) item.mean_x12 = EBmean12[hieb];
	else item.mean_x12 = previous_ped.mean_x12;
	if(EBrms12[hieb] != -999.) item.rms_x12  = EBrms12[hieb];
	else item.rms_x12 = previous_ped.rms_x12;
	ped->insert(std::make_pair(ebdetid.rawId(),item));
      }   // valid EBId
    }    //  loop over phi
  }     //   loop over eta
  // endcaps 
  //  std::ofstream fout;
  //  fout.open("Pedestal.check");
  for(int iz = -1; iz < 2; iz = iz + 2) {   // z : -1 and +1
    for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
      for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
	if (EEDetId::validDetId(iX, iY, iz)) {
	  EEDetId eedetid(iX, iY, iz);
	  unsigned int hiee = eedetid.hashedIndex();
	  //	  fout << hiee << " mean 12 " << EEmean12[hiee] << std::endl;
	  EcalPedestals::Item item;
	  EcalPedestals::Item previous_ped= peds->endcap(hiee);
	  if(debug & (EEmean12[hiee] == -999. || EErms12[hiee] == -999. || EEmean6[hiee] == -999. || EErms6[hiee] == -999. ||
		      EEmean1[hiee] == -999. || EErms1[hiee] == -999.))
	    std::cout << " bad EE channel x " << iX << " y " << iY  << " z" << iz << " " << EEmean12[hiee] << " " <<  EErms12[hiee]  
		      << " " <<EEmean6[hiee] << " " << EErms6[hiee] << " " << EEmean1[hiee] << " " << EErms1[hiee] << std::endl;
	  if(EEmean1[hiee] != -999.) item.mean_x1  = EEmean1[hiee];
	  else item.mean_x1 = previous_ped.mean_x1;
	  if(EErms1[hiee] != -999.)  item.rms_x1   = EErms1[hiee];
	  else item.rms_x1 = previous_ped.rms_x1;
	  if(EEmean6[hiee] != -999.) item.mean_x6  = EEmean6[hiee];
	  else item.mean_x6 = previous_ped.mean_x6;
	  if(EErms6[hiee] != -999.) item.rms_x6   = EErms6[hiee];
	  else item.rms_x6  = previous_ped.rms_x6;
	  if(EEmean12[hiee] != -999.) item.mean_x12 = EEmean12[hiee];
	  else item.mean_x12 = previous_ped.mean_x12;
	  if(EErms12[hiee] != -999.) item.rms_x12  = EErms12[hiee];
	  else item.rms_x12 = previous_ped.rms_x12;
	  ped->insert(std::make_pair(eedetid.rawId(),item));
	}  // val EEId
      }   //  loop over y
    }    //   loop over x
  }     //   loop over z
  //  fout.close();

  unsigned int irun = m_firstRun;
  Time_t snc= (Time_t) irun ;      	      
  m_to_transfer.push_back(std::make_pair((EcalPedestals*)ped,snc));
	      
  std::cout << "Ecal - > end of readPedestalFile -----------\n";
}

void popcon::EcalPedestalsHandler::readPedestalTree() {
  std::cout << " reading the root file " << m_filename <<  std::endl;
  TFile * hfile = new TFile(m_filename.c_str());

  TTree *treeChan = (TTree*)hfile->Get("PedChan");
  int iChannel = 0, ix = 0, iy = 0, iz = 0;
  treeChan->SetBranchAddress("Channels", &iChannel);
  treeChan->SetBranchAddress("x", &ix);
  treeChan->SetBranchAddress("y", &iy);
  treeChan->SetBranchAddress("z", &iz);
  int neventsChan = (int)treeChan->GetEntries();
  std::cout << "PedChan nb entries " << neventsChan << std::endl;
  int ringEB[kEBChannels], sideEB[kEBChannels], ixEE[kEEChannels], iyEE[kEEChannels], izEE[kEEChannels];
  for(int entry = 0; entry < neventsChan; entry++) {
    treeChan->GetEntry(entry);
    if(entry < kEBChannels) {
      ringEB[iChannel] = (abs(ix) -1)/5;    // -85...-1, 1...85 give 0...16
      sideEB[iChannel] = 1;
      if(ix < 0) sideEB[iChannel] = 0;  
      if(entry%10000 == 0) std::cout << " EB channel " << iChannel << " eta " << ix << " phi " << iy 
				<< " side " << sideEB[iChannel] << " ring " << ringEB[iChannel] << std::endl;
    }
    else {
      ixEE[iChannel] = ix;
      iyEE[iChannel] = iy;
      izEE[iChannel] = iz;
      if(entry%1000 == 0) std::cout << " EE channel " << iChannel << " x " << ixEE[iChannel] << " y " << iyEE[iChannel] << " z " << izEE[iChannel] << std::endl;
    }
  }
  Int_t pedxml[26] = {271948, 273634, 273931, 274705, 275403, 276108, 276510, 277169, 278123, 278183,
		      278246, 278389, 278499, 278693, 278858, 278888, 278931, 279728, 280129, 280263,
		      280941, 281753, 282631, 282833, 283199, 283766};
  Int_t run16Index = 0;

  Int_t fed[kChannels], chan[kChannels], id, run, run_type, seq_id, las_id, fill_num, run_num_infill, run_time, run_time_stablebeam, nxt, time[54];
  Float_t ped[kChannels], pedrms[kChannels], lumi, bfield;
  TTree *tree = (TTree*)hfile->Get("T");
  tree->Print();
  tree->SetBranchAddress("id", &id);
  tree->SetBranchAddress("run", &run);
  tree->SetBranchAddress("run_type", &run_type);
  tree->SetBranchAddress("seq_id", &seq_id);
  tree->SetBranchAddress("las_id", &las_id);
  tree->SetBranchAddress("fill_num", &fill_num);
  tree->SetBranchAddress("run_num_infill", &run_num_infill);
  tree->SetBranchAddress("run_time", &run_time);
  tree->SetBranchAddress("run_time_stablebeam", &run_time_stablebeam);
  tree->SetBranchAddress("lumi", &lumi);
  tree->SetBranchAddress("bfield", &bfield);
  tree->SetBranchAddress("nxt", &nxt);    // nb of filled Crystals
  tree->SetBranchAddress("time", time);
  tree->SetBranchAddress("fed", fed);
  tree->SetBranchAddress("chan", chan);
  tree->SetBranchAddress("ped", ped);
  tree->SetBranchAddress("pedrms", pedrms);
  int nevents = (int)tree->GetEntries();
  std::cout << " nb entries " << nevents << std::endl;
  std::ofstream fout;
  fout.open("copyTreePedestals.txt");
  if(!fout.is_open()) {
    std::cout << "ERROR : cannot open file copyTreePedestals.txt" << std::endl;
    exit (1);
  }
  Double_t EAmean1[kChannels], EArms1[kChannels], EAmean6[kChannels], EArms6[kChannels], EAmean12[kChannels], EArms12[kChannels];
  Int_t RunEntry = 0, EAentry[kChannels];
  for(int ich = 0; ich < kChannels; ich++) {
    EAmean12[ich] = 0;
    EArms12[ich] = 0;
    EAentry[ich] = 0;
  }
  tree->GetEntry(0);
  fout << " first run " << run << " fill " << fill_num << " B field " << bfield << " run type " << run_type << " seq_id " << seq_id
       << " las_id " << las_id << " run_num_infill " << run_num_infill << " lumi "<< lumi << " nb of Crystals " << nxt << std::endl;
  for(int ich = 0; ich < kChannels; ich++) {
    if(ped[ich] != 0.) {
      if(ich < kEBChannels)
	fout << " channel " << ich  << " FED " << fed[ich] << " chan " << chan[ich] << " pedestal " << ped[ich] << " RMS " << pedrms[ich] << std::endl;
      else                //   EE
	fout << " channel " << ich  << " EE channel " << ich - kEBChannels << " FED " << fed[ich] << " chan " << chan[ich] 
	     << " pedestal " << ped[ich] << " RMS " << pedrms[ich] << std::endl;
    }
  }

  int run_type_kept = m_runtype;    // 1 collision 2 cosmics 3 circulating 4 test
  int first_run_kept = (int)m_firstRun;  //  m_firstRun is unsigned!
  int runold = run;
  int firsttimeFED = -1;
  //  int timeold = -1;
  for(int entry = 0; entry < nevents; entry++) {
    tree->GetEntry(entry);
    if(run < first_run_kept) {
      runold = run;
      continue;
    }
    if(run_type  != run_type_kept) continue;     // use only wanted type runs
    if(nxt != kChannels) {
      fout << " entry " << entry << " run " << run << " sequence " << seq_id << " run_time " << run_time
	   << " ***********  Number of channels " << nxt << std::endl;
      continue;
    }
    if(bfield < 3.79) {
      fout << " entry " << entry << " run " << run << " sequence " << seq_id << " run_time " << run_time
	   << " ***********  bfield = " << bfield << std::endl;
      //      continue;  keep these runs
    }
    if(run_type_kept == 1) {   // for collision runs, keep only sequences after stable beam
      int time_seq = 0;
      for(int ifed = 0; ifed < 54; ifed++) {
	if(time[ifed] < run_time_stablebeam) {
	  if(time_seq == 0)
	    fout << " entry " << entry << " run " << run << " sequence " << seq_id << " run_time " << run_time
		 << "  ***********  sequence before stable beam at " << run_time_stablebeam << " FED " << ifed << " : " << time[ifed];
	  else
	    fout << " FED " << ifed << " : " << time[ifed];
	  time_seq++;
	}
      }
      if(time_seq != 0) {
	fout << " total nb " << time_seq  << std::endl;
	continue;
      }
    }  // only collision runs
    //    if(idtime != timeold) {
    //      std::cout << " entry "<< entry << " run " << run << " time " << idtime << " run type " << run_type << std::endl;
    //      timeold = idtime;
    //    }
    if(run == runold) {
      RunEntry++;
      for(int ich = 0; ich < kChannels; ich++) {
	if(ped[ich] < 300 && ped[ich] > 100) {
	  EAmean12[ich] += ped[ich];
	  EArms12[ich] += pedrms[ich];
	  EAentry[ich]++;
	}
      }
    }
    else {
      // new run. Write the previous one
      //      std::cout << " entry "<< entry << " fill " << fill_num << " run " << runold << " nb of events " << RunEntry
      //		<< " time " << run_time << " " << run_time_stablebeam << " " << time[0] << " run type " << run_type << std::endl;
      if(RunEntry == 0 || (run_type_kept == 2 && RunEntry < 6)) fout << " skiped run " << runold << " not enough entries : " << RunEntry << std::endl;
      else {
	fout << " entry "<< entry -1 << " run " << runold << " nb of events " << RunEntry;
	firsttimeFED = time[0];
	for(int ifed = 0; ifed < 54; ifed++) {
	  fout << " " << time[ifed];
	  if(firsttimeFED < time[ifed]) firsttimeFED = time[ifed];
	}
	fout << std::endl;

	// get gain 1 and 6 results
	bool foundNew = false;
	for(int i = run16Index; i < 26; i++) {
	  if(runold > pedxml[i]) {
	    fout << " found a new gain 1, 6 file " <<  pedxml[i] << " at index " << i << std::endl;
	    run16Index++;
	    foundNew = true;
	    if(runold < pedxml[i + 1]) break;
	  }
	}
	if(foundNew) {
	  int Indexxml = run16Index -1;
	  fout << " opening Pedestals_" <<  pedxml[Indexxml] << ".xml at index " << Indexxml << std::endl;
	  std::ifstream fxml;
	  fxml.open(Form("Pedestals_%i.xml",pedxml[Indexxml]));
	  if(!fxml.is_open()) {
	    std::cout << "ERROR : cannot open file Pedestals_" << pedxml[Indexxml] << ".xml" << std::endl;
	    exit (1);
	  }
	  std::string dummyLine, mean12, rms12, bid;
	  for(int i = 0; i < 9; i++) std::getline(fxml, dummyLine);   // skip first lines
	  // Barrel
	  for (int iEBChannel = 0; iEBChannel < kEBChannels; iEBChannel++) {
	    fxml >> mean12 >> rms12 >> bid;
	    //	  std::string stt = bid.std::substr(10,15);
	    //	  istringstream ii(stt);
	    //	  ii >> EBmean12[iEBChannel];
	    //	  fxml >> bid;
	    //	  stt = bid.substr(9,15);
	    //	  istringstream r12(stt);
	    //	  r12 >> EBrms12[iEBChannel];
	    //	  fxml >> bid;
	    std::string stt = bid.substr(9,15);
	    std::istringstream m6(stt);
	    m6 >> EAmean6[iEBChannel];
	    fxml >> bid;
	    stt = bid.substr(8,15);
	    std::istringstream r6(stt);
	    r6 >> EArms6[iEBChannel];
	    fxml >> bid;
	    stt = bid.substr(9,15);
	    std::istringstream m1(stt);
	    m1 >> EAmean1[iEBChannel];
	    fxml >> bid;
	    stt = bid.substr(8,15);
	    std::istringstream r1(stt);
	    r1 >> EArms1[iEBChannel];
	    if(iEBChannel%10000 == 0) fout << " EB channel " << iEBChannel << " " << mean12 << " " << rms12
					   << " " << EAmean6[iEBChannel] << " " << EArms6[iEBChannel] 
					   << " " << EAmean1[iEBChannel] << " " << EArms1[iEBChannel] << std::endl;
	    for(int i = 0; i < 3; i++)   std::getline(fxml, dummyLine);   // skip lines
	  }
	  for(int i = 0; i < 6; i++) std::getline(fxml, dummyLine);   // skip lines
	  for (int iEEChannel = 0; iEEChannel < kEEChannels; iEEChannel++) {
	    int ich = iEEChannel + kEBChannels;
	    fxml >> mean12 >> rms12 >> bid;
	    std::string stt = bid.substr(9,15);
	    std::istringstream m6(stt);
	    m6 >> EAmean6[ich];
	    fxml >> bid;
	    stt = bid.substr(8,15);
	    std::istringstream r6(stt);
	    r6 >> EArms6[ich];
	    fxml >> bid;
	    stt = bid.substr(9,15);
	    std::istringstream m1(stt);
	    m1 >> EAmean1[ich];
	    fxml >> bid;
	    stt = bid.substr(8,15);
	    std::istringstream r1(stt);
	    r1 >> EArms1[ich];
	    if(iEEChannel%1000 == 0) fout << " EE channel " << iEEChannel << " " << mean12 << " " << rms12
					  << " " << EAmean6[ich] << " " << EArms6[ich] 
					  << " " << EAmean1[ich] << " " << EArms1[ich] << std::endl;
	    for(int i = 0; i < 3; i++) std::getline(fxml, dummyLine);   // skip lines
	  }

	  fxml.close();
	}
	// end gain 1 and 6 results

	EcalPedestals* pedestal = new EcalPedestals();
	EcalPedestals::Item item;
	for(int ich = 0; ich < kChannels; ich++) {
	  if(EAentry[ich] != 0) {
	    EAmean12[ich] /= EAentry[ich];
	    EArms12[ich] /= EAentry[ich];
	  }
	  else {
	    EAmean12[ich] = 200.;
	    EArms12[ich] = 0.;
	  }
	  if(ich%10000 == 0) fout << " channel " << ich << " ped " << EAmean12[ich] << " RMS " << EArms12[ich] << std::endl;
	  //
	  item.mean_x1  = EAmean1[ich];
	  item.rms_x1   = EArms1[ich];
	  item.mean_x6  = EAmean6[ich];
	  item.rms_x6   = EArms6[ich];
	  item.mean_x12 = EAmean12[ich];
	  item.rms_x12  = EArms12[ich];
	  if(ich < kEBChannels) {
	    //	EBDetId ebdetid(fed, chan, EBDetId::SMCRYSTALMODE);
	    EBDetId ebdetid = EBDetId::unhashIndex(ich);
	    pedestal->insert(std::make_pair(ebdetid.rawId(),item));
	  }
	  else {
	    //	  EEDetId eedetid(iX, iY, iz);
	    int iChannel = ich - kEBChannels;
	    EEDetId eedetid = EEDetId::unhashIndex(iChannel);
	    pedestal->insert(std::make_pair(eedetid.rawId(),item));
	  }
	}  // end loop over all channels
	Time_t snc= (Time_t) runold;      	      
	m_to_transfer.push_back(std::make_pair((EcalPedestals*)pedestal, snc));
      } // skip too short cosmics runs

      runold = run;
      RunEntry = 1;
      for(int ich = 0; ich < kChannels; ich++) {
	if(ped[ich] < 300 && ped[ich] > 100) {
	  EAmean12[ich] = ped[ich];
	  EArms12[ich] = pedrms[ich];
	  EAentry[ich] = 1;
	}
	else {
	  EAmean12[ich] = 0;
	  EArms12[ich] = 0;
	  EAentry[ich] = 0;
	}
      }
    }  // new run
  }  // end loop over all entries
  // write also the last run
  fout << " last entry fill " << fill_num << " run " << runold << " nb of events " << RunEntry
       << " time " << run_time << " " << run_time_stablebeam << " " << time[0] << " run type " << run_type << std::endl;
  for(int ifed = 0; ifed < 54; ifed++) 
    fout << " " << time[ifed];
  fout << std::endl;
  EcalPedestals* pedestal = new EcalPedestals();
  EcalPedestals::Item item;
  for(int ich = 0; ich < kChannels; ich++) {
    if(EAentry[ich] != 0) {
      EAmean12[ich] /= EAentry[ich];
      EArms12[ich] /= EAentry[ich];
    }
    else {
      EAmean12[ich] = 200.;
      EArms12[ich] = 0.;
    }
    if(ich%10000 == 0) fout << " channel " << ich << " ped " << EAmean12[ich] << " RMS " << EArms12[ich] << std::endl;
    // get gain 1 and 6 results
    // ...
    //
    item.mean_x1  = EAmean1[ich];
    item.rms_x1   = EArms1[ich];
    item.mean_x6  = EAmean6[ich];
    item.rms_x6   = EArms6[ich];
    item.mean_x12 = EAmean12[ich];
    item.rms_x12  = EArms12[ich];
    if(ich < kEBChannels) {
      //	EBDetId ebdetid(fed, chan, EBDetId::SMCRYSTALMODE);
      EBDetId ebdetid = EBDetId::unhashIndex(ich);
      pedestal->insert(std::make_pair(ebdetid.rawId(),item));
    }
    else {
      //	  EEDetId eedetid(iX, iY, iz);
      int iChannel = ich - kEBChannels;
      EEDetId eedetid = EEDetId::unhashIndex(iChannel);
      pedestal->insert(std::make_pair(eedetid.rawId(),item));
    }
  }  // end loop over all channels
  Time_t snc= (Time_t) runold;      	      
  m_to_transfer.push_back(std::make_pair((EcalPedestals*)pedestal,snc));   // run based IOV
  std::cout << "Ecal - > end of readPedestalTree -----------\n";
}

void popcon::EcalPedestalsHandler::readPedestalTimestamp() {
  bool debug = false;
  std::cout << " reading the root file " << m_filename <<  std::endl;
  TFile * hfile = new TFile(m_filename.c_str());

  TTree *treeChan = (TTree*)hfile->Get("PedChan");
  int iChannel = 0, ix = 0, iy = 0, iz = 0;
  treeChan->SetBranchAddress("Channels", &iChannel);
  treeChan->SetBranchAddress("x", &ix);
  treeChan->SetBranchAddress("y", &iy);
  treeChan->SetBranchAddress("z", &iz);
  int neventsChan = (int)treeChan->GetEntries();
  std::cout << "PedChan nb entries " << neventsChan << std::endl;
  int ringEB[kEBChannels], sideEB[kEBChannels], ixEE[kEEChannels], iyEE[kEEChannels], izEE[kEEChannels];
  for(int entry = 0; entry < neventsChan; entry++) {
    treeChan->GetEntry(entry);
    if(entry < kEBChannels) {
      ringEB[iChannel] = (abs(ix) -1)/5;    // -85...-1, 1...85 give 0...16
      sideEB[iChannel] = 1;
      if(ix < 0) sideEB[iChannel] = 0;  
      if(debug && entry%10000 == 0) std::cout << " EB channel " << iChannel << " eta " << ix << " phi " << iy 
				<< " side " << sideEB[iChannel] << " ring " << ringEB[iChannel] << std::endl;
    }
    else {
      ixEE[iChannel] = ix;
      iyEE[iChannel] = iy;
      izEE[iChannel] = iz;
      if(debug && entry%1000 == 0) std::cout << " EE channel " << iChannel << " x " << ixEE[iChannel] << " y "
					     << iyEE[iChannel] << " z " << izEE[iChannel] << std::endl;
    }
  }
  Int_t pedxml[26] = {271948, 273634, 273931, 274705, 275403, 276108, 276510, 277169, 278123, 278183,
		      278246, 278389, 278499, 278693, 278858, 278888, 278931, 279728, 280129, 280263,
		      280941, 281753, 282631, 282833, 283199, 283766};
  Int_t run16Index = 0;

  Int_t fed[kChannels], chan[kChannels], id, run, run_type, seq_id, las_id, fill_num, run_num_infill, run_time, run_time_stablebeam, nxt, time[54];
  Float_t ped[kChannels], pedrms[kChannels], lumi, bfield;
  TTree *tree = (TTree*)hfile->Get("T");
  tree->Print();
  tree->SetBranchAddress("id", &id);
  tree->SetBranchAddress("run", &run);
  tree->SetBranchAddress("run_type", &run_type);
  tree->SetBranchAddress("seq_id", &seq_id);
  tree->SetBranchAddress("las_id", &las_id);
  tree->SetBranchAddress("fill_num", &fill_num);
  tree->SetBranchAddress("run_num_infill", &run_num_infill);
  tree->SetBranchAddress("run_time", &run_time);
  tree->SetBranchAddress("run_time_stablebeam", &run_time_stablebeam);
  tree->SetBranchAddress("lumi", &lumi);
  tree->SetBranchAddress("bfield", &bfield);
  tree->SetBranchAddress("nxt", &nxt);    // nb of filled Crystals
  tree->SetBranchAddress("time", time);
  tree->SetBranchAddress("fed", fed);
  tree->SetBranchAddress("chan", chan);
  tree->SetBranchAddress("ped", ped);
  tree->SetBranchAddress("pedrms", pedrms);
  int nevents = (int)tree->GetEntries();
  std::cout << " nb entries " << nevents << std::endl;
  std::ofstream fout;
  fout.open("copyTimestampPedestals.txt");
  if(!fout.is_open()) {
    std::cout << "ERROR : cannot open file copyTimestampPedestals.txt" << std::endl;
    exit (1);
  }
  Double_t EAmean1[kChannels], EArms1[kChannels], EAmean6[kChannels], EArms6[kChannels], EAmean12[kChannels], EArms12[kChannels];
  for(int ich = 0; ich < kChannels; ich++) {
    EAmean12[ich] = 200.;
    EArms12[ich] = 0.;
  }
  tree->GetEntry(0);
  fout << " first run " << run << " fill " << fill_num << " B field " << bfield << " run type " << run_type << " seq_id " << seq_id
       << " las_id " << las_id << " run_num_infill " << run_num_infill << " lumi "<< lumi << " nb of Crystals " << nxt << std::endl;
  if(debug) { 
    for(int ich = 0; ich < kChannels; ich++) {
      if(ped[ich] != 0.) {
	if(ich < kEBChannels)
	  fout << " channel " << ich  << " FED " << fed[ich] << " chan " << chan[ich] << " pedestal " << ped[ich] << " RMS " << pedrms[ich] << std::endl;
	else                //   EE
	  fout << " channel " << ich  << " EE channel " << ich - kEBChannels << " FED " << fed[ich] << " chan " << chan[ich] 
	       << " pedestal " << ped[ich] << " RMS " << pedrms[ich] << std::endl;
      }
    }
  }   // debug

  int runold = -1, fillold = -1, firsttimeFEDold = -1;
  int firsttimeFED = -1;
  bool firstSeqAfterStable = false;
  for(int entry = 0; entry < nevents; entry++) {
    tree->GetEntry(entry);
    if(nxt != kChannels) {
      fout << " entry " << entry << " run " << run << " sequence " << seq_id << " run_time " << run_time
	   << " ***********  Number of channels " << nxt;
      if(seq_id == 0) {
	fout << " rejected"  << std::endl;
	continue;
      }
      else fout << std::endl;
    }
    if(las_id != 447) {
      fout << " entry " << entry << " run " << run << " sequence " << seq_id 
	   << " ***********  laser wave length = " << las_id << std::endl;
      continue;
    }
    if(bfield < 3.79) {
      fout << " entry " << entry << " run " << run << " sequence " << seq_id << " run_time " << run_time
	   << " ***********  bfield = " << bfield << std::endl;
      //      continue;  keep these runs
    }
    fout << " entry "<< entry << " run " << run << " sequence " << seq_id;
    if(run_type == 1) fout << " stable " << run_time_stablebeam;
    firsttimeFED = time[0];
    for(int ifed = 0; ifed < 54; ifed++) {
      //      fout << " " << time[ifed];
      if(time[ifed] > firsttimeFEDold && firsttimeFED < time[ifed]) firsttimeFED = time[ifed];  // take the first AFTER the previous sequence one!...
    }
    fout << " time " << firsttimeFED << std::endl;
    if(firsttimeFED <= firsttimeFEDold) {
      std::cout << " Problem finding the IOV : old one " <<  firsttimeFEDold << " new one " << firsttimeFED  << std::endl;
      for(int ifed = 0; ifed < 54; ifed++)
	std::cout << " " << time[ifed];
      std::cout << std::endl << " ignore this entry " << std::endl;
      continue;
    }
    firsttimeFEDold = firsttimeFED;

    //    if(run != runold) firstSeqAfterStable = false;
    if(fill_num != fillold) firstSeqAfterStable = false;
    if(run_type == 1) {
      if(firsttimeFED > run_time_stablebeam) {
	if(!firstSeqAfterStable) {
	  firstSeqAfterStable = true;
	  firsttimeFED = run_time_stablebeam;
	  fout << " first sequence after stable; change the IOV " << firsttimeFED << std::endl;
	}
      }
    }  // only collision runs

    for(int ich = 0; ich < kChannels; ich++) {
      if(ped[ich] < 300 && ped[ich] > 100) {
	EAmean12[ich] = ped[ich];
	EArms12[ich] = pedrms[ich];
      }
    }
    // get gain 1 and 6 results
    bool foundNew = false;
    if(run != runold) {
      for(int i = run16Index; i < 26; i++) {
	if(run > pedxml[i]) {
	  fout << " found a new gain 1, 6 file " <<  pedxml[i] << " at index " << i << std::endl;
	  run16Index++;
	  foundNew = true;
	  if(runold < pedxml[i + 1]) break;
	}
      }
      if(foundNew) {
	int Indexxml = run16Index -1;
	fout << " opening Pedestals_" <<  pedxml[Indexxml] << ".xml at index " << Indexxml << std::endl;
	std::ifstream fxml;
	fxml.open(Form("Pedestals_%i.xml",pedxml[Indexxml]));
	if(!fxml.is_open()) {
	  std::cout << "ERROR : cannot open file Pedestals_" << pedxml[Indexxml] << ".xml" << std::endl;
	  exit (1);
	}
	std::string dummyLine, mean12, rms12, bid;
	for(int i = 0; i < 9; i++) std::getline(fxml, dummyLine);   // skip first lines
	// Barrel
	for (int iEBChannel = 0; iEBChannel < kEBChannels; iEBChannel++) {
	  fxml >> mean12 >> rms12 >> bid;
	  std::string stt = bid.substr(9,15);
	  std::istringstream m6(stt);
	  m6 >> EAmean6[iEBChannel];
	  fxml >> bid;
	  stt = bid.substr(8,15);
	  std::istringstream r6(stt);
	  r6 >> EArms6[iEBChannel];
	  fxml >> bid;
	  stt = bid.substr(9,15);
	  std::istringstream m1(stt);
	  m1 >> EAmean1[iEBChannel];
	  fxml >> bid;
	  stt = bid.substr(8,15);
	  std::istringstream r1(stt);
	  r1 >> EArms1[iEBChannel];
	  if(debug && iEBChannel%10000 == 0) fout << " EB channel " << iEBChannel << " " << mean12 << " " << rms12
						  << " " << EAmean6[iEBChannel] << " " << EArms6[iEBChannel] 
						  << " " << EAmean1[iEBChannel] << " " << EArms1[iEBChannel] << std::endl;
	  for(int i = 0; i < 3; i++)   std::getline(fxml, dummyLine);   // skip lines
	}
	for(int i = 0; i < 6; i++) std::getline(fxml, dummyLine);   // skip lines
	for (int iEEChannel = 0; iEEChannel < kEEChannels; iEEChannel++) {
	  int ich = iEEChannel + kEBChannels;
	  fxml >> mean12 >> rms12 >> bid;
	  std::string stt = bid.substr(9,15);
	  std::istringstream m6(stt);
	  m6 >> EAmean6[ich];
	  fxml >> bid;
	  stt = bid.substr(8,15);
	  std::istringstream r6(stt);
	  r6 >> EArms6[ich];
	  fxml >> bid;
	  stt = bid.substr(9,15);
	  std::istringstream m1(stt);
	  m1 >> EAmean1[ich];
	  fxml >> bid;
	  stt = bid.substr(8,15);
	  std::istringstream r1(stt);
	  r1 >> EArms1[ich];
	  if(debug && iEEChannel%1000 == 0) fout << " EE channel " << iEEChannel << " " << mean12 << " " << rms12
						 << " " << EAmean6[ich] << " " << EArms6[ich] 
						 << " " << EAmean1[ich] << " " << EArms1[ich] << std::endl;
	  for(int i = 0; i < 3; i++) std::getline(fxml, dummyLine);   // skip lines
	}
	fxml.close();
      }   // found a new gain 1 6 file
    }      // check gain 1 and 6 results only for new runs

    EcalPedestals* pedestal = new EcalPedestals();
    EcalPedestals::Item item;
    for(int ich = 0; ich < kChannels; ich++) {
      if(debug && ich%10000 == 0) fout << " channel " << ich << " ped " << EAmean12[ich] << " RMS " << EArms12[ich] << std::endl;
      //
      item.mean_x1  = EAmean1[ich];
      item.rms_x1   = EArms1[ich];
      item.mean_x6  = EAmean6[ich];
      item.rms_x6   = EArms6[ich];
      item.mean_x12 = EAmean12[ich];
      item.rms_x12  = EArms12[ich];
      if(ich < kEBChannels) {
	//	EBDetId ebdetid(fed, chan, EBDetId::SMCRYSTALMODE);
	EBDetId ebdetid = EBDetId::unhashIndex(ich);
	pedestal->insert(std::make_pair(ebdetid.rawId(),item));
      }
      else {
	//	  EEDetId eedetid(iX, iY, iz);
	int iChannel = ich - kEBChannels;
	EEDetId eedetid = EEDetId::unhashIndex(iChannel);
	pedestal->insert(std::make_pair(eedetid.rawId(),item));
      }
    }  // end loop over all channels
    uint64_t iov  = (uint64_t)firsttimeFED << 32;
    Time_t snc = (Time_t) iov;
    m_to_transfer.push_back(std::make_pair((EcalPedestals*)pedestal, snc));   // time based IOV
    fout << "  m_to_transfer " << firsttimeFED << std::endl;
    runold = run;
    fillold = fill_num;
  }  // end loop over all entries

  std::cout << "Ecal - > end of readPedestalTimestamp -----------\n";
}
