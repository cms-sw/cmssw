#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalPedestalsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::EcalPedestalsHandler::EcalPedestalsHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalPedestalsHandler")) {

	std::cout << "EcalPedestals Source handler constructor\n" << std::endl;
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

popcon::EcalPedestalsHandler::~EcalPedestalsHandler()
{
}


void popcon::EcalPedestalsHandler::getNewObjects()
{

	std::cout << "------- Ecal - > getNewObjects\n";

	if(m_locationsource=="H2") {
	  getNewObjectsH2();
	} else if (m_locationsource=="P5") {
	  getNewObjectsP5();
	}
}


bool popcon::EcalPedestalsHandler::checkPedestal( EcalPedestals::Item* item ){
  // true means all is standard and OK
  bool result=true;
  if(item->rms_x12 >3 || item->rms_x12<=0) result=false; 
  if(item->rms_x6 >3 || item->rms_x6<=0) result=false; 
  if(item->rms_x1 >3 || item->rms_x1<=0) result=false; 
  if(item->mean_x12>300 || item->mean_x12<=100) result=false; 
  if(item->mean_x1>300 || item->mean_x1<=100) result=false; 
  if(item->mean_x6>300 || item->mean_x6<=100) result=false; 
  return result; 
  }

void popcon::EcalPedestalsHandler::getNewObjectsP5()
{

  std::ostringstream ss; 
  ss<<"ECAL ";

	int max_since=0;
	max_since=(int)tagInfo().lastInterval.first;
	std::cout << "max_since : "  << max_since << endl;
	Ref ped_db = lastPayload();
	
	std::cout << "retrieved last payload "  << endl;


	// we copy the last valid record to a temporary object peds
	EcalPedestals* peds = new EcalPedestals();

	std::cout << "retrieved last payload "  << endl;

	//	const EcalPedestalsMap & pedMap = ped_db->getMap(); // map of pedestals
	std::cout << "retrieved last payload "  << endl;

	for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	  if(iEta==0) continue;
	  for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	    // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
	    if (EBDetId::validDetId(iEta,iPhi))
	      {
	

		EBDetId ebdetid(iEta,iPhi,EBDetId::ETAPHIMODE);
		//unsigned int hiee = ebdetid.hashedIndex();
		//EcalPedestals::Item aped= ped_db->barrel(hiee);
		EcalPedestals::const_iterator it =ped_db->find(ebdetid.rawId());
	
		//		pedIter= pedMap.find(ebdetid.rawId());
		//pedIter= pedMap.find(hiee);
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

	/*
	for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	  for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
	    // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
	    if (EEDetId::validDetId(iX,iY,1))
	      {
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

	      }
	    if(EEDetId::validDetId(iX,iY,-1))
	      {
		EEDetId eedetidneg(iX,iY,-1);

		unsigned int hiee = eedetidneg.hashedIndex();
		EcalPedestals::Item aped= ped_db->endcap(hiee);

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

	*/

	cout <<"WOW: we just retrieved the last valid record from DB "<< endl;


	// here we retrieve all the runs after the last from online DB 

	cout << "Retrieving run list from ONLINE DB ... " << endl;

	cout << "Making connection..." << flush;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	cout << "Done." << endl;
	
	if (!econn)
	  {
	    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
	    //	    cerr << e.what() << endl;
	    throw cms::Exception("OMDS not available");
	  } 

	// these are the online conditions DB classes 
	RunList my_runlist ;
	RunTag  my_runtag;
	LocationDef my_locdef;
	RunTypeDef my_rundef;
	
	my_locdef.setLocation(m_location);
	my_rundef.setRunType("PEDESTAL");
	my_runtag.setLocationDef(my_locdef);
	my_runtag.setRunTypeDef(my_rundef);
	my_runtag.setGeneralTag(m_gentag);
	

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
	int min_run=0;
	if(m_firstRun<max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} else {
	  min_run=(int)m_firstRun;
	}

	int max_run=(int)m_lastRun;
	mon_list=econn->fetchMonRunList(my_runtag, mon_tag,min_run,max_run );
      
	std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();
	int mon_runs=mon_run_vec.size();
	cout <<"number of Mon runs is : "<< mon_runs<< endl;

	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    unsigned long irun=(unsigned long) mon_run_vec[kr].getRunIOV().getRunNumber();
	    cout << "here is the run number: "<< mon_run_vec[kr].getRunIOV().getRunNumber() << endl;
	  
	    cout <<" retrieve the data for a given run"<< endl;
	  
	    if (mon_run_vec[kr].getSubRunNumber() <=1){ 

	      // retrieve the data for a given run
	      RunIOV runiov_prime = mon_run_vec[kr].getRunIOV();
	      
	      // retrieve the pedestals from OMDS for this run 
	      map<EcalLogicID, MonPedestalsDat> dataset_mon;
	      econn->fetchDataSet(&dataset_mon, &mon_run_vec[kr]);
	      cout <<"OMDS record for run "<<irun  <<" is made of "<< dataset_mon.size() << endl;
	      typedef map<EcalLogicID, MonPedestalsDat>::const_iterator CImon;
	      EcalLogicID ecid_xt;
	      MonPedestalsDat  rd_ped;
	      





	      // this to validate ...	      
	      int nbad=0;
	      for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
		ecid_xt = p->first;
		rd_ped  = p->second;
		int sm_num=ecid_xt.getID1();
		int xt_num=ecid_xt.getID2(); 
		
		EcalPedestals::Item item;
		item.mean_x1  =rd_ped.getPedMeanG1() ;
		item.rms_x1   =rd_ped.getPedRMSG1();
		item.mean_x6  =rd_ped.getPedMeanG6();
		item.rms_x6   =rd_ped.getPedRMSG6() ;
		item.mean_x12 =rd_ped.getPedMeanG12();
		item.rms_x12  =rd_ped.getPedRMSG12();
		
		EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
		// here we change in the peds object only the values that are available in the online DB 
		// otherwise we keep the old value 

		if(!checkPedestal(&item) ){
		  nbad++;
		  cout <<"BAD LIST: channel " << sm_num << "/" << xt_num << "ped/rms "<<item.mean_x12<< "/"<< item.rms_x12 << endl;
		}
	      }

	      // ok or bad? 
	      // a bad run is for more than 10% bad channels 

	      if(nbad<(dataset_mon.size()*0.1)){

		for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
		  ecid_xt = p->first;
		  rd_ped  = p->second;
		  int sm_num=ecid_xt.getID1();
		  int xt_num=ecid_xt.getID2(); 
		  
		  EcalPedestals::Item item;
		  item.mean_x1  =rd_ped.getPedMeanG1() ;
		  item.rms_x1   =rd_ped.getPedRMSG1();
		  item.mean_x6  =rd_ped.getPedMeanG6();
		  item.rms_x6   =rd_ped.getPedRMSG6() ;
		  item.mean_x12 =rd_ped.getPedMeanG12();
		  item.rms_x12  =rd_ped.getPedRMSG12();
		  
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
		}
		
		cout << "Generating popcon record for run " << irun << "..." << flush;
		
		// now I copy peds in pedtemp and I ship pedtemp to popcon
		// if I use always the same peds I always overwrite
		// so I really have to create new objects for each new run
		// popcon deletes everything for me 
		
		EcalPedestals* pedtemp = new EcalPedestals();
		
		for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
		  if(iEta==0) continue;
		  for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
		    // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
		    if (EBDetId::validDetId(iEta,iPhi))
		      {
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
			  cout<< "channel:" <<iEta<<"/"<<iPhi<< "/" << hiee << " ped mean 12="<< x << endl;
			}
		      }
		  }
		}


	      Time_t snc= (Time_t) irun ;
	      	      
	      m_to_transfer.push_back(std::make_pair((EcalPedestals*)pedtemp,snc));
	      

	      ss << "Run=" << irun << "_WAS_GOOD_"<<endl; 
	      m_userTextLog = ss.str()+";";


	      
	      } else {

		cout<< "Run was BAD !!!! not sent to the DB"<< endl;
		ss << "Run=" << irun << "_WAS_BAD_"<<endl; 
		m_userTextLog = ss.str()+";";
	      }
	    }
	  }
	  
	}
	
	
	delete econn;

	delete peds;  // this is the only one that popcon does not delete 

	std::cout << "Ecal - > end of getNewObjects -----------\n";
	      
	
}


void popcon::EcalPedestalsHandler::getNewObjectsH2()
{
	int max_since=0;
	max_since=(int)tagInfo().lastInterval.first;
	std::cout << "max_since : "  << max_since << endl;
	Ref ped_db = lastPayload();
	
	std::cout << "retrieved last payload "  << endl;


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
	      if(iX==ixmin && iY==iymin) cout<<"ped12 " << item.mean_x12<< endl;  
	      
	    }
	  }
	}
	

	cout <<"WOW: we just retrieved the last valid record from DB "<< endl;


	// here we retrieve all the runs after the last from online DB 

	cout << "Retrieving run list from ONLINE DB ... " << endl;
	
	cout << "Making connection..." << flush;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	cout << "Done." << endl;

	if (!econn)
	  {
	    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
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
	int min_run=(int)max_since+1; // we have to add 1 to the last transferred one 
	
	int max_run=(int)m_lastRun;
	mon_list=econn->fetchMonRunList(my_runtag, mon_tag,min_run,max_run );
      
	std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();
	int mon_runs=mon_run_vec.size();
	cout <<"number of Mon runs is : "<< mon_runs<< endl;

	if(mon_runs>0){

	  for(int kr=0; kr<mon_runs; kr++){

	    unsigned long irun=(unsigned long) mon_run_vec[kr].getRunIOV().getRunNumber();
	  
	    cout << "here is first sub run : "<< mon_run_vec[kr].getSubRunNumber() << endl;
	    cout << "here is the run number: "<< mon_run_vec[kr].getRunIOV().getRunNumber() << endl;
	  
	    cout <<" retrieve the data for a given run"<< endl;
	  
	    if (mon_run_vec[kr].getSubRunNumber() <=1){ 


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
	      int ix=0;
	      int iy=0;
	      
	      for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
		ecid_xt = p->first;
		rd_ped  = p->second;
		int sm_num=ecid_xt.getID1();
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
		if(ix==ixmin && iy==iymin) cout<<"ped12 " << item.mean_x12<< endl;  
	      }
	    
	      cout << "Generating popcon record for run " << irun << "..." << flush;


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
		      if(iX==ixmin && iY==iymin) cout<<"ped12 " << item.mean_x12<< endl;  
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
