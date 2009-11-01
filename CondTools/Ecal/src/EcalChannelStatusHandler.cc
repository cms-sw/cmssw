#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalChannelStatusHandler.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::EcalChannelStatusHandler::EcalChannelStatusHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalChannelStatusHandler")) {

	std::cout << "EcalChannelStatus Source handler constructor\n" << std::endl;
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

popcon::EcalChannelStatusHandler::~EcalChannelStatusHandler()
{
}

bool popcon::EcalChannelStatusHandler::checkPedestal( EcalPedestals::Item* item )
{
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

void popcon::EcalChannelStatusHandler::getNewObjects()
{

  std::ostringstream ss; 
  ss<<"ECAL ";

	int max_since=0;
	max_since=(int)tagInfo().lastInterval.first;
	std::cout << "max_since : "  << max_since << endl;
	Ref chan_db = lastPayload();
	
	// we copy the last valid record to a temporary object peds
	EcalChannelStatus* chans = new EcalChannelStatus();
	std::cout << "retrieved last payload "  << endl;

	for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	  if(iEta==0) continue;
	  for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	    if (EBDetId::validDetId(iEta,iPhi))
	      {
		EBDetId ebdetid(iEta,iPhi,EBDetId::ETAPHIMODE);
		EcalChannelStatus::const_iterator it =chan_db->find(ebdetid.rawId());
	
		// this is the last object in the DB
		EcalChannelStatusCode achan = (*it);

                // here I copy the last valid value in the new object
		EcalChannelStatusCode item;
		uint16_t ch_status =achan.getStatusCode();
		item=EcalChannelStatusCode(ch_status);

		chans->insert(std::make_pair(ebdetid.rawId(),item));
	
	      }
	  }
	}


	for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	  for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
	    // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
	    if (EEDetId::validDetId(iX,iY,1))
	      {
		EEDetId eedetidpos(iX,iY,1);

		EcalChannelStatus::const_iterator it =chan_db->find(eedetidpos.rawId());
		EcalChannelStatusCode achan = (*it);

		// here I copy the last valid value in the new object
		EcalChannelStatusCode item;
		uint16_t ch_status =achan.getStatusCode();
                item=EcalChannelStatusCode(ch_status);

		chans->insert(std::make_pair(eedetidpos.rawId(),item));

	      }

	    if(EEDetId::validDetId(iX,iY,-1))
	      {
		EEDetId eedetidneg(iX,iY,-1);

		EcalChannelStatus::const_iterator it =chan_db->find(eedetidneg.rawId());
		EcalChannelStatusCode achan = (*it);

		// here I copy the last valid value in the new object
		EcalChannelStatusCode item;
		uint16_t ch_status =achan.getStatusCode();
                item=EcalChannelStatusCode(ch_status);

		chans->insert(std::make_pair(eedetidneg.rawId(),item));

	      }
	  }
	}

	// here we retrieve all the pedestals runs after the last from online DB 
	cout << "Retrieving pedestals run list from ONLINE DB ... " << endl;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	cout << "Connection done" << endl;
	
	if (!econn)
	  {
	    cout << " Problem with OMDS: connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
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
	  
	    cout << "retrieve the data for run number: "<< mon_run_vec[kr].getRunIOV().getRunNumber() << endl;
	  
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
		int yt_num=ecid_xt.getID3(); 

		EcalPedestals::Item item;
		item.mean_x1  =rd_ped.getPedMeanG1() ;
		item.rms_x1   =rd_ped.getPedRMSG1();
		item.mean_x6  =rd_ped.getPedMeanG6();
		item.rms_x6   =rd_ped.getPedRMSG6() ;
		item.mean_x12 =rd_ped.getPedMeanG12();
		item.rms_x12  =rd_ped.getPedRMSG12();
		

		// here we check and count how many bad channels we have 

		if(!checkPedestal(&item) ){
		  nbad++;
		  if(nbad<50) cout <<"BAD LIST: channel " << sm_num << "/" << xt_num << "/"<< yt_num 
				   <<  "ped/rms "<<item.mean_x12<< "/"<< item.rms_x12 << endl;
		}
	      }
	    
	      // ok or bad? // maybe it was a bad pedestals runs  
	      // a bad run is for more than 10% bad channels 


	      if(nbad<(dataset_mon.size()*0.1)){

		// here do what you want .... compute etc 


		for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
		  ecid_xt = p->first;
		  rd_ped  = p->second;
		  int sm_num=ecid_xt.getID1();
		  int xt_num=ecid_xt.getID2(); 
		  int yt_num=ecid_xt.getID3(); 
		  

		  EcalChannelStatusCode item;
		  // fill the status for example noisy if it is noisy 
		  uint16_t status_now=0;

		  EcalPedestals::Item ped_item;
		  ped_item.mean_x1  =rd_ped.getPedMeanG1() ;
		  ped_item.rms_x1   =rd_ped.getPedRMSG1();
		  ped_item.mean_x6  =rd_ped.getPedMeanG6();
		  ped_item.rms_x6   =rd_ped.getPedRMSG6() ;
		  ped_item.mean_x12 =rd_ped.getPedMeanG12();
		  ped_item.rms_x12  =rd_ped.getPedRMSG12();
		
		  if( checkPedestal(&ped_item) ) status_now=0; // all OK  
		  if ( rd_ped.getPedRMSG12() > 3) status_now=5; // for example if status_now=5 means that it is noisy 


		  if(ecid_xt.getName()=="EB_crystal_number") {
		    // Barrel channel 
		    EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
		    
		    // here we change in the peds object only the channels that are available in the online DB 
		    // otherwise we keep the previous value 
		  
		    chans->insert(std::make_pair(ebdetid.rawId(),item)); 


		  } else {

		    // Endcap channel 
		    // in this case xt_num is x 
		    // yt_num is y and sm_num is the side +/- 1 
		    if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
		      EEDetId eedetid(xt_num,yt_num,sm_num);
		      
		      // here we change in the peds object only the channels that are available in the online DB 
		      // otherwise we keep the previous value 
		      chans->insert(std::make_pair(eedetid.rawId(),item)); 
		    }
		    
		  }		       
		  
		}
	      
		cout << "Generating popcon record for run " << irun << "..." << flush;
		
		// now I copy peds in pedtemp and I ship pedtemp to popcon
		// if I use always the same peds I always overwrite
		// so I really have to create new objects for each new run
		// popcon deletes everything for me 

		EcalChannelStatus* chantemp = new EcalChannelStatus();
		
		for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
		  if(iEta==0) continue;
		  for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
		    // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
		    if (EBDetId::validDetId(iEta,iPhi))
		      {
			EBDetId ebdetid(iEta,iPhi);
			unsigned int hiee = ebdetid.hashedIndex();
			EcalChannelStatusCode achan= chans->barrel(hiee);
			
			EcalChannelStatusCode item;
			uint16_t ch_status =achan.getStatusCode();
			item=EcalChannelStatusCode(ch_status);

			chantemp->insert(std::make_pair(ebdetid.rawId(),item));
			
		      }
		  }
		}
		// endcaps 
		for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
		  for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
		    // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
		    if (EEDetId::validDetId(iX,iY,1))
		      {
			EEDetId eedetid(iX,iY,1);
			unsigned int hiee = eedetid.hashedIndex();

			EcalChannelStatusCode achan= chans->endcap(hiee);
			
			EcalChannelStatusCode item;
			uint16_t ch_status =achan.getStatusCode();
			item=EcalChannelStatusCode(ch_status);

			chantemp->insert(std::make_pair(eedetid.rawId(),item));

		      }
		    if (EEDetId::validDetId(iX,iY,-1))
		      {
			EEDetId eedetid(iX,iY,-1);
			unsigned int hiee = eedetid.hashedIndex();

			EcalChannelStatusCode achan= chans->endcap(hiee);
			
			EcalChannelStatusCode item;
			uint16_t ch_status =achan.getStatusCode();
			item=EcalChannelStatusCode(ch_status);

			chantemp->insert(std::make_pair(eedetid.rawId(),item));

		      }
		  }
		}

		Time_t snc= (Time_t) irun ;
	      	      
		m_to_transfer.push_back(std::make_pair((EcalChannelStatus*)chantemp,snc));
	      

		ss << "Run=" << irun << "_WAS_GOOD_"<<endl; 
		m_userTextLog = ss.str()+";";
		
	      
	      
		} else {
		  
		  cout<< "Run was BAD !!!! not sent to the DB number of bad channels="<<nbad << endl;
		  ss << "Run=" << irun << "_WAS_BAD_"<<endl; 
		  m_userTextLog = ss.str()+";";
		}
	    }
	  }
	    
	
	  delete econn;
	  delete chans;  // this is the only one that popcon does not delete 

	}
	  std::cout << "Ecal - > end of getNewObjects -----------\n";

	
}
