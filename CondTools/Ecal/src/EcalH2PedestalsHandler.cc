#include "CondTools/Ecal/interface/EcalH2PedestalsHandler.h"

popcon::EcalH2PedestalsHandler::EcalH2PedestalsHandler(const std::string& name,
						       const std::string& cstring, const edm::Event& evt, const edm::EventSetup& est, unsigned int firstRun,unsigned int lastRun, const std::string& sid, const std::string& user, const std::string& pass) : popcon::PopConSourceHandler<EcalPedestals>(name,cstring,evt,est)
{
	std::cout << "EcalPedestals Source handler constructor\n" << std::endl;

	m_firstRun=firstRun;
	m_lastRun=lastRun;
	m_sid= sid;
	m_user=user;
	m_pass=pass;
  

}

popcon::EcalH2PedestalsHandler::~EcalH2PedestalsHandler()
{
}

void popcon::EcalH2PedestalsHandler::getNewObjects()
{

	std::cout << "------- Ecal src - > getNewObjects\n";

	// pass the container name as a parameter !!!!
	std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();
	unsigned int max_since=0;
	for(std::map<std::string, popcon::PayloadIOV>::iterator it = mp.begin(); it != mp.end();it++)
	  {
	    if(it->second.container_name=="EcalPedestalsRcd" &&  max_since<it->second.last_since){
	      max_since=it->second.last_since;
	    }
	    std::cout << it->second.container_name << " , last object valid since " << it->second.last_since << std::endl;
	}



	// this is the offline object 

	
	// we copy the last valid record to a temporary object peds
	EcalPedestals* peds = new EcalPedestals();

	// get from offline DB the last valid pedestal set ped_db
	edm::ESHandle<EcalPedestals> pedestals;
	esetup.get<EcalPedestalsRcd>().get(pedestals);
	const EcalPedestals& ped_db= *pedestals.product(); // got the pedestals


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
	      EcalPedestals::Item aped= ped_db.endcap(hiee);

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
	      int snc= (unsigned int) irun ;
	      int tll= edm::IOVSyncValue::endOfTime().eventID().run();
	      popcon::IOVPair iop = {snc,tll};
	      
	      
	      m_to_transfer->push_back(std::make_pair((EcalPedestals*)pedtemp,iop));
	      
	      std::cout << "Ecal source - > getNewObjects -----------\n";
	      
	      
	    }
	  }
	  
	}
	
	
	delete econn;
	delete peds;  // this is the only one that popcon does not delete 
	
}
