#include "CondTools/Ecal/interface/EcalDCSHandler.h"
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatusHelper.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"



#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<iostream>

popcon::EcalDCSHandler::EcalDCSHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalDCSHandler")) {

	std::cout << "EcalDCS Source handler constructor\n" << std::endl;
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");

        std::cout << m_sid<<"/"<<m_user<<"/"<<m_pass<<std::endl;


}

popcon::EcalDCSHandler::~EcalDCSHandler()
{
}

void popcon::EcalDCSHandler::printHVDataSet( const map<EcalLogicID, RunDCSHVDat>* dataset, 
					     int limit = 0 ) const
{
  cout << "==========printDataSet()" << endl;
  if (dataset->size() == 0) {
    cout << "No data in map!" << endl;
  }
  EcalLogicID ecid;
  RunDCSHVDat hv;
  
  int count = 0;
  typedef map< EcalLogicID, RunDCSHVDat >::const_iterator CI;
  for (CI p = dataset->begin(); p != dataset->end(); p++) {
    count++;
    if (limit && count > limit) { return; }
    ecid = p->first;
    hv  = p->second;
    
    cout << "SM:                     " << ecid.getID1() << endl;
    cout << "Channel:                " << ecid.getID2() << endl;
    cout << "HV:                     " << hv.getHV() << endl;
    cout << "HV nominal:             " << hv.getHVNominal() << endl;
    cout << "HV status:              " << hv.getStatus() << endl;
    cout << "========================" << endl;
  }
  cout << endl;
}

void popcon::EcalDCSHandler::printLVDataSet( const map<EcalLogicID, RunDCSLVDat>* dataset, 
					     int limit = 0 ) const
{
  cout << "==========printDataSet()" << endl;
  if (dataset->size() == 0) {
    cout << "No data in map!" << endl;
  }
  EcalLogicID ecid;
  RunDCSLVDat lv;
  
  int count = 0;
  typedef map< EcalLogicID, RunDCSLVDat >::const_iterator CI;
  for (CI p = dataset->begin(); p != dataset->end(); p++) {
    count++;
    if (limit && count > limit) { return; }
    ecid = p->first;
    lv  = p->second;
    
    cout << "SM:                     " << ecid.getID1() << endl;
    cout << "Channel:                " << ecid.getID2() << endl;
    cout << "LV:                     " << lv.getLV() << endl;
    cout << "LV nominal:             " << lv.getLVNominal() << endl;
    cout << "LV status:              " << lv.getStatus() << endl;
    cout << "========================" << endl;
    }
    cout << endl;
  }

uint16_t popcon::EcalDCSHandler::OffDBStatus( uint16_t dbStatus , int pos ) {
  uint16_t hv_off_dbstatus = ( dbStatus & (1 << pos ) ) ;
  if(hv_off_dbstatus>0) hv_off_dbstatus=1;
  return hv_off_dbstatus;
}  

uint16_t  popcon::EcalDCSHandler::updateHV( RunDCSHVDat* hv, uint16_t dbStatus) const {
  uint16_t result=0; 
  uint16_t hv_on_dbstatus=0;
  uint16_t hv_nomi_on_dbstatus=0;
  uint16_t hv_time_on_dbstatus=0;
  if(  hv->getStatus()==RunDCSHVDat::HVNOTNOMINAL ) hv_nomi_on_dbstatus=1; 
  if(  hv->getStatus()==RunDCSHVDat::HVOFF ) hv_on_dbstatus=1; 
  if(  hv->getTimeStatus()<0 ) hv_time_on_dbstatus=1; // information not updated 
   
  

  uint16_t hv_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::HVSTATUS ) ) ;
  uint16_t hv_time_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::HVTIMESTATUS ) ) ;
  uint16_t hv_nomi_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::HVNOMINALSTATUS ) ) ;
  if(hv_off_dbstatus>0) hv_off_dbstatus=1;
  if(hv_time_off_dbstatus>0) hv_time_off_dbstatus=1;
  if(hv_nomi_off_dbstatus>0) hv_nomi_off_dbstatus=1;

  
  uint16_t temp=0;
  for (int i=0; i<16; i++) {
    if( i!= EcalDCSTowerStatusHelper::HVSTATUS && i!= EcalDCSTowerStatusHelper::HVTIMESTATUS && i!=EcalDCSTowerStatusHelper::HVNOMINALSTATUS ) {
      temp = temp | (1<<i) ;  
    } else if( i== EcalDCSTowerStatusHelper::HVTIMESTATUS) {
      temp = temp | ( 0 << EcalDCSTowerStatusHelper::HVTIMESTATUS );
    } else if( i== EcalDCSTowerStatusHelper::HVNOMINALSTATUS) {
      temp = temp | ( 0 << EcalDCSTowerStatusHelper::HVNOMINALSTATUS );
    } else if ( i== EcalDCSTowerStatusHelper::HVSTATUS) {
      temp = temp | ( 0 << EcalDCSTowerStatusHelper::HVSTATUS );
    } 
  }


   result=  dbStatus & temp  ;
   result= ( ( result | ( hv_time_on_dbstatus << EcalDCSTowerStatusHelper::HVTIMESTATUS )) |  (  hv_on_dbstatus << EcalDCSTowerStatusHelper::HVSTATUS ) )   | ( hv_nomi_on_dbstatus << EcalDCSTowerStatusHelper::HVNOMINALSTATUS );

   std::cout << "HV status "<<hv_on_dbstatus<<"/"<<hv_time_on_dbstatus<<"/"<<hv_nomi_on_dbstatus<<
     "HV before status "<<hv_off_dbstatus<<"/"<<hv_time_off_dbstatus<<"/"<<hv_nomi_off_dbstatus<<" res="<< result<< endl;
  
  return result; 
}


uint16_t  popcon::EcalDCSHandler::updateLV( RunDCSLVDat* lv, uint16_t dbStatus) const {
  uint16_t result=0; 
  uint16_t lv_on_dbstatus=0;
  uint16_t lv_nomi_on_dbstatus=0;
  uint16_t lv_time_on_dbstatus=0;
  if(  lv->getStatus()==RunDCSLVDat::LVNOTNOMINAL ) lv_nomi_on_dbstatus=1; 
  if(  lv->getStatus()==RunDCSLVDat::LVOFF ) lv_on_dbstatus=1; 
  if(  lv->getTimeStatus()<0 ) lv_time_on_dbstatus=1; // information not updated 
   
  

  uint16_t lv_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::LVSTATUS ) ) ;
  uint16_t lv_time_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::LVTIMESTATUS ) ) ;
  uint16_t lv_nomi_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::LVNOMINALSTATUS ) ) ;
  if(lv_off_dbstatus>0) lv_off_dbstatus=1;
  if(lv_time_off_dbstatus>0) lv_time_off_dbstatus=1;
  if(lv_nomi_off_dbstatus>0) lv_nomi_off_dbstatus=1;

  
  uint16_t temp=0;
  for (int i=0; i<16; i++) {
    if( i!= EcalDCSTowerStatusHelper::LVSTATUS && i!= EcalDCSTowerStatusHelper::LVTIMESTATUS && i!=EcalDCSTowerStatusHelper::LVNOMINALSTATUS ) {
      temp = temp | (1<<i) ;  
    } else if( i== EcalDCSTowerStatusHelper::LVTIMESTATUS) {
      temp = temp | ( 0 << EcalDCSTowerStatusHelper::LVTIMESTATUS );
    } else if( i== EcalDCSTowerStatusHelper::LVNOMINALSTATUS) {
      temp = temp | ( 0 << EcalDCSTowerStatusHelper::LVNOMINALSTATUS );
    } else if ( i== EcalDCSTowerStatusHelper::LVSTATUS) {
      temp = temp | ( 0 << EcalDCSTowerStatusHelper::LVSTATUS );
    } 
  }


   result=  dbStatus & temp  ;
   result= ( ( result | ( lv_time_on_dbstatus << EcalDCSTowerStatusHelper::LVTIMESTATUS )) |  (  lv_on_dbstatus << EcalDCSTowerStatusHelper::LVSTATUS ) )   | ( lv_nomi_on_dbstatus << EcalDCSTowerStatusHelper::LVNOMINALSTATUS );

   std::cout << "LV status "<<lv_on_dbstatus<<"/"<<lv_time_on_dbstatus<<"/"<<lv_nomi_on_dbstatus<<
     "LV before status "<<lv_off_dbstatus<<"/"<<lv_time_off_dbstatus<<"/"<<lv_nomi_off_dbstatus<<" res="<< result<< endl;
  
  return result; 
}

bool popcon::EcalDCSHandler::insertHVDataSetToOffline( const map<EcalLogicID, RunDCSHVDat>* dataset, EcalDCSTowerStatus* dcs_temp ) const
{
  bool result=false; 
  if (dataset->size() == 0) {
    cout << "No data in map!" << endl;
  }
  EcalLogicID ecid;
  RunDCSHVDat hv;


  typedef std::map< EcalLogicID, RunDCSHVDat >::const_iterator CI ;

  for (CI p = dataset->begin(); p != dataset->end(); p++) {


    ecid = p->first;
    hv  = p->second;

    if(ecid.getName()=="EB_HV_channel"){
      int sm= ecid.getID1() ;
      int chan= ecid.getID2();
      
      int* limits=0;
      limits=  HVLogicIDToDetID(sm,chan);
      int iz=limits[0];
      int i1=limits[1];
      int i2=limits[2];
      int j=limits[3];


      for(int ik=i1; ik<=i2; ik++){ 
	if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,ik )){
	  EcalTrigTowerDetId ebid(iz,EcalBarrel,j,ik);
	  EcalDCSTowerStatus::const_iterator it =dcs_temp->find(ebid.rawId());
	  
	  uint16_t dbStatus = 0;
	  if ( it != dcs_temp->end() ) {
	    dbStatus = it->getStatusCode();
	  }
	  uint16_t new_dbStatus= updateHV(&hv, dbStatus); 
	  if(new_dbStatus != dbStatus ) result=true; 

	  dcs_temp->setValue( ebid, new_dbStatus );

	  std::cout <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<< endl; 
	}
      }

      delete [] limits; 


    } else {
      // endcaps to be done 
    }
  }
  return result; 
}

int popcon::EcalDCSHandler::detIDToLogicID(int iz, int i, int j) {
  // returns the number from 0 to 1223 from SM1 to 36 from ch 1 to 34 

  int sm=0;
  int hv_chan=0;


  sm = (i-1)/4;
  if(iz<0) sm=sm+18;
  
  int ilocal=(i-1)-sm*4;
  if(iz<0){
    if(ilocal==0 || ilocal==1) hv_chan=1;
    if(ilocal==2 || ilocal==3) hv_chan=2;
  } else {
    if(ilocal==0 || ilocal==1) hv_chan=2;
    if(ilocal==2 || ilocal==3) hv_chan=1;
  }

  sm=sm+1; 

  hv_chan=(j-1)*2+hv_chan;
  
  hv_chan=(sm-1)*34+hv_chan -1  ;
  
  return hv_chan; 

}

int * popcon::EcalDCSHandler::HVLogicIDToDetID(int sm, int chan) const {
  // returns the numbers iz, i1, i2 and j1, j2 on which to loop for the towers

      int iz=-1;
      if(sm>0 && sm <= 18) iz=1;
      int j = (chan-1)/2 +1;
      int i_local_hv = (chan-1) - (j-1)*2 + 1; // this gives 1 for odd channels and 2 for even channels 
      int i1 =0;
      int i2 =0;
      if( iz>0 ) { // EB plus phi turns opposite to HV numbering
	if(i_local_hv==1) { 
	  i1=3 ; 
	  i2=4 ; 
	} else { 
	  i1=1 ; 
	  i2=2 ; 
	}
      } else { // EB minus phi turns as HV numbering 
	if(i_local_hv==1) { 
	  i1=1 ; 
	  i2=2 ; 
	} else { 
	  i1=3 ; 
	  i2=4 ; 
	}
      }
      int ioffset=0;
      if(iz==1) ioffset=(sm-1)*4; 
      if(iz==-1) ioffset=(sm-18-1)*4; 
      i1=i1+ioffset;
      i2=i2+ioffset;

      int *result = new int[5];
      
      result[0]=iz;
      result[1]=i1;
      result[2]=i2;
      result[3]=j;
      result[4]=j;

      return result; 

}

int * popcon::EcalDCSHandler::LVLogicIDToDetID(int sm, int chan) const {
  // returns the numbers iz, i1, i2 and j1, j2 on which to loop for the towers

      int iz=-1;
      if(sm>0 && sm <= 18) iz=1;

      int j1=0;
      int j2=0;
      int i1=0;
      int i2=0;

      if(chan==1) {
	i1=1;
	i2=4;
	j1=1;
	j2=1;
      } else {
	int ch2= (chan/2)*2;
	if(ch2==chan) {
	  j1=chan;
	} else {
	  j1=chan-1;
	}
 	j2 = j1+1; 
	if( iz>0 ) { // EB plus phi turns opposite to LV numbering
	  if(ch2==chan) { 
	    i1=3 ; 
	    i2=4 ; 
	  } else { 
	    i1=1 ; 
	    i2=2 ; 
	  }
	} else { // EB minus phi turns as HV numbering 
	  if(ch2==chan) { 
	    i1=1 ; 
	    i2=2 ; 
	  } else { 
	    i1=3 ; 
	    i2=4 ; 
	  }
	}
      }
      int ioffset=0;
      if(iz==1) ioffset=(sm-1)*4; 
      if(iz==-1) ioffset=(sm-18-1)*4; 
      i1=i1+ioffset;
      i2=i2+ioffset;


      int *result = new int[5];
      result[0]=iz;
      result[1]=i1;
      result[2]=i2;
      result[3]=j1;
      result[4]=j2;

      return result; 
}


bool popcon::EcalDCSHandler::insertLVDataSetToOffline( const map<EcalLogicID, RunDCSLVDat>* dataset, EcalDCSTowerStatus* dcs_temp ) const
{
  bool result= false; 
  if (dataset->size() == 0) {
    cout << "No data in map!" << endl;
  }
  EcalLogicID ecid;
  RunDCSLVDat lv;


  typedef map< EcalLogicID, RunDCSLVDat >::const_iterator CI;
  for (CI p = dataset->begin(); p != dataset->end(); p++) {

    ecid = p->first;
    lv  = p->second;

    if(ecid.getName()=="EB_LV_channel"){

      int sm= ecid.getID1() ;
      int chan= ecid.getID2();

      int* limits=0;
      limits=   LVLogicIDToDetID(sm,chan);
      int iz=limits[0];
      int i1=limits[1];
      int i2=limits[2];
      int j1=limits[3];
      int j2=limits[4];

      for(int ik=i1; ik<=i2; ik++){ 
	for(int j=j1; j<=j2; j++){ 
	  if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,ik )){
	    EcalTrigTowerDetId ebid(iz,EcalBarrel,j,ik);
	    EcalDCSTowerStatus::const_iterator it =dcs_temp->find(ebid.rawId());
	    uint16_t dbStatus = 0;
	    if ( it != dcs_temp->end() ) {
	      dbStatus = it->getStatusCode();
	    }
	    uint16_t new_dbStatus= updateLV(&lv, dbStatus); 
	    if(new_dbStatus != dbStatus ) result=true; 
	    dcs_temp->setValue( ebid, new_dbStatus );
	  }
	}
      }
    delete [] limits; 

      
    } else {
	
	// endcaps to be done 
	
	
    }



  }
  return result; 
}

void popcon::EcalDCSHandler::getNewObjects()
{
  bool lot_of_printout=false; 
	std::cout << "------- Ecal DCS - > getNewObjects\n";

	std::ostringstream ss; 
	ss<<"ECAL ";

	unsigned long long max_since= 1;

	max_since=tagInfo().lastInterval.first;
	std::cout << "max_since : "  << max_since << endl;
	Ref dcs_db = lastPayload();
	std::cout << "retrieved last payload "  << endl;
	
	// we copy the last valid record to a temporary object peds
	EcalDCSTowerStatus* dcs_temp = new EcalDCSTowerStatus();

        // barrel
        int iz=0;
        for(int k=0 ; k<2; k++ ) {
          if(k==0) iz=-1;
          if(k==1) iz= 1;
          for(int i=1 ; i<73; i++) {
            for(int j=1 ; j<18; j++) {
              if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,i )){
                EcalTrigTowerDetId ebid(iz,EcalBarrel,j,i);

		uint16_t dbStatus = 0;
		dbStatus =(dcs_db->barrel( ebid.hashedIndex())).getStatusCode();


		EcalDCSTowerStatus::const_iterator it =dcs_db->find(ebid.rawId());
		if ( it != dcs_db->end() ) {
		} else {
		  std::cout<<"*** error channel not found: j/i="<<j<<"/"<<i << endl;
		}
		
                dcs_temp->setValue( ebid, dbStatus );
              }
            }
          }
        }

        // endcap
        for(int k=0 ; k<2; k++ ) {
          if(k==0) iz=-1;
          if(k==1) iz=+1;
          for(int i=1 ; i<21; i++) {
            for(int j=1 ; j<21; j++) {
              if (EcalScDetId::validDetId(i,j,iz )){
                EcalScDetId eeid(i,j,iz);

		EcalDCSTowerStatus::const_iterator it =dcs_db->find(eeid.rawId());

		uint16_t dbStatus = 0;
		if ( it != dcs_db->end() ) {
		  dbStatus = it->getStatusCode();
		} 
                dcs_temp->setValue( eeid, dbStatus );
              }
            }
          }
        }


	// now read the actual status from the online DB


	cout << "Retrieving DCS status from ONLINE DB ... " << endl;
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	cout << "Connection done" << endl;
	
	if (!econn)
	  {
	    cout << " Problem with OMDS: connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
	    throw cms::Exception("OMDS not available");
	  } 



	

	if(max_since<=1) { 
	  // retrieve from last value data record 	

	  map<EcalLogicID, RunDCSHVDat> dataset;
	  RunIOV *r = NULL;
	  econn->fetchDataSet(&dataset, r);
	  
	  if (!dataset.size()) {
	    throw(runtime_error("Zero rows read back"));
	  }
	  
	  
	  if(lot_of_printout) cout << "read OK" << endl;
	  if(lot_of_printout) printHVDataSet(&dataset,10);
	  
	  map<EcalLogicID, RunDCSLVDat> dataset_lv;
	  econn->fetchDataSet(&dataset_lv, r);
	  
	  if (!dataset_lv.size()) {
	    throw(runtime_error("Zero rows read back"));
	  }
	  if(lot_of_printout) cout << "read OK" << endl;
	  if(lot_of_printout) printLVDataSet(&dataset_lv);
	  
	  bool somediff_hv= insertHVDataSetToOffline(&dataset, dcs_temp );
	  bool somediff_lv= insertLVDataSetToOffline(&dataset_lv, dcs_temp );
	  
	  if(somediff_hv || somediff_lv) {
	    

	    Tm t_now_gmt;
	    t_now_gmt.setToCurrentGMTime();
	    uint64_t tsincetemp= t_now_gmt.microsTime()/1000000 ;
	    uint64_t tsince = tsincetemp<< 32; 

	    cout << "Generating popcon record for time " << tsincetemp << "..." << flush;
    
	    m_to_transfer.push_back(std::make_pair((EcalDCSTowerStatus*)dcs_temp,tsince));
	    
	    ss << "Time=" << t_now_gmt.str() << "_DCSchanged_"<<endl; 
	    m_userTextLog = ss.str()+";";

	  } else {

	    Tm t_now_gmt;
            t_now_gmt.setToCurrentGMTime();

	    cout<< "Run DCS record was the same as previous run " << endl;
	    ss << "Time=" << t_now_gmt.str() << "_DCSunchanged_"<<endl; 
	    m_userTextLog = ss.str()+";";
	    
	    delete dcs_temp; 
	    
	  }
	  
	  
	} else {
	  
	  // here we fetch historical data 

	  uint64_t t_max_val= ((max_since >> 32 ) +1)*1000000 ; // time in microseconds 
	
	  Tm t_test(t_max_val);
  
	  std::list< std::pair< Tm, std::map<  EcalLogicID, RunDCSHVDat > > > dataset;

	  econn->fetchDCSDataSet(&dataset, t_test);
	  
	  if (!dataset.size()) {
	    throw(runtime_error("Zero rows read back"));
	  }

	  std::list< std::pair< Tm, std::map<  EcalLogicID, RunDCSHVDat > > >::iterator it;
	  for (it=dataset.begin(); it!=dataset.end(); it++){
	    std::pair< Tm, std::map<  EcalLogicID, RunDCSHVDat > > a_pair =(*it);
	    Tm t_pair=a_pair.first;
	    std::map<  EcalLogicID, RunDCSHVDat > a_map = a_pair.second;
	    bool somediff_hv= insertHVDataSetToOffline(&a_map, dcs_temp );

	    if(somediff_hv ) {
	      // we have to copy this record to offline 
	      // we copy dcs_temp to dcs_pop
	      EcalDCSTowerStatus* dcs_pop = new EcalDCSTowerStatus();

	      // barrel
	      int iz=0;
	      for(int k=0 ; k<2; k++ ) {
		if(k==0) iz=-1;
		if(k==1) iz= 1;
		for(int i=1 ; i<73; i++) {
		  for(int j=1 ; j<18; j++) {
		    if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,i )){
		      EcalTrigTowerDetId ebid(iz,EcalBarrel,j,i);
		      
		      uint16_t dbStatus = 0;
		      dbStatus =(dcs_temp->barrel( ebid.hashedIndex())).getStatusCode();
		      
		      EcalDCSTowerStatus::const_iterator it =dcs_temp->find(ebid.rawId());
		      if ( it != dcs_temp->end() ) {
		      } else {
			std::cout<<"*** error channel not found: j/i="<<j<<"/"<<i << endl;
		      }
		      
		      dcs_pop->setValue( ebid, dbStatus );
		    }
		  }
		}
	      }
	      
	      // endcap
	      for(int k=0 ; k<2; k++ ) {
		if(k==0) iz=-1;
		if(k==1) iz=+1;
		for(int i=1 ; i<21; i++) {
		  for(int j=1 ; j<21; j++) {
		    if (EcalScDetId::validDetId(i,j,iz )){
		      EcalScDetId eeid(i,j,iz);
		      
		      EcalDCSTowerStatus::const_iterator it =dcs_temp->find(eeid.rawId());
		      
		      uint16_t dbStatus = 0;
		      if ( it != dcs_temp->end() ) {
			dbStatus = it->getStatusCode();
		      } 
		      dcs_pop->setValue( eeid, dbStatus );
		    }
		  }
		}
	      }

	      uint64_t tsincetemp= t_pair.microsTime()/1000000 ;

	      uint64_t tsince = tsincetemp<< 32; 
	      Tm tnew(t_pair.microsTime());

	      cout << "Generating popcon record for time " << tsince << "HRF time " << tnew.str() << "..." << flush;
    
	      m_to_transfer.push_back(std::make_pair((EcalDCSTowerStatus*)dcs_pop,tsince));
	    
	      ss << "Time=" << tnew.str() << "_DCSchanged_"<<endl; 
	      m_userTextLog = ss.str()+";";
	      



	    }

	  }

	  delete dcs_temp;
	  
	}

	delete econn;
	std::cout << "Ecal - > end of getNewObjects -----------\n";

}


