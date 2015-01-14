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

        std::cout << m_sid<<"/"<<m_user<<std::endl;


}

popcon::EcalDCSHandler::~EcalDCSHandler()
{
}

void popcon::EcalDCSHandler::printHVDataSet( const std::map<EcalLogicID, RunDCSHVDat>* dataset, 
					     int limit = 0 ) const
{
  std::cout << "==========printDataSet()" << std::endl;
  if (dataset->size() == 0) {
    std::cout << "No data in map!" << std::endl;
  }
  EcalLogicID ecid;
  RunDCSHVDat hv;
  
  int count = 0;
  typedef std::map< EcalLogicID, RunDCSHVDat >::const_iterator CI;
  for (CI p = dataset->begin(); p != dataset->end(); ++p) {
    count++;
    if (limit && count > limit) { return; }
    ecid = p->first;
    hv  = p->second;
    
    std::cout << "SM:                     " << ecid.getID1() << std::endl;
    std::cout << "Channel:                " << ecid.getID2() << std::endl;
    std::cout << "HV:                     " << hv.getHV() << std::endl;
    std::cout << "HV nominal:             " << hv.getHVNominal() << std::endl;
    std::cout << "HV status:              " << hv.getStatus() << std::endl;
    std::cout << "========================" << std::endl;
  }
  std::cout << std::endl;
}

void popcon::EcalDCSHandler::printLVDataSet( const std::map<EcalLogicID, RunDCSLVDat>* dataset, 
					     int limit = 0 ) const
{
  std::cout << "==========printDataSet()" << std::endl;
  if (dataset->size() == 0) {
    std::cout << "No data in map!" << std::endl;
  }
  EcalLogicID ecid;
  RunDCSLVDat lv;
  
  int count = 0;
  typedef std::map< EcalLogicID, RunDCSLVDat >::const_iterator CI;
  for (CI p = dataset->begin(); p != dataset->end(); ++p) {
    count++;
    if (limit && count > limit) { return; }
    ecid = p->first;
    lv  = p->second;
    
    std::cout << "SM:                     " << ecid.getID1() << std::endl;
    std::cout << "Channel:                " << ecid.getID2() << std::endl;
    std::cout << "LV:                     " << lv.getLV() << std::endl;
    std::cout << "LV nominal:             " << lv.getLVNominal() << std::endl;
    std::cout << "LV status:              " << lv.getStatus() << std::endl;
    std::cout << "========================" << std::endl;
    }
    std::cout << std::endl;
  }

uint16_t popcon::EcalDCSHandler::OffDBStatus( uint16_t dbStatus , int pos ) {
  uint16_t hv_off_dbstatus = ( dbStatus & (1 << pos ) ) ;
  if(hv_off_dbstatus>0) hv_off_dbstatus=1;
  return hv_off_dbstatus;
}  

uint16_t  popcon::EcalDCSHandler::updateHV( RunDCSHVDat* hv, uint16_t dbStatus, int mode) const {
  // mode ==0 EB ,  mode==1 EE Anode , mode==2 EE Dynode 

  uint16_t result=0; 
  uint16_t hv_on_dbstatus=0;
  uint16_t hv_nomi_on_dbstatus=0;

  if(  hv->getStatus()==RunDCSHVDat::HVNOTNOMINAL ) hv_nomi_on_dbstatus=1; 
  if(  hv->getStatus()==RunDCSHVDat::HVOFF ) hv_on_dbstatus=1; 


  
  uint16_t temp=0;

  if(mode == 0 || mode == 1) {
    for (int i=0; i<16; i++) {
      if( i!= EcalDCSTowerStatusHelper::HVSTATUS &&  i!=EcalDCSTowerStatusHelper::HVNOMINALSTATUS  ) {
	temp = temp | (1<<i) ;  
      } else {
	temp = temp | (0<<i);
      } 
    }
    result=  dbStatus & temp  ;
    result=  ( result | (  hv_on_dbstatus << EcalDCSTowerStatusHelper::HVSTATUS ) )   | ( hv_nomi_on_dbstatus << EcalDCSTowerStatusHelper::HVNOMINALSTATUS )    ;
  } else {
    for (int i=0; i<16; i++) {
      if( i!=EcalDCSTowerStatusHelper::HVEEDNOMINALSTATUS &&  i!= EcalDCSTowerStatusHelper::HVEEDSTATUS ) {
	temp = temp | (1<<i) ;  
      } else {
	temp = temp | (0<<i);
      } 
    }
    result=  dbStatus & temp  ;
    result= ( result | ( hv_on_dbstatus << EcalDCSTowerStatusHelper::HVEEDSTATUS )) |  (  hv_nomi_on_dbstatus << EcalDCSTowerStatusHelper::HVEEDNOMINALSTATUS )   ;
  }

  return result; 
}


uint16_t  popcon::EcalDCSHandler::updateLV( RunDCSLVDat* lv, uint16_t dbStatus) const {
  uint16_t result=0; 
  uint16_t lv_on_dbstatus=0;
  uint16_t lv_nomi_on_dbstatus=0;
  if(  lv->getStatus()==RunDCSLVDat::LVNOTNOMINAL ) lv_nomi_on_dbstatus=1; 
  if(  lv->getStatus()==RunDCSLVDat::LVOFF ) lv_on_dbstatus=1; 

  uint16_t lv_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::LVSTATUS ) ) ;
  uint16_t lv_nomi_off_dbstatus = ( dbStatus & (1 << EcalDCSTowerStatusHelper::LVNOMINALSTATUS ) ) ;
  if(lv_off_dbstatus>0) lv_off_dbstatus=1;
  if(lv_nomi_off_dbstatus>0) lv_nomi_off_dbstatus=1;

  
  uint16_t temp=0;
  for (int i=0; i<16; i++) {
    if( i!= EcalDCSTowerStatusHelper::LVSTATUS &&  i!=EcalDCSTowerStatusHelper::LVNOMINALSTATUS ) {
      temp = temp | (1<<i) ;  
    } else {
      temp = temp | ( 0 << i );
    } 
  }


   result=  dbStatus & temp  ;
   result=  ( result | (  lv_on_dbstatus << EcalDCSTowerStatusHelper::LVSTATUS ) )   | ( lv_nomi_on_dbstatus << EcalDCSTowerStatusHelper::LVNOMINALSTATUS ) ;

  
  return result; 
}

bool popcon::EcalDCSHandler::insertHVDataSetToOffline( const std::map<EcalLogicID, RunDCSHVDat>* dataset, EcalDCSTowerStatus* dcs_temp ) const
{
  bool result=false; 
  if (dataset->size() == 0) {
    std::cout << "No data in std::map!" << std::endl;
  }
  EcalLogicID ecid;
  RunDCSHVDat hv;


  typedef std::map< EcalLogicID, RunDCSHVDat >::const_iterator CI ;

  for (CI p = dataset->begin(); p != dataset->end(); ++p) {


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
	  int modo=0;
	  uint16_t new_dbStatus= updateHV(&hv, dbStatus, modo); 
	  if(new_dbStatus != dbStatus ) result=true; 

	  dcs_temp->setValue( ebid, new_dbStatus );

	  if(new_dbStatus != dbStatus) {
	    std::cout <<"SM/chan:"<<sm<<"/"<<chan <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<<" HV: "<< hv.getHV()<<"/"<<hv.getHVNominal()<<std::endl;
	    
	  } 
	}
      }
      delete [] limits; 
    } else {
      // endcaps 
      int dee= ecid.getID1() ;
      int chan= ecid.getID2();
      
      int* limits=0;
      limits=  HVEELogicIDToDetID(dee,chan);
      int iz=limits[0];
      int i1=limits[1];
      int i2=limits[2];
      int j1=limits[3];
      int j2=limits[4];

      int ex_x[6];
      int ex_y[6];
      if(dee==1 ) {
	ex_x[0]=4;	ex_y[0]=8;
	ex_x[1]=4;	ex_y[1]=9;
	ex_x[2]=4;	ex_y[2]=10;
	ex_x[3]=5;	ex_y[3]=9;
	ex_x[4]=5;	ex_y[4]=10;
	ex_x[5]=6;	ex_y[5]=10;
      } else if(dee==2) {
	ex_x[0]=17;	ex_y[0]=11;
	ex_x[1]=17;	ex_y[1]=12;
	ex_x[2]=17;	ex_y[2]=13;
	ex_x[3]=16;	ex_y[3]=11;
	ex_x[4]=16;	ex_y[4]=12;
	ex_x[5]=15;	ex_y[5]=11;
      } else if(dee==3) {
	ex_x[0]=17;	ex_y[0]=8;
	ex_x[1]=17;	ex_y[1]=9;
	ex_x[2]=17;	ex_y[2]=10;
	ex_x[3]=16;	ex_y[3]=9;
	ex_x[4]=16;	ex_y[4]=10;
	ex_x[5]=15;	ex_y[5]=10;
      } else if(dee==4) {
	ex_x[0]=4;	ex_y[0]=11;
	ex_x[1]=4;	ex_y[1]=12;
	ex_x[2]=4;	ex_y[2]=13;
	ex_x[3]=5;	ex_y[3]=11;
	ex_x[4]=5;	ex_y[4]=12;
	ex_x[5]=6;	ex_y[5]=11;
      }

      int modo=1;
      if(ecid.getName()=="EE_HVD_channel") modo=2;

      for(int ik=i1; ik<=i2; ik++){ 
	for(int ip=j1; ip<=j2; ip++){
	  bool not_excluded=true;
	  if(chan==2 ) { // channel 2 has half a dee minus 6 towers
	    for (int l=0; l<6; l++){
	      if(ik== ex_x[l] && ip== ex_y[l] ) not_excluded=false;
	    }
	  }
	  if(not_excluded){
	    if (EcalScDetId::validDetId(ik,ip,iz)){
	      EcalScDetId eeid(ik,ip,iz);
	      EcalDCSTowerStatus::const_iterator it =dcs_temp->find(eeid.rawId());
	  
	      uint16_t dbStatus = 0;
	      if ( it != dcs_temp->end() ) {
		dbStatus = it->getStatusCode();
	      }
	      // FIXME - UPDATE HV A and D
	      uint16_t new_dbStatus= updateHV(&hv, dbStatus, modo); 
	      if(new_dbStatus != dbStatus ) result=true; 
	      
	      dcs_temp->setValue( eeid, new_dbStatus );
	      
	      if(new_dbStatus != dbStatus) {
		std::cout <<"Dee/chan:"<<dee<<"/"<<chan <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<<" HV: "<< hv.getHV()<<"/"<<hv.getHVNominal()<<std::endl;
		
	      } 
	    }
	  }
	}
      }
      if(chan==1){ // channel 1 has half a dee plus 6 more towers 
	for (int l=0; l<6; l++){
	  int ik=ex_x[l];
	  int ip=ex_y[l];
	  if (EcalScDetId::validDetId(ik,ip,iz)){
	    EcalScDetId eeid(ik,ip,iz);
	    EcalDCSTowerStatus::const_iterator it =dcs_temp->find(eeid.rawId());
	    
	    uint16_t dbStatus = 0;
	    if ( it != dcs_temp->end() ) {
	      dbStatus = it->getStatusCode();
	    }
	    uint16_t new_dbStatus= updateHV(&hv, dbStatus,modo); 
	    if(new_dbStatus != dbStatus ) result=true; 
	    
	    dcs_temp->setValue( eeid, new_dbStatus );
	    
	    if(new_dbStatus != dbStatus) {
	      std::cout <<"Dee/chan:"<<dee<<"/"<<chan <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<<" HV: "<< hv.getHV()<<"/"<<hv.getHVNominal()<<std::endl;
	      
	    } 
	  }
	}
      }
      
      delete [] limits; 

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



int * popcon::EcalDCSHandler::HVEELogicIDToDetID(int dee, int chan) const {
  int iz=-1;
  if(dee==1 || dee==2) iz=1;
  int ix1=1;
  int ix2=1;
  int iy1=1;
  int iy2=1;

  if(dee==1 && chan==1) {
    ix1=1; ix2=10;
    iy1=11; iy2=20;
  } else if(dee==2 && chan==1) {
    ix1=11; ix2=20;
    iy1=1; iy2=10;
  } else if(dee==3 && chan==1) {
    ix1=11; ix2=20;
    iy1=11; iy2=20;
  } else if(dee==4 && chan==1) {
    ix1=1; ix2=10;
    iy1=1; iy2=10;
  } else if(dee==1 && chan==2) {
    ix1=1; ix2=10;
    iy1=1; iy2=10;
  } else if(dee==2 && chan==2) {
    ix1=11; ix2=20;
    iy1=11; iy2=20;
  } else if(dee==3 && chan==2) {
    ix1=11; ix2=20;
    iy1=1; iy2=10;
  } else if(dee==4 && chan==2) {
    ix1=1; ix2=10;
    iy1=11; iy2=20;
  }

  int *result = new int[5];
  
  result[0]=iz;
  result[1]=ix1;
  result[2]=ix2;
  result[3]=iy1;
  result[4]=iy2;
  return result; 
  
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
	} else { // EB minus phi turns as LV numbering 
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


bool popcon::EcalDCSHandler::insertLVDataSetToOffline( const std::map<EcalLogicID, RunDCSLVDat>* dataset, EcalDCSTowerStatus* dcs_temp , const std::vector<EcalLogicID>& my_EELVchan ) const
{
  bool result= false; 
  if (dataset->size() == 0) {
    std::cout << "No data in map!" << std::endl;
  }
  EcalLogicID ecid;
  RunDCSLVDat lv;


  typedef std::map< EcalLogicID, RunDCSLVDat >::const_iterator CI;
  for (CI p = dataset->begin(); p != dataset->end(); ++p) {

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

	  if(new_dbStatus != dbStatus) {
	    std::cout <<"SM/chan:"<<sm<<"/"<<chan <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<<" LV: "<< lv.getLV()<<"/"<<lv.getLVNominal()<<std::endl;
	    
	  } 


	  }
	}
      }
    delete [] limits; 

      
    } else {
	
	// endcaps 
      int dee= ecid.getID1() ;
      int chan= ecid.getID2();
      int n=my_EELVchan.size();
     
	for (int ixt=0; ixt<n; ixt++) {
	  if(my_EELVchan[ixt].getID1()==dee && my_EELVchan[ixt].getID2()==chan){

	    int ilogic=my_EELVchan[ixt].getLogicID();
	    
	    if(ilogic == 2012058060 || ilogic == 2010060058 
	       ||  ilogic == 2012043041 || ilogic == 2010041043) {
	      std::cout<< "crystal " << ilogic << " in the corner ignored" << std::endl; 
	    } else {
	      
	    int iz= (ilogic/1000000)-2010;
	    if(iz==0) iz=-1;
	    if(iz==2) iz=1;
	    if(iz != 1 && iz!= -1) std::cout<< "BAD z"<< std::endl; 
	    
	    int iy=ilogic- int(ilogic/1000)*1000;
	    
	    int ix=(ilogic- int(ilogic/1000000)*1000000 -iy)/1000;
	    
	    int ixtower=  ((ix-1)/5) +1;
	    int iytower=  ((iy-1)/5) +1;
	    
	    if(ixtower<1 || ixtower>20 || iytower <1 || iytower >20) 
	      std::cout<< "BAD x/y"<<ilogic<<"/"<< ixtower<<"/"<<iytower<< std::endl;
	    
	    if (EcalScDetId::validDetId(ixtower,iytower,iz )){
	      EcalScDetId eeid(ixtower,iytower,iz );
	      EcalDCSTowerStatus::const_iterator it =dcs_temp->find(eeid.rawId());
	      uint16_t dbStatus = 0;
	      if ( it != dcs_temp->end() ) {
		dbStatus = it->getStatusCode();
	      }
	      
	      uint16_t new_dbStatus= updateLV(&lv, dbStatus);
	      if(new_dbStatus != dbStatus ) result=true;
	      dcs_temp->setValue( eeid, new_dbStatus );
	      
	      //  std::cout <<"Dee/chan:"<<dee<<"/"<<chan <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<<" LV: "<< lv.getLV()<<"/"<<lv.getLVNominal()<<" ilogic/x/y " <<ilogic<<"/"<< ixtower<<"/"<<iytower<<std::endl;
	      
	      if(new_dbStatus != dbStatus) {
		std::cout <<"Dee/chan:"<<dee<<"/"<<chan <<" new db status ="<< new_dbStatus << " old  "<<dbStatus<<" LV: "<< lv.getLV()<<"/"<<lv.getLVNominal()<<std::endl;
		
	      } 
	      
	    }
	    
	    }
	  
	    
	  } 
	
	}
    	
    }// end of endcaps 
	


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

  // we copy the last valid record to a temporary object 
  EcalDCSTowerStatus* dcs_temp = new EcalDCSTowerStatus();
  if(tagInfo().size) {
    max_since=tagInfo().lastInterval.first;
    Ref dcs_db = lastPayload();
    std::cout << "retrieved last payload "  << std::endl;

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
	      std::cout<<"*** error channel not found: j/i="<<j<<"/"<<i << std::endl;
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
  }  // check if there is already a payload
  else {
    int iz = -1;
    for(int k = 0 ; k < 2; k++ ) {
      if(k == 1) iz = 1;
      // barrel
      for(int i = 1 ; i < 73; i++) {
	for(int j = 1 ; j < 18; j++) {
	  if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,i )){
	    EcalTrigTowerDetId ebid(iz,EcalBarrel,j,i);
	    dcs_temp->setValue( ebid, 0);
	  }
	}
      }
      // endcap
      for(int i=1 ; i<21; i++) {
	for(int j=1 ; j<21; j++) {
	  if (EcalScDetId::validDetId(i,j,iz )){
	    EcalScDetId eeid(i,j,iz);
	    dcs_temp->setValue( eeid, 0);
	  }
	}
      }
    }
  }
  std::cout << "max_since : "  << max_since << std::endl;

  // now read the actual status from the online DB
  econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
  std::cout << "Connection done" << std::endl;
	
	if (!econn)
	  {
	    std::cout << " Problem with OMDS: connection parameters " <<m_sid <<"/"<<m_user<<std::endl;
	    throw cms::Exception("OMDS not available");
	  } 



	std::cout << "Retrieving last run from ONLINE DB ... " << std::endl;
	std::map<EcalLogicID, RunDat> rundat;
	RunIOV rp ;
	run_t runmax=10000000;
	std::string location_p5="P5_Co";
	econn->fetchValidDataSet(&rundat , &rp, location_p5 ,runmax);
	
	unsigned long long  irun=(unsigned long long) rp.getRunNumber();

	// just for testing purposes
	//	irun= max_since+1; 
	
	if(max_since< irun) { 


	  // get the map of the EE LV channels to EE crystals 

	  std::cout << "Retrieving endcap channel list from ONLINE DB ... " << std::endl;
	  
	  std::vector<EcalLogicID> my_EELVchan= econn->getEcalLogicIDSetOrdered( "EE_crystal_number", 1,4,
						 1, 200, EcalLogicID::NULLID, EcalLogicID::NULLID,
						 "EE_LV_channel", 12 ) ;

	  std::cout << "done endcap channel list  ... " << std::endl;

	  // retrieve from last value data record 	
	  // always call this method at first run

	  std::map<EcalLogicID, RunDCSHVDat> dataset;
	  RunIOV *r = NULL;
	  econn->fetchDataSet(&dataset, r);
	  
	  if (!dataset.size()) {
	    throw(std::runtime_error("Zero rows read back"));
	  }
	  
	  
	  if(lot_of_printout) std::cout << "read OK" << std::endl;
	  if(lot_of_printout) printHVDataSet(&dataset,10);
	  
	  std::map<EcalLogicID, RunDCSLVDat> dataset_lv;
	  econn->fetchDataSet(&dataset_lv, r);
	  
	  if (!dataset_lv.size()) {
	    throw(std::runtime_error("Zero rows read back"));
	  }
	  if(lot_of_printout) std::cout << "read OK" << std::endl;
	  if(lot_of_printout) printLVDataSet(&dataset_lv);
	  
	  bool somediff_hv= insertHVDataSetToOffline(&dataset, dcs_temp );
	  bool somediff_lv= insertLVDataSetToOffline(&dataset_lv, dcs_temp, my_EELVchan );
	  
	  if(somediff_hv || somediff_lv) {
	    

	    /*	    Tm t_now_gmt;
		    t_now_gmt.setToCurrentGMTime();
		    uint64_t tsincetemp= t_now_gmt.microsTime()/1000000 ;
		    uint64_t tsince = tsincetemp<< 32; 
		    std::cout << "Generating popcon record for time " << tsincetemp << "..." << std::flush;
	    
	    */

	    std::cout << "Generating popcon record for run " << irun << "..." << std::flush;
    
	    // this is for timestamp
	    //	    m_to_transfer.push_back(std::make_pair((EcalDCSTowerStatus*)dcs_temp,tsince));
	    //	    ss << "Time=" << t_now_gmt.str() << "_DCSchanged_"<<std::endl; 

	    // this is for run number 
	    m_to_transfer.push_back(std::make_pair((EcalDCSTowerStatus*)dcs_temp,irun));
	    ss << "Run=" << irun << "_DCSchanged_"<<std::endl; 

	    m_userTextLog = ss.str()+";";

	  } else {

	    // Tm t_now_gmt;
            // t_now_gmt.setToCurrentGMTime();

	    std::cout<< "Run " << irun << " DCS record was the same as previous run " << std::endl;
	    ss << "Run=" << irun << "_DCSchanged_"<<std::endl; 
	    m_userTextLog = ss.str()+";";
	    
	    delete dcs_temp; 
	    
	  }

	  /*	  
	  
	} else {
	  
	  // here we fetch historical data 

	  uint64_t t_max_val= ((max_since >> 32 ) +2)*1000000 ; // time in microseconds  (2 seconds more than old data)
	
	  Tm t_test(t_max_val);
  
	  std::list< std::pair< Tm, std::map<  EcalLogicID, RunDCSHVDat > > > dataset;
	  econn->fetchDCSDataSet(&dataset, t_test);
	  
	  if (!dataset.size()) {
	    std::cout<< " DCS query retrieved zero lines  "<< std::endl;
	  } else {

	    int num_dcs=0; 
	    std::list< std::pair< Tm, std::map<  EcalLogicID, RunDCSHVDat > > >::iterator it;
	    for (it=dataset.begin(); it!=dataset.end(); ++it){
	      std::pair< Tm, std::map<  EcalLogicID, RunDCSHVDat > > a_pair =(*it);
	      Tm t_pair=a_pair.first;
	      std::map<  EcalLogicID, RunDCSHVDat > a_map = a_pair.second;
	      num_dcs=num_dcs+a_map.size();

	      bool somediff_hv= insertHVDataSetToOffline(&a_map, dcs_temp );


	      if(somediff_hv ) {
		std::cout << "some diff" << std::endl;
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
			  std::cout<<"*** error channel not found: j/i="<<j<<"/"<<i << std::endl;
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
		
		std::cout << "Generating popcon record for time " << tsince << "HRF time " << tnew.str() << "..." << std::flush;
		
		m_to_transfer.push_back(std::make_pair((EcalDCSTowerStatus*)dcs_pop,tsince));
		
		ss << "Time=" << tnew.str() << "_DCSchanged_"<<std::endl; 
		m_userTextLog = ss.str()+";";
		
		


	    }
	    }


	
	  std::cout << " num DCS = "<< num_dcs << std::endl; 
	}



	  delete dcs_temp;
	  */
	  
	}

	delete econn;
	std::cout << "Ecal - > end of getNewObjects -----------\n";

}


