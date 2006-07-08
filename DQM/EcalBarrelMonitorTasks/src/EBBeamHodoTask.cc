/*
 * \file EBBeamHodoTask.cc
 *
 * $Date: 2006/07/08 07:22:02 $
 * $Revision: 1.18 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamHodoTask.h>

EBBeamHodoTask::EBBeamHodoTask(const ParameterSet& ps){

  init_ = false;
  tableIsMoving_ = false;
  
  // to be filled all the time
  for (int i=0; i<4; i++) {
    meHodoOcc_[i] =0;
    meHodoRaw_[i] =0;
  }
  meTDCRec_      =0;

  // filled only when: the table does not move
  meHodoPosRecX_    =0;
  meHodoPosRecY_    =0;
  meHodoPosRecXY_ =0;
  meHodoSloXRec_  =0;
  meHodoSloYRec_  =0;
  meHodoQuaXRec_ =0;
  meHodoQuaYRec_ =0;
  meHodoPosXMinusCaloPosXVsCry_   =0;
  meHodoPosYMinusCaloPosYVsCry_   =0;
  meTDCTimeMinusCaloTimeVsCry_ =0;

  meEvsXRecProf_     =0;
  meEvsYRecProf_     =0;
  meEvsXRecHis_     =0;
  meEvsYRecHis_     =0;

  //                       and matrix 5x5 available
  meCaloVsHodoXPos_ =0;
  meCaloVsHodoYPos_ =0;
  meCaloVsTDCTime_    =0;
}

EBBeamHodoTask::~EBBeamHodoTask(){

}

void EBBeamHodoTask::beginJob(const EventSetup& c){

  ievt_  = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamHodoTask");
    dbe->rmdir("EcalBarrel/EBBeamHodoTask");
  }

  LV1_ = 0;
  cryInBeamCounter_ =1;
  resetNow_                =false;

}

void EBBeamHodoTask::setup(void){

  init_ = true;

  smId =1;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamHodoTask");

    // following ME (type I):
    //  *** do not need to be ever reset
    //  *** can be filled regardless of the moving/notMoving status of the table

    for (int i=0; i<4; i++) {
      sprintf(histo, "EBBHT occup SM%02d %02d", smId, i+1);
      meHodoOcc_[i] = dbe->book1D(histo, histo, 30, 0., 30.);
      sprintf(histo, "EBBHT raw SM%02d %02d", smId, i+1);
      meHodoRaw_[i] = dbe->book1D(histo, histo, 64, 0., 64.);
    }
    
    sprintf(histo, "EBBHT PosX rec SM%02d", smId);
    meHodoPosRecX_ = dbe->book1D(histo, histo, 100, -20, 20);
    
    sprintf(histo, "EBBHT PosY rec SM%02d", smId);
    meHodoPosRecY_ = dbe->book1D(histo, histo, 100, -20, 20);
    
    sprintf(histo, "EBBHT PosYX rec SM%02d", smId);
    meHodoPosRecXY_ = dbe->book2D(histo, histo, 100, -20, 20,100, -20, 20);
    
    sprintf(histo, "EBBHT SloX SM%02d", smId);
    meHodoSloXRec_ = dbe->book1D(histo, histo, 50, -0.005, 0.005);
    
    sprintf(histo, "EBBHT SloY SM%02d", smId);
    meHodoSloYRec_ = dbe->book1D(histo, histo, 50, -0.005, 0.005);
    
    sprintf(histo, "EBBHT QualX SM%02d", smId);
    meHodoQuaXRec_ = dbe->book1D(histo, histo, 50, 0, 5);
    
    sprintf(histo, "EBBHT QualY SM%02d", smId);
    meHodoQuaYRec_ = dbe->book1D(histo, histo, 50, 0, 5);
    
    sprintf(histo, "EBBHT TDC rec SM%02d", smId);
    meTDCRec_  = dbe->book1D(histo, histo, 25, 0, 1);
    
    sprintf(histo, "EBBHT Hodo-Calo X vs Cry SM%02d", smId);
    meHodoPosXMinusCaloPosXVsCry_  = dbe->book1D(histo, histo, 50, 0, 50);
    
    sprintf(histo, "EBBHT Hodo-Calo Y vs Cry SM%02d", smId);
    meHodoPosYMinusCaloPosYVsCry_  = dbe->book1D(histo, histo, 50, 0, 50);
    
    sprintf(histo, "EBBHT TDC-Calo vs Cry SM%02d", smId);
    meTDCTimeMinusCaloTimeVsCry_  = dbe->book1D(histo, histo, 50, 0, 50);

    // following ME (type II):
    //  *** can be filled only when table is **not**Moving
    //  *** need to be reset once table goes from 'moving'->notMoving

    sprintf(histo, "EBBHT prof E1 vs X SM%02d", smId);
    meEvsXRecProf_    = dbe-> bookProfile(histo, histo, 100, -20, 20, 500, 0, 5000, "s");

    sprintf(histo, "EBBHT prof E1 vs Y SM%02d", smId);
    meEvsYRecProf_    = dbe-> bookProfile(histo, histo, 100, -20, 20, 500, 0, 5000, "s");
    
    sprintf(histo, "EBBHT his E1 vs X SM%02d", smId);
    meEvsXRecHis_    = dbe-> book2D(histo, histo, 100, -20, 20, 500, 0, 5000);

    sprintf(histo, "EBBHT his E1 vs Y SM%02d", smId);
    meEvsYRecHis_    = dbe-> book2D(histo, histo, 100, -20, 20, 500, 0, 5000);

    sprintf(histo, "EBBHT PosX Hodo-Calo SM%02d", smId);
    meCaloVsHodoXPos_   = dbe->book1D(histo, histo, 40, -20, 20);

    sprintf(histo, "EBBHT PosY Hodo-Calo SM%02d", smId);
    meCaloVsHodoYPos_   = dbe->book1D(histo, histo, 40, -20, 20);

    sprintf(histo, "EBBHT TimeMax TDC-Calo SM%02d", smId);
    meCaloVsTDCTime_  = dbe->book1D(histo, histo, 100, -1, 1);//tentative

  }

}

void EBBeamHodoTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamHodoTask");

    for (int i=0; i<4; i++) {
      if ( meHodoOcc_[i] ) dbe->removeElement( meHodoOcc_[i]->getName() );
      meHodoOcc_[i] = 0;
      if ( meHodoRaw_[i] ) dbe->removeElement( meHodoRaw_[i]->getName() );
      meHodoRaw_[i] = 0;    
    }

    if ( meHodoPosRecX_ ) dbe->removeElement( meHodoPosRecX_->getName() );
    meHodoPosRecX_ = 0;
    if ( meHodoPosRecY_ ) dbe->removeElement( meHodoPosRecY_->getName() );
    meHodoPosRecY_ = 0;
    if ( meHodoPosRecXY_ ) dbe->removeElement( meHodoPosRecXY_->getName() );
    meHodoPosRecXY_ = 0;
    if ( meHodoSloXRec_ ) dbe->removeElement( meHodoSloXRec_->getName() );
    meHodoSloXRec_ = 0;
    if ( meHodoSloYRec_ ) dbe->removeElement( meHodoSloYRec_->getName() );
    meHodoSloYRec_ = 0;
    if ( meHodoQuaXRec_ ) dbe->removeElement( meHodoQuaXRec_->getName() );
    meHodoQuaXRec_ = 0;
    if ( meHodoQuaYRec_ ) dbe->removeElement( meHodoQuaYRec_->getName() );
    meHodoQuaYRec_ = 0;
    if ( meTDCRec_ ) dbe->removeElement( meTDCRec_->getName() );
    meTDCRec_ = 0;
    if ( meEvsXRecProf_ ) dbe->removeElement( meEvsXRecProf_->getName() );
    meEvsXRecProf_ = 0;
    if ( meEvsYRecProf_ ) dbe->removeElement( meEvsYRecProf_->getName() );
    meEvsYRecProf_ = 0;
    if ( meEvsXRecHis_ ) dbe->removeElement( meEvsXRecHis_->getName() );
    meEvsXRecHis_ = 0;
    if ( meEvsYRecHis_ ) dbe->removeElement( meEvsYRecHis_->getName() );
    meEvsYRecHis_ = 0;
    if ( meCaloVsHodoXPos_ ) dbe->removeElement( meCaloVsHodoXPos_->getName() );
    meCaloVsHodoXPos_ = 0;
    if ( meCaloVsHodoYPos_ ) dbe->removeElement( meCaloVsHodoYPos_->getName() );
    meCaloVsHodoYPos_ = 0;
    if ( meCaloVsTDCTime_ ) dbe->removeElement( meCaloVsTDCTime_->getName() );
    meCaloVsTDCTime_ = 0;
    if ( meHodoPosXMinusCaloPosXVsCry_  ) dbe->removeElement( meHodoPosXMinusCaloPosXVsCry_ ->getName() );
    meHodoPosXMinusCaloPosXVsCry_  = 0;
    if ( meHodoPosYMinusCaloPosYVsCry_  ) dbe->removeElement( meHodoPosYMinusCaloPosYVsCry_ ->getName() );
    meHodoPosYMinusCaloPosYVsCry_  = 0;
    if ( meTDCTimeMinusCaloTimeVsCry_  ) dbe->removeElement( meTDCTimeMinusCaloTimeVsCry_  ->getName() );
    meTDCTimeMinusCaloTimeVsCry_  = 0;

  }

  init_ = false;

}

void EBBeamHodoTask::endJob(void){

  LogInfo("EBBeamHodoTask") << "analyzed " << ievt_ << " events";

  this->cleanup();

}

void EBBeamHodoTask::analyze(const Event& e, const EventSetup& c){
  
  bool enable = false;
  
  Handle<EcalRawDataCollection> dcchs;
  Handle<EcalTBEventHeader> pEvH;
  try{
    e.getByLabel("ecalEBunpacker", dcchs);
    
    int nebc = dcchs->size();
    LogDebug("EBBeamHodoTask") << "event: " << ievt_ << " DCC headers collection size: " << nebc;
    
    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {
      
      EcalDCCHeaderBlock dcch = (*dcchItr);
      
      if ( dcch.getRunType() == EcalDCCHeaderBlock::BEAMH4
	   || dcch.getRunType() == EcalDCCHeaderBlock::BEAMH2  ) enable = true;
    }
    
  }
  catch ( std::exception& ex) {
    LogDebug("EcalBeamTask") << " EcalRawDataCollection not in event. Trying EcalTBEventHeader (2004 data)." << std::endl;
    
    try {
      e.getByType(pEvH);
      enable = true;
      LogDebug("EcalBeamTask") << " EcalTBEventHeader found, instead." << std::endl;
    } 
    catch ( std::exception& ex ) {
      LogError("EBBeamHodoTask") << "EcalTBEventHeader not present in event TOO! Returning." << std::endl;
    }
    
  }
  
  
  if ( ! enable ) return;
  if ( ! init_ ) this->setup();
  ievt_++;

  //  LV1_ = (* dcchs->begin()) .getLV1();
  // temporarily, identify LV1 and #monitoredEvents
  LV1_ = ievt_;
  
  Handle<EcalUncalibratedRecHitCollection> pUncalRH;
  const EcalUncalibratedRecHitCollection* uncalRecH =0;
  try {
    e.getByLabel("ecalUncalibHitMaker", "EcalUncalibRecHitsEB", pUncalRH);
    //    e.getByType(pUncalRH);
    uncalRecH = pUncalRH.product(); // get a ptr to the product
    int neh = pUncalRH->size();
    LogDebug("EBBeamHodoTask") << "EcalUncalibRecHitsEB found in event " << ievt_ << "; hits collection size " << neh;
  } 
  catch ( std::exception& ex ) {
    LogError("EBBeamHodoTask") << "EcalUncalibRecHitsEB not found in event! Returning." << std::endl;
  }


  Handle<EcalTBHodoscopeRawInfo> pHodoRaw;
  const EcalTBHodoscopeRawInfo* rawHodo=0;
  try {
    e.getByType( pHodoRaw );
    rawHodo = pHodoRaw.product();
    LogDebug("EcalBeamTask") << "hodoscopeRaw:  num planes: " <<  rawHodo->planes() 
			     << " channels in plane 1: "  <<  rawHodo->channels(0) << endl;
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBHodoscopeRawInfo" << std::endl;
  }


   Handle<EcalTBTDCRawInfo> pTDCRaw;
   const EcalTBTDCRawInfo* rawTDC=0;
   try {
     e.getByType( pTDCRaw );
     rawTDC = pTDCRaw.product();
   } catch ( std::exception& ex ) {
     LogError("EcalBeamTask") << "Error! Can't get the product EcalTBTDCRawInfo" << std::endl;
   }


  Handle<EcalTBTDCRecInfo> pTDC;
  const EcalTBTDCRecInfo* recTDC=0;
  try {
    e.getByLabel( "tdcReco", "EcalTBTDCRecInfo", pTDC);
    recTDC = pTDC.product();
    LogDebug("EBBeamHodoTask") << " TDC offset is: " << recTDC->offset() << endl;
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBTDCRecInfo" << std::endl;
  }
  
  if ( !rawTDC ||!rawHodo || !uncalRecH)
    {
      LogError("EcalBeamTask") << "analyze: missing a needed collection, returning.\n\n\n" << std::endl;
      return;
    }
  LogDebug("EBBeamHodoTask") << " TDC raw, Hodo raw, uncalRecH and DCCheader found." << std::endl;




  // in temporary absence of table status in DCCheader, here mimic:
  //  **   changes from 'table-is-still' to 'table-is-moving', and viceversa
  //  **   new value for cry-in-beam
  if (ievt_%300 ==0)
    {
      // change status every 3000 events
      tableIsMoving_ = (! tableIsMoving_); 

      // if table has come to a stop:
      //   - increase counter of crystals that have been on beam
      //   - set flag for resetting
      if (! tableIsMoving_) {
	cryInBeamCounter_++;
	resetNow_ = true;
	LogDebug("EcalBeamTask")  << "At event LV1: " << LV1_ << " switching table status by hand: from still to moving. " << endl;
      }
      else
	{// if table has started moving
	  LogDebug("EcalBeamTask")  << "At event LV1: " << LV1_ <<  " switching table status by hand: from moving to still. " << endl;
	  // fill here plots which keep history of beamed crystals

	  float HodoPosXMinusCaloPosXVsCry_mean=0;
	  float HodoPosXMinusCaloPosXVsCry_rms   =0;
	  float HodoPosYMinusCaloPosYVsCry_mean=0;
	  float HodoPosYMinusCaloPosYVsCry_rms   =0;
	  float TDCTimeMinusCaloTimeVsCry_mean    =0;
	  float TDCTimeMinusCaloTimeVsCry_rms       =0;
	  
	  if (meCaloVsHodoXPos_ -> getEntries()  > 30){
	    HodoPosXMinusCaloPosXVsCry_mean = meCaloVsHodoXPos_ -> getMean(1);
	    HodoPosXMinusCaloPosXVsCry_rms    = meCaloVsHodoXPos_ -> getRMS(1);
	  }
	  if (meCaloVsHodoYPos_ -> getEntries()  > 30){
	    HodoPosYMinusCaloPosYVsCry_mean = meCaloVsHodoYPos_ -> getMean(1);
	    HodoPosYMinusCaloPosYVsCry_rms    = meCaloVsHodoYPos_ -> getRMS(1);
	  }
	  if (meCaloVsTDCTime_ -> getEntries()  > 30){
	    TDCTimeMinusCaloTimeVsCry_mean     = meCaloVsTDCTime_    -> getMean(1);
	    TDCTimeMinusCaloTimeVsCry_rms        = meCaloVsTDCTime_    -> getRMS(1);
	  }
	  meHodoPosXMinusCaloPosXVsCry_ ->  setBinContent( cryInBeamCounter_, HodoPosYMinusCaloPosYVsCry_mean);
	  meHodoPosXMinusCaloPosXVsCry_ ->  setBinError(      cryInBeamCounter_, HodoPosYMinusCaloPosYVsCry_rms);
	  meHodoPosYMinusCaloPosYVsCry_ ->  setBinContent( cryInBeamCounter_, HodoPosXMinusCaloPosXVsCry_mean);
	  meHodoPosYMinusCaloPosYVsCry_ ->  setBinError(      cryInBeamCounter_, HodoPosXMinusCaloPosXVsCry_rms);
	  meTDCTimeMinusCaloTimeVsCry_     ->  setBinContent(cryInBeamCounter_, TDCTimeMinusCaloTimeVsCry_mean);
	  meTDCTimeMinusCaloTimeVsCry_     ->  setBinError(cryInBeamCounter_, TDCTimeMinusCaloTimeVsCry_rms);

	  LogDebug("EcalBeamTask")  << "At event LV1: " << LV1_ <<  " trace histos filled ( cryInBeamCounter_=" 
				    << cryInBeamCounter_ << ")"<< endl;

	}
    }



  // if table has come to rest from movement, reset concerned ME's
  if (resetNow_)
    {
      EBMUtilsTasks::resetHisto(    meEvsXRecProf_ );
      EBMUtilsTasks::resetHisto(   meEvsYRecProf_);
      EBMUtilsTasks::resetHisto(    meEvsXRecHis_ );
      EBMUtilsTasks::resetHisto(    meEvsYRecHis_ );
      EBMUtilsTasks::resetHisto(    meCaloVsHodoXPos_  );
      EBMUtilsTasks::resetHisto(    meCaloVsHodoYPos_  );
      EBMUtilsTasks::resetHisto(    meCaloVsTDCTime_  );
      
      resetNow_ = false;
    }

  
  // handling histos (type I):

  for (unsigned int planeId=0; planeId <4; planeId++){
    
    const EcalTBHodoscopePlaneRawHits& planeRaw = rawHodo->getPlaneRawHits(planeId);
    LogDebug("EcalBeamTask")  << "\t plane: " << (planeId+1)
			      << "\t number of fibers: " << planeRaw.channels()
			      << "\t number of hits: " << planeRaw.numberOfFiredHits();
    meHodoOcc_[planeId] -> Fill( planeRaw.numberOfFiredHits() );
    
    for (unsigned int i=0;i<planeRaw.channels();i++)
      {
	if (planeRaw.isChannelFired(i))
	   {
	     LogInfo("EcalBeamTask")<< " channel " << (i+1) << " has fired";
	     meHodoRaw_[planeId] -> Fill(i+0.5);
	   }
      }
  }
  
  meTDCRec_        ->Fill( recTDC->offset());



  Handle<EcalTBHodoscopeRecInfo> pHodo;
  const EcalTBHodoscopeRecInfo* recHodo=0;
  try {
    e.getByLabel( "hodoscopeReco", "EcalTBHodoscopeRecInfo", pHodo);
    recHodo = pHodo.product();
    LogDebug("EcalBeamTask") << "hodoscopeReco:    x: " << recHodo->posX()
			     << "\ty: " << recHodo->posY()
			     << "\t sx: " << recHodo->slopeX() << "\t qualx: " << recHodo->qualX() 
			     << "\t sy: " << recHodo->slopeY() << "\t qualy: " << recHodo->qualY() 
			     << endl;
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBHodoscopeRecInfo" << std::endl;
  }
  if (!recHodo)
    {
       LogError("EcalBeamTask") << "analyze: missing a needed collection, returning.\n\n\n" << std::endl;
       return;
     }
   LogDebug("EBBeamHodoTask") << " Hodo reco found." << std::endl;

  meHodoPosRecXY_    ->Fill( recHodo->posX(), recHodo->posY() );
  meHodoPosRecX_       ->Fill( recHodo->posX());
  meHodoPosRecY_       ->Fill( recHodo->posY() );
  meHodoSloXRec_        ->Fill( recHodo->slopeX());
  meHodoSloYRec_        ->Fill( recHodo->slopeY());
  meHodoQuaXRec_       ->Fill( recHodo->qualX());
  meHodoQuaYRec_       ->Fill( recHodo->qualY());



  // handling histos (type II)
  
  if (tableIsMoving_)
    {
      LogDebug("EcalBeamTask")<< "At event LV1:" << LV1_ << " table is moving. Not filling concerned monigoring elements. ";
      return;
    }

  float maxE =0;
  EBDetId maxHitId(0);
  for (  EBUncalibratedRecHitCollection::const_iterator uncalHitItr = pUncalRH->begin();  uncalHitItr!= pUncalRH->end(); uncalHitItr++ ) {
    double e = (*uncalHitItr).amplitude();
    if ( e > maxE )  {
      maxE       = e;
      maxHitId = (*uncalHitItr).id();
    }
  }
  if (  maxHitId == EBDetId(0) )
    { 
      LogError("EBBeamHodoTask") << "No positive UncalRecHit found in ECAL in event " << ievt_ << " - returning." << std::endl;
      return;
    }
  
  meEvsXRecProf_ -> Fill(recHodo->posX(), maxE);
  meEvsYRecProf_ -> Fill(recHodo->posY(), maxE);
  meEvsXRecHis_   -> Fill(recHodo->posX(), maxE);
  meEvsYRecHis_   -> Fill(recHodo->posY(), maxE);

  LogInfo("EcalBeamTask")<< " channel with max is " << maxHitId;
  
  bool mat5x5 =true;
  int ietaMax = maxHitId.ieta();
  int iphiMax = (maxHitId.iphi() % 20); 
  if (ietaMax ==1 || ietaMax ==2 || ietaMax ==84 || ietaMax == 85 ||
      iphiMax ==1 || iphiMax ==2 || iphiMax ==19 || iphiMax == 20 )
    {mat5x5 =false;}
  
  if (!mat5x5) return;
  
  
  EBDetId Xtals5x5[25];
  double   ene5x5[25];
  double   e25    =0;
  for (unsigned int icry=0;icry<25;icry++)
    {
      unsigned int row = icry / 5;
      unsigned int column= icry %5;
      try{
	Xtals5x5[icry]=EBDetId(maxHitId.ieta()+column-2,maxHitId.iphi()+row-2,EBDetId::ETAPHIMODE);
	double e = (*  pUncalRH->find( Xtals5x5[icry] )  ).amplitude(); 
	ene5x5[icry] =e;
	e25 +=e;
       }catch ( std::runtime_error &e )
	{
	  LogDebug("EcalBeamTask")<< "Cannot construct 5x5 matrix around EBDetId " << maxHitId;
	  mat5x5 =false;
	}
    }
  
  if (!mat5x5) return;
  LogDebug("EcalBeamTask")<< "Could construct 5x5 matrix around EBDetId " << maxHitId;

  // im mm
  float sideX =24.06;
  float sideY =22.02;
  float caloX =0;
  float caloY =0;
  float weight=0;
  float sumWeight=0;
  // X and Y calculated from log-weighted energy center of mass
  for (unsigned int icry=0;icry<25;icry++)
    {
      weight = log( ene5x5[icry] / e25) + 3.8;
      if (weight>0)
	{
	  unsigned int row      = icry / 5;
	  unsigned int column = icry %5;
	  caloX +=  (column-2) * sideX * weight;
	  caloY -=  (row-2) * sideY * weight;
	  sumWeight += weight;
	}
    }
  caloX /=sumWeight;
  caloY /=sumWeight;

  meCaloVsHodoXPos_  ->Fill( recHodo->posX()-caloX );
  meCaloVsHodoYPos_  ->Fill( recHodo->posY()-caloY );
  meCaloVsTDCTime_     ->Fill( (*  pUncalRH->find( maxHitId )  ).jitter()  -  recTDC->offset() - 3);
  LogDebug("EcalBeamTask")<< "jiitter from uncalRecHit: " <<  (*  pUncalRH->find( maxHitId )  ).jitter() << std::endl;

}
