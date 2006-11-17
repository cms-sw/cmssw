/*
 * \file EBBeamHodoTask.cc
 *
 * $Date: 2006/08/03 06:35:09 $
 * $Revision: 1.27 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
 */

#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamHodoTask.h>

EBBeamHodoTask::EBBeamHodoTask(const ParameterSet& ps){

  init_ = false;
  tableIsMoving_ = false;
  cryInBeam_ =0;
  previousCryInBeam_ = -99999;  

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
  meMissingCollections_        =0;

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
  cryInBeamCounter_ =0;
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
    
    sprintf(histo, "EBBHT Missing Collections SM%02d", smId);
    meMissingCollections_ = dbe->book1D(histo, histo, 7, 0, 7);

    // following ME (type II):
    //  *** can be filled only when table is **not** moving
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
    if ( meMissingCollections_  ) dbe->removeElement( meMissingCollections_ ->getName() );
    meMissingCollections_  = 0;

  }

  init_ = false;

}

void EBBeamHodoTask::endJob(void){

  LogInfo("EBBeamHodoTask") << "analyzed " << ievt_ << " events";

  this->cleanup();

}

void EBBeamHodoTask::analyze(const Event& e, const EventSetup& c){
  
  bool enable = false;


  Handle<EcalTBEventHeader> pHeader;
  const EcalTBEventHeader * Header =0;
  try{
    e.getByLabel("ecalEBunpacker", pHeader);
    Header = pHeader.product(); // get a ptr to the product
    if (!Header) {
      LogDebug("EBBeamHodoTask") << "Event header not found. Returning. ";
      meMissingCollections_ -> Fill(0); // bin1: missing CMSSW Event header
      return;
    }
    tableIsMoving_     = Header->tableIsMoving();
    cryInBeam_           = Header->crystalInBeam();  //  cryInBeam_         = Header->nominalCrystalInBeam();
    if (previousCryInBeam_ == -99999 )
      {      previousCryInBeam_ = cryInBeam_ ;    }
  
    LogDebug("EBBeamHodoTask") << "event: " << ievt_ << " event header found ";    
    if (tableIsMoving_){
      LogDebug("EBBeamHodoTask") << "Table is moving. ";  }
    else
      {      LogDebug("EBBeamHodoTask") << "Table is not moving. ";    }
  }// end try
  catch ( std::exception& ex) {
    LogWarning("EBBeamHodoTask") << "Event header not found (exception caught). Returning. ";    
    return;    
  }




  Handle<EcalRawDataCollection> dcchs;
  //  Handle<EcalTBEventHeader> pEvH;
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
    LogDebug("EcalBeamTask") << " EcalRawDataCollection not in event. Returning.";
    meMissingCollections_ -> Fill(1); // bin2: missing DCC headers
    return;
    // see bottom of cc file for compatibility to 2004 data [***]
  }
  
  
  if ( ! enable ) return;
  if ( ! init_ ) this->setup();
  ievt_++;

  LV1_ = Header->eventNumber();


  
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
    LogError("EBBeamHodoTask") << "EcalUncalibRecHitsEB not found in event! Returning.";
    meMissingCollections_ -> Fill(2); // bin3: missing uncalibRecHits
    return;
  }


  Handle<EcalTBHodoscopeRawInfo> pHodoRaw;
  const EcalTBHodoscopeRawInfo* rawHodo=0;
  try {
    e.getByType( pHodoRaw );
    rawHodo = pHodoRaw.product();
    if(rawHodo->planes() ){
    LogDebug("EcalBeamTask") << "hodoscopeRaw:  num planes: " <<  rawHodo->planes() 
			     << " channels in plane 1: "  <<  rawHodo->channels(0);
    }
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBHodoscopeRawInfo. Returning";
    meMissingCollections_ -> Fill(3); // bin4: missing raw hodo hits collection
    return;
  }


  Handle<EcalTBTDCRawInfo> pTDCRaw;
  const EcalTBTDCRawInfo* rawTDC=0;
  try {
    e.getByType( pTDCRaw );
    rawTDC = pTDCRaw.product();
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBTDCRawInfo. Returning.";
    meMissingCollections_ -> Fill(4); // bin5: missing raw TDC
    return;
  }

  
  if ( !rawTDC ||!rawHodo || !uncalRecH  || !( rawHodo->planes() ))
    {
      LogWarning("EcalBeamTask") << "analyze: missing a needed collection or hodo collection empty. Returning.";
      return;
    }
  LogDebug("EBBeamHodoTask") << " TDC raw, Hodo raw, uncalRecH and DCCheader found.";



  // table has come to a stop is identified by new value of cry_in_beam
  //   - increase counter of crystals that have been on beam
  //   - set flag for resetting
  if (cryInBeam_ != previousCryInBeam_ )
    {
      previousCryInBeam_ = cryInBeam_ ;
      cryInBeamCounter_++;
      resetNow_ = true;	

      // since flag "tableIsMoving==false" is reliable (as we can tell, so far),
      // operations due when "table has started moving"
      // can be done after the change in crystal in beam
      
      LogDebug("EcalBeamTask")  << "At event number : " << LV1_ <<  " switching table status: from moving to still. "
				<< " cry in beam is: " << cryInBeam_ << ", step being: " << cryInBeamCounter_ ;
	
      // fill here plots which keep history of beamed crystals
      float HodoPosXMinusCaloPosXVsCry_mean  =0;
      float HodoPosXMinusCaloPosXVsCry_rms   =0;
      float HodoPosYMinusCaloPosYVsCry_mean  =0;
      float HodoPosYMinusCaloPosYVsCry_rms   =0;
      float TDCTimeMinusCaloTimeVsCry_mean   =0;
      float TDCTimeMinusCaloTimeVsCry_rms    =0;
      
      // min number of entries chosen assuming:
      //                 prescaling = 100 X 2FU
      //                 that we want at leas 2k events per crystal

      if (meCaloVsHodoXPos_ -> getEntries()  > 10){
	HodoPosXMinusCaloPosXVsCry_mean = meCaloVsHodoXPos_ -> getMean(1);
	HodoPosXMinusCaloPosXVsCry_rms  = meCaloVsHodoXPos_ -> getRMS(1);
	meHodoPosXMinusCaloPosXVsCry_ ->  setBinContent( cryInBeamCounter_, HodoPosXMinusCaloPosXVsCry_mean);
	meHodoPosXMinusCaloPosXVsCry_ ->  setBinError(      cryInBeamCounter_, HodoPosXMinusCaloPosXVsCry_rms);
	LogDebug("EcalBeamTask")  << "At event number: " << LV1_ << " step: " << cryInBeamCounter_
				  <<  " DeltaPosX is: " << (meCaloVsHodoXPos_ -> getMean(1))
				  << " +-" << ( meCaloVsHodoXPos_ -> getRMS(1));
      }
      if (meCaloVsHodoYPos_ -> getEntries()  > 10){
	HodoPosYMinusCaloPosYVsCry_mean = meCaloVsHodoYPos_ -> getMean(1);
	HodoPosYMinusCaloPosYVsCry_rms  = meCaloVsHodoYPos_ -> getRMS(1);
	meHodoPosYMinusCaloPosYVsCry_ ->  setBinContent( cryInBeamCounter_, HodoPosYMinusCaloPosYVsCry_mean);
	meHodoPosYMinusCaloPosYVsCry_ ->  setBinError(      cryInBeamCounter_, HodoPosYMinusCaloPosYVsCry_rms);
	LogDebug("EcalBeamTask")  << "At event number: " << LV1_ << " step: " << cryInBeamCounter_
				  <<  " DeltaPosY is: " << (meCaloVsHodoYPos_ -> getMean(1))
				  << " +-" << ( meCaloVsHodoYPos_ -> getRMS(1));
      }
      if (meCaloVsTDCTime_ -> getEntries()  > 10){
	TDCTimeMinusCaloTimeVsCry_mean     = meCaloVsTDCTime_    -> getMean(1);
	TDCTimeMinusCaloTimeVsCry_rms      = meCaloVsTDCTime_    -> getRMS(1);
	meTDCTimeMinusCaloTimeVsCry_     ->  setBinContent(cryInBeamCounter_, TDCTimeMinusCaloTimeVsCry_mean);
	meTDCTimeMinusCaloTimeVsCry_     ->  setBinError(cryInBeamCounter_, TDCTimeMinusCaloTimeVsCry_rms);
	LogDebug("EcalBeamTask")  << "At event number: " << LV1_ << " step: " << cryInBeamCounter_
				  <<  " DeltaT is: " << (meCaloVsTDCTime_ -> getMean(1))
				  << " +-" << ( meCaloVsTDCTime_ -> getRMS(1));
      }

      LogDebug("EcalBeamTask")  << "At event number: " << LV1_ <<  " trace histos filled ( cryInBeamCounter_=" 
				<< cryInBeamCounter_ << ")";

    }




  // if table has come to rest (from movement), reset concerned ME's
  if (resetNow_)
    {
      EBMUtilsTasks::resetHisto(    meEvsXRecProf_ );
      EBMUtilsTasks::resetHisto(    meEvsYRecProf_);
      EBMUtilsTasks::resetHisto(    meEvsXRecHis_ );
      EBMUtilsTasks::resetHisto(    meEvsYRecHis_ );
      EBMUtilsTasks::resetHisto(    meCaloVsHodoXPos_  );
      EBMUtilsTasks::resetHisto(    meCaloVsHodoYPos_  );
      EBMUtilsTasks::resetHisto(    meCaloVsTDCTime_  );
      
      resetNow_ = false;
    }




  /**************************************  
  // handling histos type I:
  **************************************/
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



  Handle<EcalTBTDCRecInfo> pTDC;
  const EcalTBTDCRecInfo* recTDC=0;
  try {
    e.getByLabel( "tdcReco", "EcalTBTDCRecInfo", pTDC);
    recTDC = pTDC.product();
    LogDebug("EBBeamHodoTask") << " TDC offset is: " << recTDC->offset();
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBTDCRecInfo. Returning";
    meMissingCollections_ -> Fill(5); // bin6: missing reconstructed TDC 
    return;
  }

  Handle<EcalTBHodoscopeRecInfo> pHodo;
  const EcalTBHodoscopeRecInfo* recHodo=0;
  try {
    e.getByLabel( "hodoscopeReco", "EcalTBHodoscopeRecInfo", pHodo);
    recHodo = pHodo.product();
    LogDebug("EcalBeamTask") << "hodoscopeReco:    x: " << recHodo->posX()
			     << "\ty: " << recHodo->posY()
			     << "\t sx: " << recHodo->slopeX() << "\t qualx: " << recHodo->qualX() 
			     << "\t sy: " << recHodo->slopeY() << "\t qualy: " << recHodo->qualY();
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! Can't get the product EcalTBHodoscopeRecInfo";
    meMissingCollections_ -> Fill(6); // bin7: missing reconstructed hodoscopes
    return;
  }


  if ( (!recHodo) || (!recTDC) ) 
    {
      LogWarning("EcalBeamTask") << "analyze: missing a needed collection, recHodo or recTDC. Returning.";
      return;
    }
  LogDebug("EBBeamHodoTask") << " Hodo reco and TDC reco found.";
   
  meTDCRec_        ->Fill( recTDC->offset());
   

  meHodoPosRecXY_    ->Fill( recHodo->posX(), recHodo->posY() );
  meHodoPosRecX_       ->Fill( recHodo->posX());
  meHodoPosRecY_       ->Fill( recHodo->posY() );
  meHodoSloXRec_        ->Fill( recHodo->slopeX());
  meHodoSloYRec_        ->Fill( recHodo->slopeY());
  meHodoQuaXRec_       ->Fill( recHodo->qualX());
  meHodoQuaYRec_       ->Fill( recHodo->qualY());



  /**************************************  
  // handling histos type II:
  **************************************/
  
  if (tableIsMoving_)
    {
      LogDebug("EcalBeamTask")<< "At event number:" << LV1_ << " table is moving. Not filling concerned monitoring elements. ";
      return;
    }
  else
    {
      LogDebug("EcalBeamTask")<< "At event number:" << LV1_ << " table is not moving - thus filling alos monitoring elements requiring so.";
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
      LogError("EBBeamHodoTask") << "No positive UncalRecHit found in ECAL in event " << ievt_ << " - returning.";
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
  LogDebug("EcalBeamTask")<< "jiitter from uncalRecHit: " <<  (*  pUncalRH->find( maxHitId )  ).jitter();

}





//[***]
// used to debug importing 2004 data
//    LogDebug("EcalBeamTask") << " EcalRawDataCollection not in event. Trying EcalTBEventHeader (2004 data).";
//    try {
//      e.getByType(pEvH);
//      enable = true;
//      LogDebug("EcalBeamTask") << " EcalTBEventHeader found, instead.";
//    } 
//    catch ( std::exception& ex ) {
//      LogError("EBBeamHodoTask") << "EcalTBEventHeader not present in event TOO! Returning.";
//    }
