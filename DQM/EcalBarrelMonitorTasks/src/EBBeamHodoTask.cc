/*
 * \file EBBeamHodoTask.cc
 *
 * $Date: 2006/05/17 13:43:18 $
 * $Revision: 1.6 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamHodoTask.h>

EBBeamHodoTask::EBBeamHodoTask(const ParameterSet& ps){

  init_ = false;
  
  // to be filled all the time
  for (int i=0; i<4; i++)
    {
      meHodoOcc_[i] =0;
      meHodoRaw_[i] =0;
    }
  meTDCRec_      =0;

  // filled only when: the table does not move
  meHodoPosRec_    =0;
  meHodoSloXRec_  =0;
  meHodoSloYRec_  =0;
  meHodoQuaXRec_ =0;
  meHodoQuaYRec_ =0;
  meEvsXRec_     =0;
  meEvsYRec_     =0;

  //                       and matrix 5x5 available
  meCaloVsHodoXPos_ =0;
  meCaloVsHodoYPos_ =0;
  meCaloVsTDCTime_    =0;
}

EBBeamHodoTask::~EBBeamHodoTask(){

}

void EBBeamHodoTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBBeamHodoTask::setup(void){

  init_ = true;

  smId =1;

  Char_t histo[20];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamHodoTask");

    for (int i=0; i<4; i++)
      {
	sprintf(histo, "EBBT hodo occ SM%02d, %02d", smId, i+1);
	meHodoOcc_[i] = dbe->book1D(histo, histo, 64, 0., 64.);
	sprintf(histo, "EBBT hodo raw SM%02d, %02d", smId, i+1);
	meHodoRaw_[i] = dbe->book1D(histo, histo, 64, 0., 64.);
      }
    
    sprintf(histo, "EBBT ho Po rec SM%02d", smId);
    meHodoPosRec_ = dbe->book1D(histo, histo, 100, -20, 20);
    
    sprintf(histo, "EBBT ho SloX SM%02d", smId);
    meHodoSloXRec_ = dbe->book1D(histo, histo, 50, -0.005, 0.005);
    
    sprintf(histo, "EBBT ho SloY SM%02d", smId);
    meHodoSloYRec_ = dbe->book1D(histo, histo, 50, -0.005, 0.005);
    
    sprintf(histo, "EBBT ho QualX SM%02d", smId);
    meHodoQuaXRec_ = dbe->book1D(histo, histo, 50, 0, 3);
    
    sprintf(histo, "EBBT ho QualY SM%02d", smId);
    meHodoQuaYRec_ = dbe->book1D(histo, histo, 50, 0, 3);
    
    sprintf(histo, "EBBT TDC rec SM%02d", smId);
    meTDCRec_  = dbe->book1D(histo, histo, 25, 0, 1);
    
    sprintf(histo, "EBBT E1vsX SM%02d", smId);
    meEvsXRec_    = dbe-> bookProfile(histo, histo, 100, -20, 20, 500, 0, 5000);

    sprintf(histo, "EBBT E1vsY SM%02d", smId);
    meEvsYRec_    = dbe-> bookProfile(histo, histo, 100, -20, 20, 500, 0, 5000);

    sprintf(histo, "EBBT PosX: Hodo-Calo SM%02d", smId);
    meCaloVsHodoXPos_   = dbe->book1D(histo, histo, 40, -20, 20);

    sprintf(histo, "EBBT PosY: Hodo-Calo SM%02d", smId);
    meCaloVsHodoYPos_   = dbe->book1D(histo, histo, 40, -20, 20);

    sprintf(histo, "EBBT TimeMax: TDC-Calo SM%02d", smId);
    meCaloVsTDCTime_  = dbe->book1D(histo, histo, 100, -1, 1);//tentative

  }

}

void EBBeamHodoTask::endJob(){

  LogInfo("EBBeamHodoTask") << "analyzed " << ievt_ << " events";

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

  


  // to be added shortly:
  // if table is moving, return




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

  meHodoPosRec_    ->Fill( recHodo->posX(), recHodo->posY() );
  meHodoSloXRec_  ->Fill( recHodo->slopeX());
  meHodoSloYRec_  ->Fill( recHodo->slopeY());
  meHodoQuaXRec_ ->Fill( recHodo->qualX());
  meHodoQuaYRec_ ->Fill( recHodo->qualY());

  
  
  float maxE =0;
  EBDetId maxHitId(0);
  for (  EBUncalibratedRecHitCollection::const_iterator uncalHitItr = pUncalRH->begin();  uncalHitItr!= pUncalRH->end(); uncalHitItr++ ) {
    double e = (*uncalHitItr).amplitude();
    if ( e > maxE )  {
      maxE       = e;
      maxHitId = (*uncalHitItr).id();
    }
  }
  
  meEvsXRec_ -> Fill(recHodo->posX(), maxE);
  meEvsYRec_ -> Fill(recHodo->posY(), maxE);
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
	  caloY +=  (row-2) * sideY * weight;
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
