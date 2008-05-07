/*
 *  simple validation package for CSC DIGIs, RECHITs and SEGMENTs.
 *
 *  Michael Schmitt
 *  Andy Kubik
 *  Northwestern University
 */
#include "RecoLocalMuon/CSCValidation/src/CSCValidation.h"

using namespace std;
using namespace edm;


///////////////////
//  CONSTRUCTOR  //
///////////////////
CSCValidation::CSCValidation(const ParameterSet& pset){

  // Get the various input parameters
  rootFileName         = pset.getUntrackedParameter<string>("rootFileName");
  isSimulation         = pset.getUntrackedParameter<bool>("isSimulation");
  writeTreeToFile      = pset.getUntrackedParameter<bool>("writeTreeToFile");
  makePlots            = pset.getUntrackedParameter<bool>("makePlots");
  makeComparisonPlots  = pset.getUntrackedParameter<bool>("makeComparisonPlots");
  refRootFile          = pset.getUntrackedParameter<string>("refRootFile","null");

  // flags to switch on/off individual modules
  makeOccupancyPlots   = pset.getUntrackedParameter<bool>("makeOccupancyPlots",true);
  makeStripPlots       = pset.getUntrackedParameter<bool>("makeStripPlots",true);
  makeWirePlots        = pset.getUntrackedParameter<bool>("makeWirePlots",true);
  makeRecHitPlots      = pset.getUntrackedParameter<bool>("makeRecHitPlots",true);
  makeSimHitPlots      = pset.getUntrackedParameter<bool>("makeSimHitPlots",true);
  makeSegmentPlots     = pset.getUntrackedParameter<bool>("makeSegmentPlots",true);
  makePedNoisePlots    = pset.getUntrackedParameter<bool>("makePedNoisePlots",true);
  makeEfficiencyPlots  = pset.getUntrackedParameter<bool>("makeEfficiencyPlots",true);
  makeGasGainPlots     = pset.getUntrackedParameter<bool>("makeGasGainPlots",true);

  // set counter to zero
  nEventsAnalyzed = 0;
  
  // Create the root file for the histograms
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Create object of class CSCValidationHistos to manage histograms
  histos = new CSCValHists();

  // book histos Eff histos
  hSSTE = new TH1F("hSSTE","hSSTE",20,0,20);
  hRHSTE = new TH1F("hRHSTE","hRHSTE",20,0,20);
  hSEff = new TH1F("hSEff","Segment Efficiency",10,0.5,10.5);
  hRHEff = new TH1F("hRHEff","recHit Efficiency",10,0.5,10.5);

  // setup trees to hold global position data for rechits and segments
  histos->setupTrees();


}

//////////////////
//  DESTRUCTOR  //
//////////////////
CSCValidation::~CSCValidation(){

  // produce final efficiency histograms
  histoEfficiency(hRHSTE,hRHEff);
  histoEfficiency(hSSTE,hSEff);

  // write histos to the specified file
  histos->writeHists(theFile);
  if (makePlots) histos->printPlots();
  if (makeComparisonPlots) histos->printComparisonPlots(refRootFile);
  if (writeTreeToFile) histos->writeTrees(theFile);
  theFile->cd();
  theFile->cd("recHits");
  hRHEff->Write();
  theFile->cd();
  theFile->cd("Segments");
  hSEff->Write();
  theFile->Close();

}

////////////////
//  Analysis  //
////////////////
void CSCValidation::analyze(const Event & event, const EventSetup& eventSetup){
  
  // increment counter
  nEventsAnalyzed++;

  int iRun   = event.id().run();
  int iEvent = event.id().event();

  LogInfo("EventInfo") << "Run: " << iRun << "    Event: " << iEvent;

  // Get the Digis
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  event.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
  event.getByLabel("muonCSCDigis","MuonCSCStripDigi",strips);

  // Get the CSC Geometry :
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits;
  event.getByLabel("csc2DRecHits",recHits);

  // Get the SimHits (if applicable)
  Handle<PSimHitContainer> simHits;
  if (isSimulation) event.getByLabel("g4SimHits", "MuonCSCHits", simHits);

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);

  /////////////////////
  // Run the modules //
  /////////////////////

  // Only do this for the first event
  if (nEventsAnalyzed == 1) doCalibrations(eventSetup);

  // look at various chamber occupancies
  if (makeOccupancyPlots) doOccupancies(strips,wires,recHits,cscSegments);

  // general look at strip digis
  if (makeStripPlots) doStripDigis(strips);

  // general look at wire digis
  if (makeWirePlots) doWireDigis(wires);

  // general look at rechits
  if (makeRecHitPlots) doRecHits(recHits,strips,cscGeom);

  // look at simHits
  if (isSimulation && makeSimHitPlots) doSimHits(recHits,simHits);

  // general look at Segments
  if (makeSegmentPlots) doSegments(cscSegments,cscGeom);

  // look at Pedestal Noise
  if (makePedNoisePlots) doPedestalNoise(strips);
  
  // look at recHit and segment efficiencies
  if (makeEfficiencyPlots) doEfficiencies(recHits, cscSegments);

  // gas gain
  if (makeGasGainPlots) doGasGain(*wires,*strips,*recHits);

}

// ==============================================
//
// look at Occupancies
//
// ==============================================

void CSCValidation::doOccupancies(edm::Handle<CSCStripDigiCollection> strips, edm::Handle<CSCWireDigiCollection> wires,
                                  edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments){

  bool wireo[2][4][4][36];
  bool stripo[2][4][4][36];
  bool rechito[2][4][4][36];
  bool segmento[2][4][4][36];

  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          wireo[e][s][r][c] = false;
          stripo[e][s][r][c] = false;
          rechito[e][s][r][c] = false;
          segmento[e][s][r][c] = false;
        }
      }
    }
  }

  //wires
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
  }
  
  //strips
  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      std::vector<int> myADCVals = digiItr->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
        diff = (float)myADCVals[iCount]-thisPedestal;
        if (diff > threshold) { thisStripFired = true; }
      }
      if (thisStripFired) {
        stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      }
    }
  }

  //rechits
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
  }

  //segments
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
  }

  // Fill occupancy plots
  histos->fillOccupancyHistos(wireo,stripo,rechito,segmento);

}

// ==============================================
//
// look at Calibrations
//
// ==============================================

void CSCValidation::doCalibrations(const edm::EventSetup& eventSetup){

  // Only do this for the first event
  if (nEventsAnalyzed == 1){

    LogDebug("Calibrations") << "Loading Calibrations...";

    // get the gains
    edm::ESHandle<CSCDBGains> hGains;
    eventSetup.get<CSCDBGainsRcd>().get( hGains );
    const CSCDBGains* pGains = hGains.product();
    // get the crosstalks
    edm::ESHandle<CSCDBCrosstalk> hCrosstalk;
    eventSetup.get<CSCDBCrosstalkRcd>().get( hCrosstalk );
    const CSCDBCrosstalk* pCrosstalk = hCrosstalk.product();
    // get the noise matrix
    edm::ESHandle<CSCDBNoiseMatrix> hNoiseMatrix;
    eventSetup.get<CSCDBNoiseMatrixRcd>().get( hNoiseMatrix );
    const CSCDBNoiseMatrix* pNoiseMatrix = hNoiseMatrix.product();
    // get pedestals
    edm::ESHandle<CSCDBPedestals> hPedestals;
    eventSetup.get<CSCDBPedestalsRcd>().get( hPedestals );
    const CSCDBPedestals* pPedestals = hPedestals.product();

    LogDebug("Calibrations") << "Calibrations Loaded!";

    for (int i = 0; i < 400; i++){
      int bin = i+1;
      histos->fillCalibHist(pGains->gains[i].gain_slope,"hCalibGainsS","Gains Slope",400,0,400,bin,"Calib");
      histos->fillCalibHist(pCrosstalk->crosstalk[i].xtalk_slope_left,"hCalibXtalkSL","Xtalk Slope Left",400,0,400,bin,"Calib");
      histos->fillCalibHist(pCrosstalk->crosstalk[i].xtalk_slope_right,"hCalibXtalkSR","Xtalk Slope Right",400,0,400,bin,"Calib");
      histos->fillCalibHist(pCrosstalk->crosstalk[i].xtalk_intercept_left,"hCalibXtalkIL","Xtalk Intercept Left",400,0,400,bin,"Calib");
      histos->fillCalibHist(pCrosstalk->crosstalk[i].xtalk_intercept_right,"hCalibXtalkIR","Xtalk Intercept Right",400,0,400,bin,"Calib");
      histos->fillCalibHist(pPedestals->pedestals[i].ped,"hCalibPedsP","Peds",400,0,400,bin,"Calib");
      histos->fillCalibHist(pPedestals->pedestals[i].rms,"hCalibPedsR","Peds RMS",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem33,"hCalibNoise33","Noise Matrix 33",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem34,"hCalibNoise34","Noise Matrix 34",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem35,"hCalibNoise35","Noise Matrix 35",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem44,"hCalibNoise44","Noise Matrix 44",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem45,"hCalibNoise45","Noise Matrix 45",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem46,"hCalibNoise46","Noise Matrix 46",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem55,"hCalibNoise55","Noise Matrix 55",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem56,"hCalibNoise56","Noise Matrix 56",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem57,"hCalibNoise57","Noise Matrix 57",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem66,"hCalibNoise66","Noise Matrix 66",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem67,"hCalibNoise67","Noise Matrix 67",400,0,400,bin,"Calib");
      histos->fillCalibHist(pNoiseMatrix->matrix[i].elem77,"hCalibNoise77","Noise Matrix 77",400,0,400,bin,"Calib");

 
    }

  } // end calib

}


// ==============================================
//
// look at WIRE DIGIs
//
// ==============================================

void CSCValidation::doWireDigis(edm::Handle<CSCWireDigiCollection> wires){

  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myWire = digiItr->getWireGroup();
      int myTBin = digiItr->getTimeBin();
      nWireGroupsTotal++;
      int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
      int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;
      histos->fill1DHist(myWire,"hWireAll","all wire group numbers",121,-0.5,120.5,"Digis");
      histos->fill1DHistByType(myWire,"hWireWire","Wiregroup Number",id,113,-0.5,112.5,"Digis");
      histos->fill1DHistByType(kLayer,"hWireLayer","Wires Fired per Layer",id,8,-0.5,7.5,"Digis");
      histos->fill1DHistByType(myTBin,"hWireTBinAll","Wire TimeBin Fired",id,21,-0.5,20.5,"Digis");
      if (kStation == 1) histos->fill1DHist(kCodeNarrow,"hWireCodeNarrow1","narrow scope wire code station 1",801,-400.5,400.5,"Digis");
      if (kStation == 2) histos->fill1DHist(kCodeNarrow,"hWireCodeNarrow2","narrow scope wire code station 2",801,-400.5,400.5,"Digis");
      if (kStation == 3) histos->fill1DHist(kCodeNarrow,"hWireCodeNarrow3","narrow scope wire code station 3",801,-400.5,400.5,"Digis");
      if (kStation == 4) histos->fill1DHist(kCodeNarrow,"hWireCodeNarrow4","narrow scope wire code station 4",801,-400.5,400.5,"Digis");
      histos->fill1DHist(kCodeBroad,"hWireCodeBroad","broad scope code for wires",33,-16.5,16.5,"Digis");
    }
  } // end wire loop

  // this way you can zero suppress but still store info on # events with no digis
  if (nWireGroupsTotal == 0) nWireGroupsTotal = -1;

  histos->fill1DHist(nWireGroupsTotal,"hWirenGroupsTotal","total number of wire groups",41,-0.5,40.5,"Digis");
  
}

// ==============================================
//
// look at STRIP DIGIs
//
// ==============================================

void CSCValidation::doStripDigis(edm::Handle<CSCStripDigiCollection> strips){

  int nStripsFired = 0;
  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
	diff = (float)myADCVals[iCount]-thisPedestal;
	if (diff > threshold) { thisStripFired = true; }
      } 
      if (thisStripFired) {
        nStripsFired++;
        int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
        int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;
        // fill strip histos
        histos->fill1DHist(myStrip,"hStripAll","all strip numbers",81,-0.5,80.5,"Digis");
        histos->fill1DHistByType(myStrip,"hStripStrip","Strip Number",id,81,-0.5,80.5,"Digis");
        histos->fill1DHistByType(kLayer,"hStripLayer","Strips Fired per Layer",id,8,-0.5,7.5,"Digis");
        if (kStation == 1) histos->fill1DHist(kCodeNarrow,"hStripCodeNarrow1","narrow scope strip code station 1",801,-400.5,400.5,"Digis");
        if (kStation == 2) histos->fill1DHist(kCodeNarrow,"hStripCodeNarrow2","narrow scope strip code station 2",801,-400.5,400.5,"Digis");
        if (kStation == 3) histos->fill1DHist(kCodeNarrow,"hStripCodeNarrow3","narrow scope strip code station 3",801,-400.5,400.5,"Digis");
        if (kStation == 4) histos->fill1DHist(kCodeNarrow,"hStripCodeNarrow4","narrow scope strip code station 4",801,-400.5,400.5,"Digis");
        histos->fill1DHist(kCodeBroad,"hStripCodeBroad","broad scope code for strips",33,-16.5,16.5,"Digis");
      }
    }
  } // end strip loop

  if (nStripsFired == 0) nStripsFired = -1;

  histos->fill1DHist(nStripsFired,"hStripNFired","total number of fired strips",101,-0.5,100.5,"Digis");

}

//=======================================================
//
// Look at the Pedestal Noise Distributions
//
//=======================================================

void CSCValidation::doPedestalNoise(edm::Handle<CSCStripDigiCollection> strips){

  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
      float TotalADC = getSignal(*strips, id, myStrip);
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float thisSignal = (1./6)*(myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
      float threshold = 13.3;
      if(kStation == 1 && kRing == 4)
	{
	  kRing = 1;
	  if(myStrip <= 16) myStrip += 64; // no trapping for any bizarreness
	}
      int globalStrip = cEndcap*( kStation*1000000 + kRing*100000 + kChamber*1000 + kLayer*100 + myStrip);
      if (TotalADC > threshold) { thisStripFired = true;}
      if (!thisStripFired){
	float ADC = thisSignal - thisPedestal;
        histos->fill1DHist(ADC,"hStripPed","Pedestal Noise Distribution",50,-25.,25.,"PedestalNoise");
        histos->fill1DHistByType(ADC,"hStripPedME","Pedestal Noise Distribution",id,50,-25.,25.,"PedestalNoise");
      }
    }
  }

}


// ==============================================
//
// look at RECHITs
//
// ==============================================

void CSCValidation::doRecHits(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCStripDigiCollection> strips, edm::ESHandle<CSCGeometry> cscGeom){

  // Get the RecHits collection :
  int nRecHits = recHits->size();
 
  // ---------------------
  // Loop over rechits 
  // ---------------------
  int iHit = 0;

  // Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    iHit++;

    // Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int cEndcap  = idrec.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    // Store rechit as a Local Point:
    LocalPoint rhitlocal = (*recIt).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    float zreco = rhitlocal.z();
    float phireco = rhitlocal.phi();
    LocalError rerrlocal = (*recIt).localPositionError();  
    float xxerr = rerrlocal.xx();
    float yyerr = rerrlocal.yy();
    float xyerr = rerrlocal.xy();

    // Find the strip containing this hit
    CSCRecHit2D::ChannelContainer hitstrips = (*recIt).channels();
    int nStrips     =  hitstrips.size();
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];

    // Find the charge associated with this hit

    CSCRecHit2D::ADCContainer adcs = (*recIt).adcs();
    int adcsize = adcs.size();
    float rHSumQ = 0;
    float sumsides = 0;
    for (int i = 0; i < adcsize; i++){
      if (i != 3 && i != 7 && i != 11){
        rHSumQ = rHSumQ + adcs[i]; 
      }
      if (adcsize == 12 && (i < 3 || i > 7) && i < 12){
        sumsides = sumsides + adcs[i];
      }
    }
    float rHratioQ = sumsides/rHSumQ;
    if (adcsize != 12) rHratioQ = -99;

    // Get the signal timing of this hit
    //float rHtime = (*recIt).tpeak();
    float rHtime = getTiming(*strips, idrec, centerStrip);

    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx   =  rhitglobal.x();
    float grecy   =  rhitglobal.y();
    float grecz   =  rhitglobal.z();
    float grecphi =  rhitglobal.phi();
    float grecr   =  sqrt(grecx*grecx + grecy+grecy);

    // Fill the rechit position branch
    if (writeTreeToFile) histos->fillRechitTree(xreco, yreco, grecx, grecy, kEndcap, kStation, kRing, kChamber, kLayer);
    
    // Simple occupancy variables
    int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;

    // Fill some histograms
    histos->fill1DHist(kCodeBroad,"hRHCodeBroad","broad scope code for recHits",33,-16.5,16.5,"recHits");
    if (kStation == 1) histos->fill1DHist(kCodeNarrow,"hRHCodeNarrow1","narrow scope recHit code station 1",801,-400.5,400.5,"recHits");
    if (kStation == 2) histos->fill1DHist(kCodeNarrow,"hRHCodeNarrow2","narrow scope recHit code station 2",801,-400.5,400.5,"recHits");
    if (kStation == 3) histos->fill1DHist(kCodeNarrow,"hRHCodeNarrow3","narrow scope recHit code station 3",801,-400.5,400.5,"recHits");
    if (kStation == 4) histos->fill1DHist(kCodeNarrow,"hRHCodeNarrow4","narrow scope recHit code station 4",801,-400.5,400.5,"recHits");
    histos->fill1DHistByType(kLayer,"hRHLayer","RecHits per Layer",idrec,8,-0.5,7.5,"recHits");
    histos->fill1DHistByType(xreco,"hRHX","Local X of recHit",idrec,160,-80.,80.,"recHits");
    histos->fill1DHistByType(yreco,"hRHY","Local Y of recHit",idrec,60,-180.,180.,"recHits");
    if (kStation == 1 && (kRing == 1 || kRing == 4)) histos->fill1DHistByType(rHSumQ,"hRHSumQ","Sum 3x3 recHit Charge",idrec,250,0,4000,"recHits");
    else histos->fill1DHistByType(rHSumQ,"hRHSumQ","Sum 3x3 recHit Charge",idrec,250,0,2000,"recHits");
    histos->fill1DHistByType(rHratioQ,"hRHRatioQ","Ratio (Ql+Qr)/Qt)",idrec,120,-0.1,1.1,"recHits");
    histos->fill1DHistByType(rHtime,"hRHTiming","recHit Timing",idrec,100,0,10,"recHits");
    histos->fill2DHistByStation(grecx,grecy,"hRHGlobal","recHit Global Position",idrec,400,-800.,800.,400,-800.,800.,"recHits");

  } //end rechit loop

  if (nRecHits == 0) nRecHits = -1;

  histos->fill1DHist(nRecHits,"hRHnrechits","recHits per Event (all chambers)",41,-0.5,40.5,"recHits");

}

// ==============================================
//
// look at SIMHITS
//
// ==============================================

void CSCValidation::doSimHits(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<PSimHitContainer> simHits){

  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {

    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    LocalPoint rhitlocal = (*recIt).localPosition();
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    float simHitXres = -99;
    float simHitYres = -99;
    float mindiffX   = 99;
    float mindiffY   = 10;
    // If MC, find closest muon simHit to check resolution:
    PSimHitContainer::const_iterator simIt;
    for (simIt = simHits->begin(); simIt != simHits->end(); simIt++){
      // Get DetID for this simHit:
      CSCDetId sId = (CSCDetId)(*simIt).detUnitId();
      // Check if the simHit detID matches that of current recHit
      // and make sure it is a muon hit:
      if (sId == idrec && abs((*simIt).particleType()) == 13){
        // Get the position of this simHit in local coordinate system:
        LocalPoint sHitlocal = (*simIt).localPosition();
        // Now we need to make reasonably sure that this simHit is
        // responsible for this recHit:
        if ((sHitlocal.x() - xreco) < mindiffX && (sHitlocal.y() - yreco) < mindiffY){
          simHitXres = (sHitlocal.x() - xreco);
          simHitYres = (sHitlocal.y() - yreco);
          mindiffX = (sHitlocal.x() - xreco);
        }
      }
    }

    histos->fill1DHistByType(simHitXres,"hRHResid","SimHitX - Reconstructed X",idrec,100,-1.0,1.0,"recHits");

  }

}

// ==============================================
//
// look at SEGMENTs
//
// ===============================================

void CSCValidation::doSegments(edm::Handle<CSCSegmentCollection> cscSegments, edm::ESHandle<CSCGeometry> cscGeom){

  // get CSC segment collection
  int nSegments = cscSegments->size();

  // -----------------------
  // loop over segments
  // -----------------------
  int iSegment = 0;
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    iSegment++;
    //
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();

    //
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();
    int nDOF       = 2*nhits-4;
    double chisqProb = ChiSquaredProbability( (double)chisq, nDOF );
    LocalPoint localPos = (*it).localPosition();
    float segX     = localPos.x();
    float segY     = localPos.y();
    float segZ     = localPos.z();
    float segPhi   = localPos.phi();
    LocalVector segDir = (*it).localDirection();
    double theta   = segDir.theta();
    double phi     = segDir.phi();

    //
    // try to get the CSC recHits that contribute to this segment.
    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    int nRH = (*it).nRecHits();
    int jRH = 0;
    HepMatrix sp(6,1);
    HepMatrix se(6,1);
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      int kEndcap  = idRH.endcap();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
      int kChamber = idRH.chamber();
      int kLayer   = idRH.layer();

      // Find the strip containing this hit
      CSCRecHit2D::ChannelContainer hitstrips = (*iRH).channels();
      int nStrips     =  hitstrips.size();
      int centerid    =  nStrips/2 + 1;
      int centerStrip =  hitstrips[centerid - 1];

      // If this segment has 6 hits, find the position of each hit on the strip in units of stripwidth and store values
      if (nRH == 6){
        float stpos = (*iRH).positionWithinStrip();
        se(kLayer,1) = (*iRH).errorWithinStrip();
        // Take into account half-strip staggering of layers (ME1/1 has no staggering)
        if (kStation == 1 && (kRing == 1 || kRing == 4)) sp(kLayer,1) = stpos + centerStrip;
        else{
          if (kLayer == 1 || kLayer == 3 || kLayer == 5) sp(kLayer,1) = stpos + centerStrip;
          if (kLayer == 2 || kLayer == 4 || kLayer == 6) sp(kLayer,1) = stpos - 0.5 + centerStrip;
        }
      }

    }

    float residual = -99;
    // Fit all points except layer 3, then compare expected value for layer 3 to reconstructed value
    if (nRH == 6){
      float expected = fitX(sp,se);
      residual = expected - sp(3,1);
    }

    // global transformation
    float globX = 0.;
    float globY = 0.;
    float globZ = 0.;
    float globpPhi = 0.;
    float globR = 0.;
    float globTheta = 0.;
    float globPhi   = 0.;
    const CSCChamber* cscchamber = cscGeom->chamber(id);
    if (cscchamber) {
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globX = globalPosition.x();
      globY = globalPosition.y();
      globZ = globalPosition.z();
      globpPhi =  globalPosition.phi();
      globR   =  sqrt(globX*globX + globY*globY);
      GlobalVector globalDirection = cscchamber->toGlobal(segDir);
      globTheta = globalDirection.theta();
      globPhi   = globalDirection.phi();
    }

    // Simple occupancy variables
    int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;

    // Fill segment position branch
    if (writeTreeToFile) histos->fillSegmentTree(segX, segY, globX, globY, kEndcap, kStation, kRing, kChamber);

    // Fill histos
    histos->fill1DHist(kCodeBroad,"hSCodeBroad","broad scope code for segmentss",33,-16.5,16.5,"Segments");
    if (kStation == 1) histos->fill1DHist(kCodeNarrow,"hSCodeNarrow1","narrow scope segment code station 1",801,-400.5,400.5,"Segments");
    if (kStation == 2) histos->fill1DHist(kCodeNarrow,"hSCodeNarrow2","narrow scope segment code station 2",801,-400.5,400.5,"Segments");
    if (kStation == 3) histos->fill1DHist(kCodeNarrow,"hSCodeNarrow3","narrow scope segment code station 3",801,-400.5,400.5,"Segments");
    if (kStation == 4) histos->fill1DHist(kCodeNarrow,"hSCodeNarrow4","narrow scope segment code station 4",801,-400.5,400.5,"Segments");
    histos->fill2DHistByStation(globX,globY,"hSGlobal","Segment Global Positions",id,400,-800.,800.,400,-800.,800.,"Segments");
    histos->fill1DHistByType(nhits,"hSnHits","N hits on Segments",id,8,-0.5,7.5,"Segments");
    histos->fill1DHistByType(theta,"hSTheta","local theta segments",id,128,-3.2,3.2,"Segments");
    histos->fill1DHistByType(residual,"hSResid","Fitted Position on Strip - Reconstructed for Layer 3",id,100,-0.5,0.5,"recHits");
    histos->fill1DHistByType(chisqProb,"hSChiSqProb","segments chi-squared probability",id,110,-0.05,1.05,"Segments");
    histos->fill1DHist(globTheta,"hSGlobalTheta","segment global theta",64,0,1.6,"Segments");
    histos->fill1DHist(globPhi,"hSGlobalPhi","segment global phi",128,-3.2,3.2,"Segments");


  } // end segment loop

  if (nSegments == 0) nSegments = -1;

  histos->fill1DHist(nSegments,"hSnSegments","number of segments per event",11,-0.5,10.5,"Segments");

}


//-------------------------------------------------------------------------------------
// Fits a straight line to a set of 5 points with errors.  Functions assumes 6 points
// and removes hit in layer 3.  It then returns the expected position value in layer 3
// based on the fit.
//-------------------------------------------------------------------------------------
float CSCValidation::fitX(HepMatrix points, HepMatrix errors){

  float S   = 0;
  float Sx  = 0;
  float Sy  = 0;
  float Sxx = 0;
  float Sxy = 0;
  float sigma2 = 0;

  for (int i=1;i<7;i++){
    if (i != 3){
      sigma2 = errors(i,1)*errors(i,1);
      S = S + (1/sigma2);
      Sy = Sy + (points(i,1)/sigma2);
      Sx = Sx + ((i)/sigma2);
      Sxx = Sxx + (i*i)/sigma2;
      Sxy = Sxy + (((i)*points(i,1))/sigma2);
    }
  }

  float delta = S*Sxx - Sx*Sx;
  float intercept = (Sxx*Sy - Sx*Sxy)/delta;
  float slope = (S*Sxy - Sx*Sy)/delta;

  float chi = 0;
  float chi2 = 0;

  // calculate chi2 (not currently used)
  for (int i=1;i<7;i++){
    chi = (points(i,1) - intercept - slope*i)/(errors(i,1));
    chi2 = chi2 + chi*chi;
  }

  return (intercept + slope*3);

}

//---------------------------------------------------------------------------------------
// Find the signal timing based on a weighted mean of the pulse.
// Function is meant to take the DetId and center strip number of a recHit and return
// the timing in units of time buckets (50ns)
//---------------------------------------------------------------------------------------

float CSCValidation::getTiming(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){

  float ADC[8];
  float timing = 0;
  float peakADC = 0;
  int peakTime = 0;

  // Loop over strip digis responsible for this recHit and sum charge
  CSCStripDigiCollection::DigiRangeIterator sIt;

  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idRH){
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
          float diff = 0;
          vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
          for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
            diff = (float)myADCVals[iCount]-thisPedestal;
            ADC[iCount] = diff;
            if (diff > peakADC){
              peakADC = diff;
              peakTime = iCount;
            }
          }
        }
      }

    }

  }

  float normADC;
  for (int i = 0; i < 8; i++){
    normADC = ADC[i]/ADC[peakTime];
    histos->fillProfileByChamber(i,normADC,"signal_profile","Normalized Signal Profile",idRH,8,-0.5,7.5,-0.1,1.1,"Digis");
  }

  histos->fill1DHistByChamber(ADC[0],"ped_subtracted","ADC in first time bin",idRH,400,-300,100,"Digis");
  histos->fill1DHist(ADC[0],"ped_subtracted_all","ADC in first time bin",400,-300,100,"Digis");

  timing = (ADC[2]*2 + ADC[3]*3 + ADC[4]*4 + ADC[5]*5 + ADC[6]*6)/(ADC[2] + ADC[3] + ADC[4] + ADC[5] + ADC[6]);

  return timing;

}

//----------------------------------------------------------------------------
// Calculate basic efficiencies for recHits and Segments
// Author: S. Stoynev
//----------------------------------------------------------------------------

void CSCValidation::doEfficiencies(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments){

  bool AllRecHits[2][4][4][36][6];
  bool AllSegments[2][4][4][36];
  //bool MultiSegments[2][4][4][36];
  for(int iE = 0;iE<2;iE++){
    for(int iS = 0;iS<4;iS++){
      for(int iR = 0; iR<4;iR++){
        for(int iC =0;iC<36;iC++){
          AllSegments[iE][iS][iR][iC] = false;
          //MultiSegments[iE][iS][iR][iC] = false;
          for(int iL=0;iL<6;iL++){
            AllRecHits[iE][iS][iR][iC][iL] = false;
          }
        }
      }
    }
  }
  
  for (CSCRecHit2DCollection::const_iterator recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    //CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    CSCDetId  idrec = (CSCDetId)(*recIt).cscDetId();
    AllRecHits[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;

  }
   

  for(CSCSegmentCollection::const_iterator segIt=cscSegments->begin(); segIt != cscSegments->end(); segIt++) {
    CSCDetId idseg  = (CSCDetId)(*segIt).cscDetId();
    //if(AllSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()]){
    //MultiSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()] = true;
    //}
    AllSegments[idseg.endcap() -1][idseg.station() -1][idseg.ring() -1][idseg.chamber() -1] = true;
  }

  
  for(int iE = 0;iE<2;iE++){
    for(int iS = 0;iS<4;iS++){
      for(int iR = 0; iR<4;iR++){
        for(int iC =0;iC<36;iC++){
          int NumberOfLayers = 0;
          for(int iL=0;iL<6;iL++){
            if(AllRecHits[iE][iS][iR][iC][iL]){
              NumberOfLayers++;
            }
          }
          int bin = 0;
          if (iS==0) bin = iR+1;
          else bin = (iS+1)*2 + (iR+1);
          if(NumberOfLayers>1){
            //if(!(MultiSegments[iE][iS][iR][iC])){
            if(AllSegments[iE][iS][iR][iC]){
              //---- Efficient segment evenents
              hSSTE->AddBinContent(bin);
            }
            //---- All segment events (normalization)
            hSSTE->AddBinContent(10+bin);
            //}
          }
          if(AllSegments[iE][iS][iR][iC]){
            if(NumberOfLayers==6){
              //---- Efficient rechit events
              hRHSTE->AddBinContent(bin);;
            }
            //---- All rechit events (normalization)
            hRHSTE->AddBinContent(10+bin);;
          }
        }
      }
    }
  }

}


//---------------------------------------------------------------------------------------
// Given a set of digis, the CSCDetId, and the central strip of your choosing, returns
// the avg. Signal-Pedestal for 6 time bin x 5 strip .
//
// Author: P. Jindal
//---------------------------------------------------------------------------------------

float CSCValidation::getSignal(const CSCStripDigiCollection& stripdigis, CSCDetId idCS, int centerStrip){

  float SigADC[5];
  float TotalADC = 0;
  SigADC[0] = 0;
  SigADC[1] = 0;
  SigADC[2] = 0;
  SigADC[3] = 0;
  SigADC[4] = 0;

 
  // Loop over strip digis 
  CSCStripDigiCollection::DigiRangeIterator sIt;
  
  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idCS){

      // First, find the Signal-Pedestal for center strip
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[0] = thisSignal - 6*thisPedestal;
	}
     // Now,find the Signal-Pedestal for neighbouring 4 strips
        if (thisStrip == (centerStrip+1)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[1] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip+2)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[2] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip-1)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[3] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip-2)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[4] = thisSignal - 6*thisPedestal;
	}
      }
      TotalADC = 0.2*(SigADC[0]+SigADC[1]+SigADC[2]+SigADC[3]+SigADC[4]);
    }
  }
  return TotalADC;
}

void CSCValidation::getEfficiency(float bin, float Norm, std::vector<float> &eff){
  //---- Efficiency with binomial error
  float Efficiency = 0.;
  float EffError = 0.;
  if(fabs(Norm)>0.000000001){
    Efficiency = bin/Norm;
    if(bin<Norm){
      EffError = sqrt( (1.-Efficiency)*Efficiency/Norm );
    }
  }
  eff[0] = Efficiency;
  eff[1] = EffError;
}

void CSCValidation::histoEfficiency(TH1F *readHisto, TH1F *writeHisto){
  std::vector<float> eff(2);
  int Nbins =  readHisto->GetSize()-2;//without underflows and overflows
  std::vector<float> bins(Nbins);
  std::vector<float> Efficiency(Nbins);
  std::vector<float> EffError(Nbins);
  float Num = 1;
  float Den = 1;
  for (int i=0;i<10;i++){
    Num = readHisto->GetBinContent(i+1);
    Den = readHisto->GetBinContent(i+11);
    getEfficiency(Num, Den, eff);
    Efficiency[i] = eff[0];
    EffError[i] = eff[1];
    writeHisto->SetBinContent(i+1, Efficiency[i]);
    writeHisto->SetBinError(i+1, EffError[i]);
  }
}

//---------------------------------------------------------------------------
// Module for looking at gas gains
// Author N. Terentiev
//---------------------------------------------------------------------------
void CSCValidation::bookForId(int casenmb, const int& idint,
                         const std::string& idstring ) {
  std::ostringstream ss;
    
  switch (casenmb) {
  
  case 6001:
  ss <<"gas_gain_wires_strips_rechits_present";
  mh_gas_gain_wires_strips_rechits_present[idint]=new TH1F(ss.str().c_str(),"",16, 0.0, 16.0);
  mh_gas_gain_wires_strips_rechits_present[idint]->GetXaxis()->SetTitle("Bit 1-Wire, 2-Strip, 3-RecHit");
  mh_gas_gain_wires_strips_rechits_present[idint]->GetYaxis()->SetTitle("Entries");
  mh_gas_gain_wires_strips_rechits_present[idint]->SetFillColor(4);
  ss.str(""); // clear
  break;
 
  case 6002:
  ss <<"gas_gain_wires_per_layer";
  mh_gas_gain_wires_per_layer[idint]=new TH1F(ss.str().c_str(),"",32,0.0,32.0);
  mh_gas_gain_wires_per_layer[idint]->GetXaxis()->SetTitle("Wire multiplicity per CSC layer"); 
  mh_gas_gain_wires_per_layer[idint]->GetYaxis()->SetTitle("Entries");
  mh_gas_gain_wires_per_layer[idint]->SetFillColor(4);
  ss.str(""); // clear
  break;

  case 6003:
  ss <<"gas_gain_rechits_per_layer";
  mh_gas_gain_rechits_per_layer[idint]=new TH1F(ss.str().c_str(),"",32,0.0,32.0);
  mh_gas_gain_rechits_per_layer[idint]->GetXaxis()->SetTitle("Rechit multiplicity per CSC layer");  
  mh_gas_gain_rechits_per_layer[idint]->GetYaxis()->SetTitle("Entries");
  mh_gas_gain_rechits_per_layer[idint]->SetFillColor(4);
  ss.str(""); // clear
  break;

  case 6004:
  ss <<idint<<"_gas_gain_rechit_adc_3_3_sum_location";
  mh_gas_gain_rechit_adc_3_3_sum_location[idint]=new TH2F(ss.str().c_str(),"",30,1.0,31.0,50,0.0,2000.0);  
  mh_gas_gain_rechit_adc_3_3_sum_location[idint]->GetXaxis()->SetTitle("Location=(layer-1)*nsegm+segm");
  mh_gas_gain_rechit_adc_3_3_sum_location[idint]->GetYaxis()->SetTitle("Rechit 3X3 ADC sum");
  mh_gas_gain_rechit_adc_3_3_sum_location[idint]->SetFillColor(4);
  mh_gas_gain_rechit_adc_3_3_sum_location[idint]->SetOption("BOX");
  ss.str(""); // clear
  break;

  case 6011:
  ss <<idint<<"_gas_gain_mean";
  mh_gas_gain_mean[idint]=new TH2F(ss.str().c_str(),"",40,0.0,40.0,30,1.0,31.0);
  mh_gas_gain_mean[idint]->GetXaxis()->SetTitle("CSC");   
  mh_gas_gain_mean[idint]->GetYaxis()->SetTitle("Location=(layer-1)*nsegm+segm");
  mh_gas_gain_mean[idint]->SetFillColor(4);
  mh_gas_gain_mean[idint]->SetOption("BOX");
  ss.str(""); // clear
  break;

  case 6012:
  ss <<idint<<"_gas_gain_entries";
  mh_gas_gain_entries[idint]=new TH2F(ss.str().c_str(),"",40,0.0,40.0,30,1.0,31.0);
  mh_gas_gain_entries[idint]->GetXaxis()->SetTitle("CSC");
  mh_gas_gain_entries[idint]->GetYaxis()->SetTitle("Location=(layer-1)*nsegm+segm");
  mh_gas_gain_entries[idint]->SetFillColor(4);
  mh_gas_gain_entries[idint]->SetOption("BOX");
  ss.str(""); // clear
  break;

  case 6101:
  ss <<idint<<"_wire_timing_mean";
  mh_wire_timing_mean[idint]=new TH2F(ss.str().c_str(),"",40,0.0,40.0,42,1.0,43.0);  
  mh_wire_timing_mean[idint]->GetXaxis()->SetTitle("CSC");
  mh_wire_timing_mean[idint]->GetYaxis()->SetTitle("AFEB");
  mh_wire_timing_mean[idint]->SetFillColor(4);
  mh_wire_timing_mean[idint]->SetOption("BOX");
  ss.str(""); // clear
  break;

  case 6102:
  ss <<idint<<"_wire_timing_entries";
  mh_wire_timing_entries[idint]=new TH2F(ss.str().c_str(),"",40,0.0,40.0,42,1.0,43.0);  
  mh_wire_timing_entries[idint]->GetXaxis()->SetTitle("CSC");
  mh_wire_timing_entries[idint]->GetYaxis()->SetTitle("AFEB");
  mh_wire_timing_entries[idint]->SetFillColor(4);
  mh_wire_timing_entries[idint]->SetOption("BOX");
  ss.str(""); // clear
  break;


  default:  std::cout<<"CSCValHists::bookForId:  No booked hist for case nmb "
                     <<casenmb<<std::endl;
  }
}

// Fill 1D histograms for gas gain
void CSCValidation::hf1ForId(std::map<int, TH1*>& mp, 
                             std::vector<TH1*>& th1, 
                             int flag, const int& id, float& x, float w) {
  std::map<int,TH1*>::iterator h;
  h=mp.find(id);
  if (h==mp.end()) {
     bookForId(flag,id,"");
     h=mp.find(id);
     th1.push_back(h->second);
     int nentries=0;
     for(Int_t i=1;i<=h->second->GetNbinsX();i++) {
        int m=(int)h->second->GetEntries();
        nentries=nentries+m;
     }
     if(nentries>0)
       std::cout<<"Hist "<<flag<<" "<<id<<" not empty when booking"<<std::endl;
 }
  h->second->Fill(x,w);
}

// Fill 2D histograms for gas gain
void CSCValidation::hf2ForId(std::map<int, TH2*>& mp, 
                             std::vector<TH2*>& th2,
                        int flag, const int& id, float& x, float& y,  float w) {
  std::map<int,TH2*>::iterator h;
  h=mp.find(id);
  if (h==mp.end()) {
     bookForId(flag,id,"");
     h=mp.find(id);
     th2.push_back(h->second);
     int nentries=0;
     for(Int_t i=1;i<=h->second->GetNbinsX();i++)
        for(Int_t j=1;j<=h->second->GetNbinsY();j++) {
           int m=(int)h->second->GetEntries();
        nentries=nentries+m;
       }
     if(nentries>0)
       std::cout<<"Hist "<<flag<<" "<<id<<" not empty when booking"<<std::endl;
 }
  h->second->Fill(x,y,w);
}

void CSCValidation::doGasGain(const CSCWireDigiCollection& wirecltn, 
                              const CSCStripDigiCollection&   strpcltn,
                              const CSCRecHit2DCollection& rechitcltn) {
     float x,adcsum;
     int channel,mult,wire,wiretbin,strip,layer,idafeb,idlayer,idchamber,wire_strip_rechit_present;
     CSCIndexer indexer;
     std::map<int,int>::iterator intIt;

     m_single_wire_layer.clear();
     m_nmbrechit_layer.clear();
     m_rechit_adc_3_3_sum.clear();
     m_rechit_strip.clear();
     m_csc_type.clear();

  if(nEventsAnalyzed==1) {

  m_mean_adc_3_3_sum.clear();
  m_nmb_adc_3_3_sum.clear();
  m_mean_wire_timing.clear();
  m_nmb_wire_timing.clear();
  m_index_csc.clear();

  // HV segments, their # and location in terms of wire groups

  m_wire_hvsegm.clear();
  std::map<int,std::vector<int> >::iterator intvecIt;
  //                    ME1a ME1b ME1/2 ME1/3 ME2/1 ME2/2 ME3/1 ME3/2 ME4/1 ME4/2 
  int csctype[10]=     {1,   2,   3,    4,    5,    6,    7,    8,    9,    10};
  int hvsegm_layer[10]={1,   1,   3,    3,    3,    5,    3,    5,    3,    5};
  int id;
  nmbhvsegm.clear();
  for(int i=0;i<10;i++) nmbhvsegm.push_back(hvsegm_layer[i]);
  // For ME1/1a
  std::vector<int> zer_1_1a(49,0);
  id=csctype[0];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_1_1a;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=48;wire++)  intvecIt->second[wire]=1;  // Segment 1

  // For ME1/1b
  std::vector<int> zer_1_1b(49,0);
  id=csctype[1];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_1_1b;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=48;wire++)  intvecIt->second[wire]=1;  // Segment 1
 
  // For ME1/2
  std::vector<int> zer_1_2(65,0);
  id=csctype[2];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_1_2;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=24;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=25;wire<=48;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=49;wire<=64;wire++) intvecIt->second[wire]=3;  // Segment 3
 
  // For ME1/3
  std::vector<int> zer_1_3(33,0);
  id=csctype[3];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_1_3;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=12;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=13;wire<=22;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=23;wire<=32;wire++) intvecIt->second[wire]=3;  // Segment 3
 
  // For ME2/1
  std::vector<int> zer_2_1(113,0);
  id=csctype[4];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_2_1;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=44;wire++)   intvecIt->second[wire]=1;  // Segment 1
  for(int wire=45;wire<=80;wire++)  intvecIt->second[wire]=2;  // Segment 2
  for(int wire=81;wire<=112;wire++) intvecIt->second[wire]=3;  // Segment 3
 
  // For ME2/2
  std::vector<int> zer_2_2(65,0);
  id=csctype[5];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_2_2;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=16;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=17;wire<=28;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=29;wire<=40;wire++) intvecIt->second[wire]=3;  // Segment 3
  for(int wire=41;wire<=52;wire++) intvecIt->second[wire]=4;  // Segment 4
  for(int wire=53;wire<=64;wire++) intvecIt->second[wire]=5;  // Segment 5

  // For ME3/1
  std::vector<int> zer_3_1(97,0);
  id=csctype[6];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_3_1;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=32;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=33;wire<=64;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=65;wire<=96;wire++) intvecIt->second[wire]=3;  // Segment 3
 
  // For ME3/2
  std::vector<int> zer_3_2(65,0);
  id=csctype[7];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_3_2;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=16;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=17;wire<=28;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=29;wire<=40;wire++) intvecIt->second[wire]=3;  // Segment 3
  for(int wire=41;wire<=52;wire++) intvecIt->second[wire]=4;  // Segment 4
  for(int wire=53;wire<=64;wire++) intvecIt->second[wire]=5;  // Segment 5

  // For ME4/1
  std::vector<int> zer_4_1(97,0);
  id=csctype[8];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_4_1;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=32;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=33;wire<=64;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=65;wire<=96;wire++) intvecIt->second[wire]=3;  // Segment 3

  // For ME4/2
  std::vector<int> zer_4_2(65,0);
  id=csctype[9];
  if(m_wire_hvsegm.find(id) == m_wire_hvsegm.end()) m_wire_hvsegm[id]=zer_4_2;
  intvecIt=m_wire_hvsegm.find(id);
  for(int wire=1;wire<=16;wire++)  intvecIt->second[wire]=1;  // Segment 1
  for(int wire=17;wire<=28;wire++) intvecIt->second[wire]=2;  // Segment 2
  for(int wire=29;wire<=40;wire++) intvecIt->second[wire]=3;  // Segment 3
  for(int wire=41;wire<=52;wire++) intvecIt->second[wire]=4;  // Segment 4
  for(int wire=53;wire<=64;wire++) intvecIt->second[wire]=5;  // Segment 5

  } // end of if(nEventsAnalyzed==1)


     // do wires, strips and rechits present?
     wire_strip_rechit_present=0;
     if(wirecltn.begin() != wirecltn.end())  
       wire_strip_rechit_present= wire_strip_rechit_present+1;
     if(strpcltn.begin() != strpcltn.end())    
       wire_strip_rechit_present= wire_strip_rechit_present+2;
     if(rechitcltn.begin() != rechitcltn.end())
       wire_strip_rechit_present= wire_strip_rechit_present+4;

     x=wire_strip_rechit_present;
     
  hf1ForId(mh_gas_gain_wires_strips_rechits_present,th1_gas_gain,6001,0,x,1.0); 

     if(wire_strip_rechit_present==7) {

       // cycle on wire collection for all CSC to select single wire hit layers
       CSCWireDigiCollection::DigiRangeIterator wiredetUnitIt;
 
       for(wiredetUnitIt=wirecltn.begin();wiredetUnitIt!=wirecltn.end();
          ++wiredetUnitIt) {
          const CSCDetId id = (*wiredetUnitIt).first;
          idlayer=indexer.dbIndex(id, channel);
          idchamber=idlayer/10;
          layer=id.layer();
          // looping in the layer of given CSC
          mult=0; wire=0; 
          const CSCWireDigiCollection::Range& range = (*wiredetUnitIt).second;
          for(CSCWireDigiCollection::const_iterator digiIt =
             range.first; digiIt!=range.second; ++digiIt){
             wire=(*digiIt).getWireGroup();
             wiretbin=(*digiIt).getTimeBin();
             mult++;
          }     // end of digis loop in layer

          x=mult;
          hf1ForId(mh_gas_gain_wires_per_layer,th1_gas_gain,6002,0,x,1.0);

          // select layers with single wire hit
          if(mult==1) {
            if(m_single_wire_layer.find(idlayer) == m_single_wire_layer.end())
              m_single_wire_layer[idlayer]=wire;

            // calculate average timing per afeb
            idafeb=idchamber*100+3*((wire-1)/8)+(layer+1)/2;
//            std::cout<<"Event "<<nEventsAnalyzed<<"   ";
//            std::cout<<idafeb<<"   "<<idchamber<<" "<<wire<<" "<<layer<<
//                       " "<<wiretbin<<std::endl;
            if(m_nmb_wire_timing.find(idafeb)==m_nmb_wire_timing.end())
              m_nmb_wire_timing[idafeb]=0.0;
            m_nmb_wire_timing[idafeb]=m_nmb_wire_timing[idafeb]+1.0;
            float entries= m_nmb_wire_timing[idafeb];
            if(m_mean_wire_timing.find(idafeb)==m_mean_wire_timing.end())
              m_mean_wire_timing[idafeb]=0.0;
            float mean_prev=m_mean_wire_timing[idafeb];
            m_mean_wire_timing[idafeb]=((entries-1.0)*mean_prev+wiretbin)/entries;
            //std::cout<<"idafeb mean entries   "<<idafeb<<"   "<<m_mean_wire_timing[idafeb]<<"  "<<entries<<std::endl;
          } // end of if(mult==1)
       }   // end of cycle on detUnit

       // The first pass thru rechit collection to count # rechits per layer
       CSCRecHit2DCollection::const_iterator recIt;
       for(recIt = rechitcltn.begin(); recIt != rechitcltn.end(); ++recIt) {
          CSCDetId id = (CSCDetId)(*recIt).cscDetId();
          idlayer=indexer.dbIndex(id, channel);
          if(m_single_wire_layer.find(idlayer) != m_single_wire_layer.end()) {
            if(m_nmbrechit_layer.find(idlayer) == m_nmbrechit_layer.end())
              m_nmbrechit_layer[idlayer]=0;
              m_nmbrechit_layer[idlayer]=m_nmbrechit_layer[idlayer]+1;
          }
       } // end of the first pass thru rechit collection

       for(intIt=m_nmbrechit_layer.begin();intIt!=m_nmbrechit_layer.end();++intIt) {
          x=(*intIt).second;
          hf1ForId(mh_gas_gain_rechits_per_layer,th1_gas_gain,6003,0,x,1.0);
       }

       // The second pass thru rechit collection, selecting single rechit/layer
       CSCRecHit2D::ADCContainer m_adc;
       CSCRecHit2D::ChannelContainer m_strip;
       for(recIt = rechitcltn.begin(); recIt != rechitcltn.end(); ++recIt) {
          CSCDetId id = (CSCDetId)(*recIt).cscDetId();
          idlayer=indexer.dbIndex(id, channel);
          idchamber=idlayer/10;
          // select layer with single rechit
          if(m_nmbrechit_layer.find(idlayer) != m_nmbrechit_layer.end() &&
            m_nmbrechit_layer[idlayer]==1 )  {          

            // getting strips comprising rechit
            m_strip=(CSCRecHit2D::ChannelContainer)(*recIt).channels(); 
            if(m_strip.size()==3 &&  
               m_rechit_strip.find(idlayer) == m_rechit_strip.end()) {        

              m_rechit_strip[idlayer]=m_strip[1]; //take central from 3 strips
              // save chamber type
              if(m_csc_type.find(idchamber) == m_csc_type.end())
                m_csc_type[idchamber]=histos->tempChamberType(id.station(),id.ring());
              // save chamber index
              if(id.station() != 1 && id.ring() !=4) 
                if(m_index_csc.find(idchamber) == m_index_csc.end())
                   m_index_csc[idchamber]=indexer.chamberIndex(id);

              // get 3X3 ADC Sum
              m_adc=(CSCRecHit2D::ADCContainer)(*recIt).adcs();
              std::vector<float> adc_left,adc_center,adc_right;
              int binmx=0;
              float adcmax=0.0;
              unsigned k=0;
 
              for(int i=0;i<3;i++) 
                 for(int j=0;j<4;j++){
                    if(m_adc[k]>adcmax) {adcmax=m_adc[k]; binmx=j;}
                    if(i==0) adc_left.push_back(m_adc[k]);
                    if(i==1) adc_center.push_back(m_adc[k]);
                    if(i==2) adc_right.push_back(m_adc[k]);
                    k=k+1;
                 }
        
              if(m_rechit_adc_3_3_sum.find(idlayer)==m_rechit_adc_3_3_sum.end()) {
                m_rechit_adc_3_3_sum[idlayer]=0.0;
                for(int j=binmx-1;j<=binmx+1;j++) {
                   m_rechit_adc_3_3_sum[idlayer]=m_rechit_adc_3_3_sum[idlayer]
                                                +adc_left[j]
                                                +adc_center[j]
                                                +adc_right[j];
                }
              }  // end of if (m_rechit_adc_3_3_sum.find(idlayer)==
            } // end of if if(m_strip.size()==3
          } // end of if if(m_nmbrechit_layer.find(idlayer) !=
       } // end of the second pass thru CSCRecHit2DCollection

       // Add 3X3 ADC Sum to corresponding location of gas gain
       // calculating average at the same time
       if(m_rechit_adc_3_3_sum.size() > 0) {
         std::map<int,float>::iterator floatIt;
         for(floatIt=m_rechit_adc_3_3_sum.begin();floatIt!=
m_rechit_adc_3_3_sum.end();++floatIt) {          
            idlayer=(*floatIt).first;
            if(m_single_wire_layer.find(idlayer) !=m_single_wire_layer.end() &&
               m_rechit_strip.find(idlayer) != m_rechit_strip.end()) {
               adcsum=(*floatIt).second;
               if(adcsum>0.0 && adcsum<2000.0) {
                 idchamber=idlayer/10;
                 layer=idlayer-idchamber*10;
                 wire= m_single_wire_layer[idlayer];
                 strip= m_rechit_strip[idlayer];

                 int chambertype=m_csc_type[idchamber];
                 int hvsgmtnmb=m_wire_hvsegm[chambertype][wire];
                 int nmbofhvsegm=nmbhvsegm[chambertype-1];
                 int location= (layer-1)*nmbofhvsegm+hvsgmtnmb;
                 int idlocation=idchamber*100+location;
                 float x=location;
                 hf2ForId(mh_gas_gain_rechit_adc_3_3_sum_location,th2_gas_gain,6004,idchamber,x,adcsum,1.0);                 
/*
 std::cout<<"idchamber layer wire strip chambertype hvsgmtnmb nmbofhvsegm location idlocation  "<<std::endl;  
std::cout<<idchamber<<" "<<layer<<" "<< wire<<" "<<strip<<" "<< chambertype<<" "<< 
hvsgmtnmb<<" "<< nmbofhvsegm<<" "<< location<<" "<< idlocation<<std::endl;
*/
                 if(m_nmb_adc_3_3_sum.find(idlocation)==m_nmb_adc_3_3_sum.end())
                   m_nmb_adc_3_3_sum[idlocation]=0.0;
                 m_nmb_adc_3_3_sum[idlocation]=m_nmb_adc_3_3_sum[idlocation]+1.0;
                 float entries= m_nmb_adc_3_3_sum[idlocation];
                 if(m_mean_adc_3_3_sum.find(idlocation)==m_mean_adc_3_3_sum.end())
                   m_mean_adc_3_3_sum[idlocation]=0.0;
                 float adc_prev=m_mean_adc_3_3_sum[idlocation];
                 m_mean_adc_3_3_sum[idlocation]=
                 ((entries-1.0)*adc_prev+adcsum)/entries;
               } // end if(adcsum>0.0 && adcsum<2000.0)
//....................              
            } // end of if(m_single_wire_layer.find(idlayer) !=
         } // end of for(floatIt=m_rechit_adc_3_3_sum.begin()
       }
     }   // end of if wire and strip present 
}


void CSCValidation::endJob() {

     std::cout<<"Events in "<<nEventsAnalyzed<<std::endl;

     // filling mean gas gain hists
     if(m_mean_adc_3_3_sum.size()>0) {
       std::map<int,float>::iterator floatIt;
       for(floatIt=m_mean_adc_3_3_sum.begin();floatIt!=m_mean_adc_3_3_sum.end();++floatIt) {
            int idlocation=(*floatIt).first;
            int idchamber=idlocation/100;
            float y=idlocation-idchamber*100;
            float wmean=(*floatIt).second;
            float wentr=m_nmb_adc_3_3_sum[idlocation];
            int endcap=idchamber/10000;
            int station=idchamber/1000-endcap*10;
            int idint=idchamber/100;
            int csc=idchamber/100; csc=idchamber-csc*100; float x=csc;
            hf2ForId(mh_gas_gain_mean,th2_gas_gain,6011,idint,x,y,wmean);
            hf2ForId(mh_gas_gain_entries,th2_gas_gain,6012,idint,x,y,wentr);
       }
     }     

     // filling mean and entries for wire timing hists
     if(m_mean_wire_timing.size()>0) {
       std::map<int,float>::iterator floatIt;
       for(floatIt=m_mean_wire_timing.begin();floatIt!=m_mean_wire_timing.end();++floatIt) {
            int idafeb=(*floatIt).first;
            int idchamber=idafeb/100;
            float y=idafeb-idchamber*100;
            float wmean=(*floatIt).second;
            float wentr=m_nmb_wire_timing[idafeb];
            int endcap=idchamber/10000;
            int station=idchamber/1000-endcap*10;
            int idint=idchamber/100;
            int csc=idchamber/100; csc=idchamber-csc*100; 
            float x=csc;
            hf2ForId(mh_wire_timing_mean,th2_gas_gain,6101,idint,x,y,wmean);
            hf2ForId(mh_wire_timing_entries,th2_gas_gain,6102,idint,x,y,wentr);
       }
     }
     // writing gas gain hists to folder GasGain
     theFile->cd();
     theFile->mkdir("GasGain");
     theFile->cd("GasGain");
     if(th1_gas_gain.size() > 0) 
       for(unsigned int i=0;i<th1_gas_gain.size();i++) th1_gas_gain[i]->Write();
     if(th2_gas_gain.size() > 0) 
       for(unsigned int i=0;i<th2_gas_gain.size();i++) th2_gas_gain[i]->Write();

}


DEFINE_FWK_MODULE(CSCValidation);

