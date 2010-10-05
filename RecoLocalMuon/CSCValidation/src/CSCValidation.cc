/*
 *  validation package for CSC DIGIs, RECHITs and SEGMENTs + more.
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
  rootFileName         = pset.getUntrackedParameter<string>("rootFileName","valHists.root");
  isSimulation         = pset.getUntrackedParameter<bool>("isSimulation",false);
  writeTreeToFile      = pset.getUntrackedParameter<bool>("writeTreeToFile",true);
  detailedAnalysis     = pset.getUntrackedParameter<bool>("detailedAnalysis",false);
  useDigis             = pset.getUntrackedParameter<bool>("useDigis",true);
  useTrigger           = pset.getUntrackedParameter<bool>("useTrigger",false);
  filterCSCEvents      = pset.getUntrackedParameter<bool>("filterCSCEvents",false);

  // input tags for collections
  stripDigiTag  = pset.getParameter<edm::InputTag>("stripDigiTag");
  wireDigiTag   = pset.getParameter<edm::InputTag>("wireDigiTag"); 
  compDigiTag   = pset.getParameter<edm::InputTag>("compDigiTag");
  cscRecHitTag  = pset.getParameter<edm::InputTag>("cscRecHitTag");
  cscSegTag     = pset.getParameter<edm::InputTag>("cscSegTag");
  saMuonTag     = pset.getParameter<edm::InputTag>("saMuonTag");
  l1aTag        = pset.getParameter<edm::InputTag>("l1aTag");
  simHitTag     = pset.getParameter<edm::InputTag>("simHitTag");

  // flags to switch on/off individual modules
  makeOccupancyPlots   = pset.getUntrackedParameter<bool>("makeOccupancyPlots",true);
  makeStripPlots       = pset.getUntrackedParameter<bool>("makeStripPlots",true);
  makeWirePlots        = pset.getUntrackedParameter<bool>("makeWirePlots",true);
  makeRecHitPlots      = pset.getUntrackedParameter<bool>("makeRecHitPlots",true);
  makeSimHitPlots      = pset.getUntrackedParameter<bool>("makeSimHitPlots",true);
  makeSegmentPlots     = pset.getUntrackedParameter<bool>("makeSegmentPlots",true);
  makeResolutionPlots  = pset.getUntrackedParameter<bool>("makeResolutionPlots",true);
  makePedNoisePlots    = pset.getUntrackedParameter<bool>("makePedNoisePlots",true);
  makeEfficiencyPlots  = pset.getUntrackedParameter<bool>("makeEfficiencyPlots",true);
  makeGasGainPlots     = pset.getUntrackedParameter<bool>("makeGasGainPlots",true);
  makeAFEBTimingPlots  = pset.getUntrackedParameter<bool>("makeAFEBTimingPlots",true);
  makeCompTimingPlots  = pset.getUntrackedParameter<bool>("makeCompTimingPlots",true);
  makeADCTimingPlots   = pset.getUntrackedParameter<bool>("makeADCTimingPlots",true);
  makeRHNoisePlots     = pset.getUntrackedParameter<bool>("makeRHNoisePlots",false);
  makeCalibPlots       = pset.getUntrackedParameter<bool>("makeCalibPlots",false);
  makeStandalonePlots  = pset.getUntrackedParameter<bool>("makeStandalonePlots",false);

  // set counters to zero
  nEventsAnalyzed = 0;
  rhTreeCount = 0;
  segTreeCount = 0;
  firstEvent = true; 
 
  // Create the root file for the histograms
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Create object of class CSCValHists to manage histograms
  histos = new CSCValHists();

  // book Occupancy Histos
  hOWires    = new TH2I("hOWires","Wire Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
  hOStrips   = new TH2I("hOStrips","Strip Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
  hORecHits  = new TH2I("hORecHits","RecHit Occupancy",36,0.5,36.5,20,0.5,20.5);
  hOSegments = new TH2I("hOSegments","Segments Occupancy",36,0.5,36.5,20,0.5,20.5);

  // book Eff histos
  hSSTE = new TH1F("hSSTE","hSSTE",40,0,40);
  hRHSTE = new TH1F("hRHSTE","hRHSTE",40,0,40);
  hSEff = new TH1F("hSEff","Segment Efficiency",20,0.5,20.5);
  hRHEff = new TH1F("hRHEff","recHit Efficiency",20,0.5,20.5);

  const int nChambers = 36; 
  const int nTypes = 18;
  float nCH_min = 0.5;
  float nCh_max = float(nChambers) + 0.5;
  float nT_min = 0.5;
  float nT_max = float(nTypes) + 0.5;

  hSSTE2 = new TH2F("hSSTE2","hSSTE2",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  hRHSTE2 = new TH2F("hRHSTE2","hRHSTE2",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  hStripSTE2 = new TH2F("hStripSTE2","hStripSTE2",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  hWireSTE2 = new TH2F("hWireSTE2","hWireSTE2",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  

  hEffDenominator = new TH2F("hEffDenominator","hEffDenominator",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  hSEff2 = new TH2F("hSEff2","Segment Efficiency 2D",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  hRHEff2 = new TH2F("hRHEff2","recHit Efficiency 2D",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);

  hStripEff2 = new TH2F("hStripEff2","strip Efficiency 2D",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
  hWireEff2 = new TH2F("hWireEff2","wire Efficiency 2D",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);

  hSensitiveAreaEvt = new TH2F("hSensitiveAreaEvt","events in sensitive area",nChambers,nCH_min,nCh_max, nTypes, nT_min, nT_max);
 
  // setup trees to hold global position data for rechits and segments
  if (writeTreeToFile) histos->setupTrees();


}

//////////////////
//  DESTRUCTOR  //
//////////////////
CSCValidation::~CSCValidation(){

  // produce final efficiency histograms
  histoEfficiency(hRHSTE,hRHEff);
  histoEfficiency(hSSTE,hSEff);
  hSEff2->Divide(hSSTE2,hEffDenominator,1.,1.,"B");
  hRHEff2->Divide(hRHSTE2,hEffDenominator,1.,1.,"B");
  hStripEff2->Divide(hStripSTE2,hEffDenominator,1.,1.,"B");
  hWireEff2->Divide(hWireSTE2,hEffDenominator,1.,1.,"B");

  histos->insertPlot(hSEff,"hSEff","Efficiency");
  histos->insertPlot(hRHEff,"hRHEff","Efficiency");

  histos->insertPlot(hSEff2,"hSEff2","Efficiency");
  histos->insertPlot(hRHEff2,"hRHEff2","Efficiency");
  histos->insertPlot(hStripEff2,"hStripff2","Efficiency");
  histos->insertPlot(hWireEff2,"hWireff2","Efficiency");
  
  histos->insertPlot(hSensitiveAreaEvt,"","Efficiency");

  // throw in occupancy plots so they're saved
  histos->insertPlot(hOWires,"hOWires","Digis");
  histos->insertPlot(hOStrips,"hOStrips","Digis");
  histos->insertPlot(hORecHits,"hORecHits","recHits");
  histos->insertPlot(hOSegments,"hOSegments","Segments");

  
  // write histos to the specified file
  histos->writeHists(theFile);
  if (writeTreeToFile) histos->writeTrees(theFile);
  theFile->Close();

}

////////////////
//  Analysis  //
////////////////
void CSCValidation::analyze(const Event & event, const EventSetup& eventSetup){
  
  // increment counter
  nEventsAnalyzed++;

  //int iRun   = event.id().run();
  //int iEvent = event.id().event();

  // Get the Digis
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCComparatorDigiCollection> compars;
  if (useDigis){
    event.getByLabel(stripDigiTag,strips);
    event.getByLabel(wireDigiTag,wires);
    event.getByLabel(compDigiTag,compars);
  }

  // Get the CSC Geometry :
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits;
  event.getByLabel(cscRecHitTag,recHits);

  // Get the SimHits (if applicable)
  Handle<PSimHitContainer> simHits;
  if (isSimulation) event.getByLabel(simHitTag, simHits);

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel(cscSegTag, cscSegments);

  // get the trigger collection
  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  if (useTrigger){
    event.getByLabel(l1aTag,pCollection);
  }

  // get the standalone muon collection
  Handle<reco::TrackCollection> saMuons;
  if (makeStandalonePlots) event.getByLabel(saMuonTag,saMuons);



  /////////////////////
  // Run the modules //
  /////////////////////

  // Only do this for the first event
  if (nEventsAnalyzed == 1 && makeCalibPlots) doCalibrations(eventSetup);

  // Look at the l1a trigger info (returns true if csc L1A present)
  bool CSCL1A = true;
  if (useTrigger) CSCL1A = doTrigger(pCollection);

  cleanEvent = true;
  if (filterCSCEvents) cleanEvent = filterEvents(recHits);
  
  // look at various chamber occupancies
  if (makeOccupancyPlots) doOccupancies(strips,wires,recHits,cscSegments);

  if (cleanEvent && CSCL1A){
    // general look at strip digis
    if (makeStripPlots && useDigis) doStripDigis(strips);

    // general look at wire digis
    if (makeWirePlots && useDigis) doWireDigis(wires);

    // general look at rechits
    if (makeRecHitPlots) doRecHits(recHits,strips,cscGeom);

    // look at simHits
    if (isSimulation && makeSimHitPlots) doSimHits(recHits,simHits);

    // general look at Segments
    if (makeSegmentPlots) doSegments(cscSegments,cscGeom);

    // look at hit resolution
    if (makeResolutionPlots) doResolution(cscSegments,cscGeom);

    // look at Pedestal Noise
    if (makePedNoisePlots && useDigis) doPedestalNoise(strips);
  
    // look at recHit and segment efficiencies
    if (makeEfficiencyPlots) doEfficiencies(wires,strips, recHits, cscSegments,cscGeom);

    // gas gain
    if (makeGasGainPlots && useDigis) doGasGain(*wires,*strips,*recHits);

    // AFEB timing
    if (makeAFEBTimingPlots && useDigis) doAFEBTiming(*wires);

    // Comparators timing
    if (makeCompTimingPlots && useDigis) doCompTiming(*compars);

    // strip ADC timing
    if (makeADCTimingPlots && useDigis) doADCTiming(*strips,*recHits);

    // recHit Noise
    if (makeRHNoisePlots && useDigis) doNoiseHits(recHits,cscSegments,cscGeom,strips);

    // look at standalone muons (not implemented yet)
    if (makeStandalonePlots) doStandalone(saMuons);

    firstEvent = false;

  }

}

// ==============================================
//
// event filter, returns false for 'messy' events
//
// ==============================================

bool CSCValidation::filterEvents(edm::Handle<CSCRecHit2DCollection> recHits){

  int rechito[2][4][4][36];
  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          rechito[e][s][r][c] = 0;
        }
      }
    }
  }

  //rechits
  CSCRecHit2DCollection::const_iterator recIt;
  int nRecHits = recHits->size();
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1]++;
  }

  int nHitChambers = 0;
  int nMessyChambers = 0;
  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          if (rechito[e][s][r][c] > 0) nHitChambers++;
          if (rechito[e][s][r][c] > 24 && r != 3) nMessyChambers++;
        }
      }
    }
  }

  // fill some histograms
  histos->fill1DHist(nHitChambers,"hNHitChambers","Number of Hit Chambers per Event",21,-0.5,20.5,"MessyEvents");
  histos->fill1DHist(nMessyChambers,"hNMessyChambers","Number of Messy Chambers (nhits > 24) per Event",11,-0.5,10.5,"MessyEvents");


  if (nRecHits < 0) return false;  // dummy for now
  if (nHitChambers > 12) return false;
  if (nMessyChambers > 0) return false;
  return true;

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

  bool hasWires = false;
  bool hasStrips = false;
  bool hasRecHits = false;
  bool hasSegments = false;

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

  if (useDigis){
    //wires
    for (CSCWireDigiCollection::DigiRangeIterator wi=wires->begin(); wi!=wires->end(); wi++) {
      CSCDetId id = (CSCDetId)(*wi).first;
      int kEndcap  = id.endcap();
      int kRing    = id.ring();
      int kStation = id.station();
      int kChamber = id.chamber();
      std::vector<CSCWireDigi>::const_iterator wireIt = (*wi).second.first;
      std::vector<CSCWireDigi>::const_iterator lastWire = (*wi).second.second;
      for( ; wireIt != lastWire; ++wireIt){
        if (!wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
          wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
          hOWires->Fill(kChamber,typeIndex(id));
          histos->fill1DHist(chamberSerial(id),"hOWireSerial","Wire Occupancy by Chamber Serial",601,-0.5,600.5,"Digis");
          hasWires = true;
        }
      }
    }
  
    //strips
    for (CSCStripDigiCollection::DigiRangeIterator si=strips->begin(); si!=strips->end(); si++) {
      CSCDetId id = (CSCDetId)(*si).first;
      int kEndcap  = id.endcap();
      int kRing    = id.ring();
      int kStation = id.station();
      int kChamber = id.chamber();
      std::vector<CSCStripDigi>::const_iterator stripIt = (*si).second.first;
      std::vector<CSCStripDigi>::const_iterator lastStrip = (*si).second.second;
      for( ; stripIt != lastStrip; ++stripIt) {
        std::vector<int> myADCVals = stripIt->getADCCounts();
        bool thisStripFired = false;
        float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
        float threshold = 13.3 ;
        float diff = 0.;
        for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
          diff = (float)myADCVals[iCount]-thisPedestal;
          if (diff > threshold) { thisStripFired = true; }
        }
        if (thisStripFired) {
          if (!stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
            stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
            hOStrips->Fill(kChamber,typeIndex(id));
            histos->fill1DHist(chamberSerial(id),"hOStripSerial","Strip Occupancy by Chamber Serial",601,-0.5,600.5,"Digis");
            hasStrips = true;
          }
        }
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
    if (!rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
      rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      histos->fill1DHist(chamberSerial(idrec),"hORecHitsSerial","RecHit Occupancy by Chamber Serial",601,-0.5,600.5,"recHits");
      hORecHits->Fill(kChamber,typeIndex(idrec));
      hasRecHits = true;
    }
  }

  //segments
  for(CSCSegmentCollection::const_iterator segIt=cscSegments->begin(); segIt != cscSegments->end(); segIt++) {
    CSCDetId id  = (CSCDetId)(*segIt).cscDetId();
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    if (!segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
      segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      histos->fill1DHist(chamberSerial(id),"hOSegmentsSerial","Segment Occupancy by Chamber Serial",601,-0.5,600.5,"Segments");
      hOSegments->Fill(kChamber,typeIndex(id));
      hasSegments = true;
    }
  }

  // overall CSC occupancy (events with CSC data compared to total)
  histos->fill1DHist(1,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");
  if (hasWires) histos->fill1DHist(3,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");
  if (hasStrips) histos->fill1DHist(5,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");
  if (hasWires && hasStrips) histos->fill1DHist(7,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");
  if (hasRecHits) histos->fill1DHist(9,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");
  if (hasSegments) histos->fill1DHist(11,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");
  if (!cleanEvent) histos->fill1DHist(13,"hCSCOccupancy","overall CSC occupancy",15,-0.5,14.5,"GeneralHists");

}

// ==============================================
//
// look at Trigger info
//
// ==============================================

bool CSCValidation::doTrigger(edm::Handle<L1MuGMTReadoutCollection> pCollection){

  vector<L1MuGMTReadoutRecord> L1Mrec = pCollection->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;

  bool csc_l1a  = false;
  bool dt_l1a   = false;
  bool rpcf_l1a = false;
  bool rpcb_l1a = false;
  bool beamHaloTrigger = false;

  int myBXNumber = -1000;

  for(igmtrr=L1Mrec.begin(); igmtrr!=L1Mrec.end(); igmtrr++) {
    std::vector<L1MuRegionalCand>::const_iterator iter1;
    std::vector<L1MuRegionalCand> rmc;

    // CSC
    int icsc = 0;
    rmc = igmtrr->getCSCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( !(*iter1).empty() ) {
        icsc++;
        int kQuality = (*iter1).quality();   // kQuality = 1 means beam halo
        if (kQuality == 1) beamHaloTrigger = true;
      }
    }
    if (igmtrr->getBxInEvent() == 0 && icsc>0) csc_l1a = true;
    if (igmtrr->getBxInEvent() == 0 ) { myBXNumber = igmtrr->getBxNr(); }

    // DT
    int idt = 0;
    rmc = igmtrr->getDTBXCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( !(*iter1).empty() ) {
        idt++;
      }
    }
    if(igmtrr->getBxInEvent()==0 && idt>0) dt_l1a = true;

    // RPC Barrel
    int irpcb = 0;
    rmc = igmtrr->getBrlRPCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( !(*iter1).empty() ) {
        irpcb++;
      }
    }
    if(igmtrr->getBxInEvent()==0 && irpcb>0) rpcb_l1a = true;

    // RPC Forward
    int irpcf = 0;
    rmc = igmtrr->getFwdRPCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( !(*iter1).empty() ) {
        irpcf++;
      }
    }
    if(igmtrr->getBxInEvent()==0 && irpcf>0) rpcf_l1a = true;

  }

  // Fill some histograms with L1A info
  if (csc_l1a)          histos->fill1DHist(myBXNumber,"vtBXNumber","BX Number",4001,-0.5,4000.5,"Trigger");
  if (csc_l1a)          histos->fill1DHist(1,"vtBits","trigger bits",11,-0.5,10.5,"Trigger");
  if (dt_l1a)           histos->fill1DHist(2,"vtBits","trigger bits",11,-0.5,10.5,"Trigger");
  if (rpcb_l1a)         histos->fill1DHist(3,"vtBits","trigger bits",11,-0.5,10.5,"Trigger");
  if (rpcf_l1a)         histos->fill1DHist(4,"vtBits","trigger bits",11,-0.5,10.5,"Trigger");
  if (beamHaloTrigger)  histos->fill1DHist(8,"vtBits","trigger bits",11,-0.5,10.5,"Trigger");

  if (csc_l1a) {
    histos->fill1DHist(1,"vtCSCY","trigger bits",11,-0.5,10.5,"Trigger");
    if (dt_l1a)   histos->fill1DHist(2,"vtCSCY","trigger bits",11,-0.5,10.5,"Trigger");
    if (rpcb_l1a) histos->fill1DHist(3,"vtCSCY","trigger bits",11,-0.5,10.5,"Trigger");
    if (rpcf_l1a) histos->fill1DHist(4,"vtCSCY","trigger bits",11,-0.5,10.5,"Trigger");
    if (  dt_l1a || rpcb_l1a || rpcf_l1a)  histos->fill1DHist(5,"vtCSCY","trigger bits",11,-0.5,10.5,"Trigger");
    if (!(dt_l1a || rpcb_l1a || rpcf_l1a)) histos->fill1DHist(6,"vtCSCY","trigger bits",11,-0.5,10.5,"Trigger");
  } else {
    histos->fill1DHist(1,"vtCSCN","trigger bits",11,-0.5,10.5,"Trigger");
    if (dt_l1a)   histos->fill1DHist(2,"vtCSCN","trigger bits",11,-0.5,10.5,"Trigger");
    if (rpcb_l1a) histos->fill1DHist(3,"vtCSCN","trigger bits",11,-0.5,10.5,"Trigger");
    if (rpcf_l1a) histos->fill1DHist(4,"vtCSCN","trigger bits",11,-0.5,10.5,"Trigger");
    if (  dt_l1a || rpcb_l1a || rpcf_l1a)  histos->fill1DHist(5,"vtCSCN","trigger bits",11,-0.5,10.5,"Trigger");
    if (!(dt_l1a || rpcb_l1a || rpcf_l1a)) histos->fill1DHist(6,"vtCSCN","trigger bits",11,-0.5,10.5,"Trigger");
  }

  // if valid CSC L1A then return true for possible use elsewhere

  if (csc_l1a) return true;
  
  return false;

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

  }


}


// ==============================================
//
// look at WIRE DIGIs
//
// ==============================================

void CSCValidation::doWireDigis(edm::Handle<CSCWireDigiCollection> wires){

  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator dWDiter=wires->begin(); dWDiter!=wires->end(); dWDiter++) {
    CSCDetId id = (CSCDetId)(*dWDiter).first;
    std::vector<CSCWireDigi>::const_iterator wireIter = (*dWDiter).second.first;
    std::vector<CSCWireDigi>::const_iterator lWire = (*dWDiter).second.second;
    for( ; wireIter != lWire; ++wireIter) {
      int myWire = wireIter->getWireGroup();
      int myTBin = wireIter->getTimeBin();
      nWireGroupsTotal++;
      histos->fill1DHistByType(myWire,"hWireWire","Wiregroup Numbers Fired",id,113,-0.5,112.5,"Digis");
      histos->fill1DHistByType(myTBin,"hWireTBin","Wire TimeBin Fired",id,17,-0.5,16.5,"Digis");
      histos->fillProfile(chamberSerial(id),myTBin,"hWireTBinProfile","Wire TimeBin Fired",601,-0.5,600.5,-0.5,16.5,"Digis");
      if (detailedAnalysis){
        histos->fill1DHistByLayer(myWire,"hWireWire","Wiregroup Numbers Fired",id,113,-0.5,112.5,"WireNumberByLayer");
        histos->fill1DHistByLayer(myTBin,"hWireTBin","Wire TimeBin Fired",id,17,-0.5,16.5,"WireTimeByLayer");
      }
    }
  } // end wire loop

  // this way you can zero suppress but still store info on # events with no digis
  if (nWireGroupsTotal == 0) nWireGroupsTotal = -1;

  histos->fill1DHist(nWireGroupsTotal,"hWirenGroupsTotal","Wires Fired Per Event",151,-0.5,150.5,"Digis");
  
}

// ==============================================
//
// look at STRIP DIGIs
//
// ==============================================

void CSCValidation::doStripDigis(edm::Handle<CSCStripDigiCollection> strips){

  int nStripsFired = 0;
  for (CSCStripDigiCollection::DigiRangeIterator dSDiter=strips->begin(); dSDiter!=strips->end(); dSDiter++) {
    CSCDetId id = (CSCDetId)(*dSDiter).first;
    std::vector<CSCStripDigi>::const_iterator stripIter = (*dSDiter).second.first;
    std::vector<CSCStripDigi>::const_iterator lStrip = (*dSDiter).second.second;
    for( ; stripIter != lStrip; ++stripIter) {
      int myStrip = stripIter->getStrip();
      std::vector<int> myADCVals = stripIter->getADCCounts();
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
        // fill strip histos
        histos->fill1DHistByType(myStrip,"hStripStrip","Strip Number",id,81,-0.5,80.5,"Digis");
        if (detailedAnalysis){
          histos->fill1DHistByLayer(myStrip,"hStripStrip","Strip Number",id,81,-0.5,80.5,"StripNumberByLayer");
        }
      }
    }
  } // end strip loop

  if (nStripsFired == 0) nStripsFired = -1;

  histos->fill1DHist(nStripsFired,"hStripNFired","Fired Strips per Event",251,-0.5,250.5,"Digis");

}

//=======================================================
//
// Look at the Pedestal Noise Distributions
//
//=======================================================

void CSCValidation::doPedestalNoise(edm::Handle<CSCStripDigiCollection> strips){

  for (CSCStripDigiCollection::DigiRangeIterator dPNiter=strips->begin(); dPNiter!=strips->end(); dPNiter++) {
    CSCDetId id = (CSCDetId)(*dPNiter).first;
    std::vector<CSCStripDigi>::const_iterator pedIt = (*dPNiter).second.first;
    std::vector<CSCStripDigi>::const_iterator lStrip = (*dPNiter).second.second;
    for( ; pedIt != lStrip; ++pedIt) {
      int myStrip = pedIt->getStrip();
      std::vector<int> myADCVals = pedIt->getADCCounts();
      float TotalADC = getSignal(*strips, id, myStrip);
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float thisSignal = (1./6)*(myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
      float threshold = 13.3;
      if(id.station() == 1 && id.ring() == 4)
	{
	  if(myStrip <= 16) myStrip += 64; // no trapping for any bizarreness
	}
      if (TotalADC > threshold) { thisStripFired = true;}
      if (!thisStripFired){
	float ADC = thisSignal - thisPedestal;
        histos->fill1DHist(ADC,"hStripPed","Pedestal Noise Distribution",50,-25.,25.,"PedestalNoise");
        histos->fill1DHistByType(ADC,"hStripPedME","Pedestal Noise Distribution",id,50,-25.,25.,"PedestalNoise");
        histos->fillProfile(chamberSerial(id),ADC,"hStripPedMEProfile","Wire TimeBin Fired",601,-0.5,600.5,-25,25,"PedestalNoise");
        if (detailedAnalysis){
          histos->fill1DHistByLayer(ADC,"hStripPedME","Pedestal Noise Distribution",id,50,-25.,25.,"PedestalNoiseByLayer");
        }
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
  CSCRecHit2DCollection::const_iterator dRHIter;
  for (dRHIter = recHits->begin(); dRHIter != recHits->end(); dRHIter++) {
    iHit++;

    // Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*dRHIter).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    // Store rechit as a Local Point:
    LocalPoint rhitlocal = (*dRHIter).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    LocalError rerrlocal = (*dRHIter).localPositionError();  
    //these errors are squared!
    float xxerr = rerrlocal.xx();
    float yyerr = rerrlocal.yy();
    float xyerr = rerrlocal.xy();
    // errors in strip units
    float stpos = (*dRHIter).positionWithinStrip();
    float sterr = (*dRHIter).errorWithinStrip();

    // Find the strip containing this hit
    CSCRecHit2D::ChannelContainer hitstrips = (*dRHIter).channels();
    int nStrips     =  hitstrips.size();
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];

    // Find the charge associated with this hit
    CSCRecHit2D::ADCContainer adcs = (*dRHIter).adcs();
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
    float rHtime = 0;
    if (useDigis){
      rHtime = getTiming(*strips, idrec, centerStrip);
    }
    else{
      rHtime = (*dRHIter).tpeak()/50.;
    }

    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx   =  rhitglobal.x();
    float grecy   =  rhitglobal.y();

    // Fill the rechit position branch
    if (writeTreeToFile && rhTreeCount < 1500000){
      histos->fillRechitTree(xreco, yreco, grecx, grecy, kEndcap, kStation, kRing, kChamber, kLayer);
      rhTreeCount++;
    }    

    // Fill some histograms
    // only fill if 3 strips were used in the hit
    histos->fill2DHistByStation(grecx,grecy,"hRHGlobal","recHit Global Position",idrec,100,-800.,800.,100,-800.,800.,"recHits");
    if (kStation == 1 && (kRing == 1 || kRing == 4)) histos->fill1DHistByType(rHSumQ,"hRHSumQ","Sum 3x3 recHit Charge",idrec,125,0,4000,"recHits");
    else histos->fill1DHistByType(rHSumQ,"hRHSumQ","Sum 3x3 recHit Charge",idrec,125,0,2000,"recHits");
    histos->fill1DHistByType(rHratioQ,"hRHRatioQ","Charge Ratio (Ql+Qr)/Qt",idrec,120,-0.1,1.1,"recHits");
    histos->fill1DHistByType(rHtime,"hRHTiming","recHit Timing",idrec,100,0,10,"recHits");
    histos->fill1DHistByType(sqrt(xxerr),"hRHxerr","RecHit Error on Local X",idrec,100,-0.1,2,"recHits");
    histos->fill1DHistByType(sqrt(yyerr),"hRHyerr","RecHit Error on Local Y",idrec,100,-0.1,2,"recHits");
    histos->fill1DHistByType(xyerr,"hRHxyerr","Corr. RecHit XY Error",idrec,100,-1,2,"recHits");
    if (adcsize == 12) histos->fill1DHistByType(stpos,"hRHstpos","Reconstructed Position on Strip",idrec,120,-0.6,0.6,"recHits");
    histos->fill1DHistByType(sterr,"hRHsterr","Estimated Error on Strip Measurement",idrec,120,-0.05,0.25,"recHits");
    histos->fillProfile(chamberSerial(idrec),rHSumQ,"hRHSumQProfile","Sum 3x3 recHit Charge",601,-0.5,600.5,0,4000,"recHits");
    histos->fillProfile(chamberSerial(idrec),rHtime,"hRHTimingProfile","recHit Timing",601,-0.5,600.5,-1,11,"recHits");
    if (detailedAnalysis){
      if (kStation == 1 && (kRing == 1 || kRing == 4)) histos->fill1DHistByLayer(rHSumQ,"hRHSumQ","Sum 3x3 recHit Charge",idrec,125,0,4000,"RHQByLayer");
      else histos->fill1DHistByLayer(rHSumQ,"hRHSumQ","Sum 3x3 recHit Charge",idrec,125,0,2000,"RHQByLayer");
      histos->fill1DHistByLayer(rHratioQ,"hRHRatioQ","Charge Ratio (Ql+Qr)/Qt",idrec,120,-0.1,1.1,"RHQByLayer");
      histos->fill1DHistByLayer(rHtime,"hRHTiming","recHit Timing",idrec,100,0,10,"RHTimingByLayer");
      histos->fill2DHistByLayer(xreco,yreco,"hRHLocalXY","recHit Local Position",idrec,50,-100.,100.,75,-150.,150.,"RHLocalXYByLayer");
      histos->fill1DHistByLayer(sqrt(xxerr),"hRHxerr","RecHit Error on Local X",idrec,100,-0.1,2,"RHErrorsByLayer");
      histos->fill1DHistByLayer(sqrt(yyerr),"hRHyerr","RecHit Error on Local Y",idrec,100,-0.1,2,"RHErrorsByLayer");
      histos->fill1DHistByType(stpos,"hRHstpos","Reconstructed Position on Strip",idrec,120,-0.6,0.6,"RHStripPosByLayer");
      histos->fill1DHistByType(sterr,"hRHsterr","Estimated Error on Strip Measurement",idrec,120,-0.05,0.25,"RHStripPosByLayer");
    }

  } //end rechit loop

  if (nRecHits == 0) nRecHits = -1;

  histos->fill1DHist(nRecHits,"hRHnrechits","recHits per Event (all chambers)",151,-0.5,150.5,"recHits");

}

// ==============================================
//
// look at SIMHITS
//
// ==============================================

void CSCValidation::doSimHits(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<PSimHitContainer> simHits){

  CSCRecHit2DCollection::const_iterator dSHrecIter;
  for (dSHrecIter = recHits->begin(); dSHrecIter != recHits->end(); dSHrecIter++) {

    CSCDetId idrec = (CSCDetId)(*dSHrecIter).cscDetId();
    LocalPoint rhitlocal = (*dSHrecIter).localPosition();
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    float simHitXres = -99;
    float simHitYres = -99;
    float mindiffX   = 99;
    float mindiffY   = 10;
    // If MC, find closest muon simHit to check resolution:
    PSimHitContainer::const_iterator dSHsimIter;
    for (dSHsimIter = simHits->begin(); dSHsimIter != simHits->end(); dSHsimIter++){
      // Get DetID for this simHit:
      CSCDetId sId = (CSCDetId)(*dSHsimIter).detUnitId();
      // Check if the simHit detID matches that of current recHit
      // and make sure it is a muon hit:
      if (sId == idrec && abs((*dSHsimIter).particleType()) == 13){
        // Get the position of this simHit in local coordinate system:
        LocalPoint sHitlocal = (*dSHsimIter).localPosition();
        // Now we need to make reasonably sure that this simHit is
        // responsible for this recHit:
        if ((sHitlocal.x() - xreco) < mindiffX && (sHitlocal.y() - yreco) < mindiffY){
          simHitXres = (sHitlocal.x() - xreco);
          simHitYres = (sHitlocal.y() - yreco);
          mindiffX = (sHitlocal.x() - xreco);
        }
      }
    }

    histos->fill1DHistByType(simHitXres,"hRHResid","SimHitX - Reconstructed X",idrec,100,-1.0,1.0,"Resolution");
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
  for(CSCSegmentCollection::const_iterator dSiter=cscSegments->begin(); dSiter != cscSegments->end(); dSiter++) {
    iSegment++;
    //
    CSCDetId id  = (CSCDetId)(*dSiter).cscDetId();
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();

    //
    float chisq    = (*dSiter).chi2();
    int nhits      = (*dSiter).nRecHits();
    int nDOF       = 2*nhits-4;
    double chisqProb = ChiSquaredProbability( (double)chisq, nDOF );
    LocalPoint localPos = (*dSiter).localPosition();
    float segX     = localPos.x();
    float segY     = localPos.y();
    LocalVector segDir = (*dSiter).localDirection();
    double theta   = segDir.theta();

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


    // Fill segment position branch
    if (writeTreeToFile && segTreeCount < 1500000){
      histos->fillSegmentTree(segX, segY, globX, globY, kEndcap, kStation, kRing, kChamber);
      segTreeCount++;
    }

    // Fill histos
    histos->fill2DHistByStation(globX,globY,"hSGlobal","Segment Global Positions;global x (cm)",id,100,-800.,800.,100,-800.,800.,"Segments");
    histos->fill1DHistByType(nhits,"hSnHits","N hits on Segments",id,8,-0.5,7.5,"Segments");
    histos->fill1DHistByType(theta,"hSTheta","local theta segments",id,128,-3.2,3.2,"Segments");
    histos->fill1DHistByType((chisq/nDOF),"hSChiSq","segments chi-squared/ndof",id,110,-0.05,10.5,"Segments");
    histos->fill1DHistByType(chisqProb,"hSChiSqProb","segments chi-squared probability",id,110,-0.05,1.05,"Segments");
    histos->fill1DHist(globTheta,"hSGlobalTheta","segment global theta",128,0,3.2,"Segments");
    histos->fill1DHist(globPhi,"hSGlobalPhi","segment global phi",128,-3.2,3.2,"Segments");
    histos->fillProfile(chamberSerial(id),nhits,"hSnHitsProfile","N hits on Segments",601,-0.5,600.5,-0.5,7.5,"Segments");
    if (detailedAnalysis){
      histos->fill1DHistByChamber(nhits,"hSnHits","N hits on Segments",id,8,-0.5,7.5,"HitsOnSegmentByChamber");
      histos->fill1DHistByChamber(theta,"hSTheta","local theta segments",id,128,-3.2,3.2,"DetailedSegments");
      histos->fill1DHistByChamber((chisq/nDOF),"hSChiSq","segments chi-squared/ndof",id,110,-0.05,10.5,"SegChi2ByChamber");
      histos->fill1DHistByChamber(chisqProb,"hSChiSqProb","segments chi-squared probability",id,110,-0.05,1.05,"SegChi2ByChamber");
    }

  } // end segment loop

  if (nSegments == 0) nSegments = -1;

  histos->fill1DHist(nSegments,"hSnSegments","Segments per Event",31,-0.5,30.5,"Segments");

}

// ==============================================
//
// look at hit Resolution
//
// ===============================================

void CSCValidation::doResolution(edm::Handle<CSCSegmentCollection> cscSegments, edm::ESHandle<CSCGeometry> cscGeom){


  for(CSCSegmentCollection::const_iterator dSiter=cscSegments->begin(); dSiter != cscSegments->end(); dSiter++) {

    CSCDetId id  = (CSCDetId)(*dSiter).cscDetId();

    //
    // try to get the CSC recHits that contribute to this segment.
    std::vector<CSCRecHit2D> theseRecHits = (*dSiter).specificRecHits();
    int nRH = (*dSiter).nRecHits();
    int jRH = 0;
    CLHEP::HepMatrix sp(6,1);
    CLHEP::HepMatrix se(6,1);
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
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

    // Fill histos
    histos->fill1DHistByType(residual,"hSResid","Fitted Position on Strip - Reconstructed for Layer 3",id,100,-0.5,0.5,"Resolution");
    histos->fillProfile(chamberSerial(id),residual,"hSResidProfile","Fitted Position on Strip - Reconstructed for Layer 3",601,-0.5,600.5,-0.5,0.5,"Resolution");
    if (detailedAnalysis){
      histos->fill1DHistByChamber(residual,"hSResid","Fitted Position on Strip - Reconstructed for Layer 3",id,100,-0.5,0.5,"DetailedResolution");
    }


  }

}

// ==============================================
//
// look at Standalone Muons
//
// ==============================================

void CSCValidation::doStandalone(Handle<reco::TrackCollection> saMuons){

  int nSAMuons = saMuons->size();
  histos->fill1DHist(nSAMuons,"trNSAMuons","N Standalone Muons per Event",6,-0.5,5.5,"STAMuons");

  for(reco::TrackCollection::const_iterator muon = saMuons->begin(); muon != saMuons->end(); ++ muon ) {
    float preco  = muon->p();
    float ptreco = muon->pt();
    int   n = muon->recHitsSize();
    float chi2 = muon->chi2();
    float normchi2 = muon->normalizedChi2();

    // loop over hits
    int nDTHits = 0;
    int nCSCHits = 0;
    int nCSCHitsp = 0;
    int nCSCHitsm = 0;
    int nRPCHits = 0;
    int nRPCHitsp = 0;
    int nRPCHitsm = 0;
    int np = 0;
    int nm = 0;
    vector<CSCDetId> staChambers;
    for (trackingRecHit_iterator hit = muon->recHitsBegin(); hit != muon->recHitsEnd(); ++hit ) {
      const DetId detId( (*hit)->geographicalId() );
      if (detId.det() == DetId::Muon) {
        if (detId.subdetId() == MuonSubdetId::RPC) {
          RPCDetId rpcId(detId.rawId());
          nRPCHits++;
          if (rpcId.region() == 1){ nRPCHitsp++; np++;}
          if (rpcId.region() == -1){ nRPCHitsm++; nm++;}
        }
        if (detId.subdetId() == MuonSubdetId::DT) {
          nDTHits++;
        }
        else if (detId.subdetId() == MuonSubdetId::CSC) {
          CSCDetId cscId(detId.rawId());
          staChambers.push_back(detId.rawId());
          nCSCHits++;
          if (cscId.endcap() == 1){ nCSCHitsp++; np++;}
          if (cscId.endcap() == 2){ nCSCHitsm++; nm++;}
        }
      }
    }

    // fill histograms
    histos->fill1DHist(n,"trN","N hits on a STA Muon Track",51,-0.5,50.5,"STAMuons");
    if (np != 0) histos->fill1DHist(np,"trNp","N hits on a STA Muon Track (plus endcap)",51,-0.5,50.5,"STAMuons");
    if (nm != 0) histos->fill1DHist(nm,"trNm","N hits on a STA Muon Track (minus endcap)",51,-0.5,50.5,"STAMuons");
    histos->fill1DHist(nDTHits,"trNDT","N DT hits on a STA Muon Track",51,-0.5,50.5,"STAMuons");
    histos->fill1DHist(nCSCHits,"trNCSC","N CSC hits on a STA Muon Track",51,-0.5,50.5,"STAMuons");
    if (nCSCHitsp != 0) histos->fill1DHist(nCSCHitsp,"trNCSCp","N CSC hits on a STA Muon Track (+ endcap)",51,-0.5,50.5,"STAMuons");
    if (nCSCHitsm != 0) histos->fill1DHist(nCSCHitsm,"trNCSCm","N CSC hits on a STA Muon Track (- endcap)",51,-0.5,50.5,"STAMuons");
    histos->fill1DHist(nRPCHits,"trNRPC","N RPC hits on a STA Muon Track",51,-0.5,50.5,"STAMuons");
    if (nRPCHitsp != 0) histos->fill1DHist(nRPCHitsp,"trNRPCp","N RPC hits on a STA Muon Track (+ endcap)",51,-0.5,50.5,"STAMuons");
    if (nRPCHitsm != 0) histos->fill1DHist(nRPCHitsm,"trNRPCm","N RPC hits on a STA Muon Track (- endcap)",51,-0.5,50.5,"STAMuons");
    histos->fill1DHist(preco,"trP","STA Muon Momentum",100,0,300,"STAMuons");
    histos->fill1DHist(ptreco,"trPT","STA Muon pT",100,0,40,"STAMuons");
    histos->fill1DHist(chi2,"trChi2","STA Muon Chi2",100,0,200,"STAMuons");
    histos->fill1DHist(normchi2,"trNormChi2","STA Muon Normalized Chi2",100,0,10,"STAMuons");

  }

}


//--------------------------------------------------------------
// Compute a serial number for the chamber.
// This is useful when filling histograms and working with arrays.
//--------------------------------------------------------------
int CSCValidation::chamberSerial( CSCDetId id ) {
  int st = id.station();
  int ri = id.ring();
  int ch = id.chamber();
  int ec = id.endcap();
  int kSerial = ch;
  if (st == 1 && ri == 1) kSerial = ch;
  if (st == 1 && ri == 2) kSerial = ch + 36;
  if (st == 1 && ri == 3) kSerial = ch + 72;
  if (st == 1 && ri == 4) kSerial = ch;
  if (st == 2 && ri == 1) kSerial = ch + 108;
  if (st == 2 && ri == 2) kSerial = ch + 126;
  if (st == 3 && ri == 1) kSerial = ch + 162;
  if (st == 3 && ri == 2) kSerial = ch + 180;
  if (st == 4 && ri == 1) kSerial = ch + 216;
  if (st == 4 && ri == 2) kSerial = ch + 234;  // one day...
  if (ec == 2) kSerial = kSerial + 300;
  return kSerial;
}



//-------------------------------------------------------------------------------------
// Fits a straight line to a set of 5 points with errors.  Functions assumes 6 points
// and removes hit in layer 3.  It then returns the expected position value in layer 3
// based on the fit.
//-------------------------------------------------------------------------------------
float CSCValidation::fitX(CLHEP::HepMatrix points, CLHEP::HepMatrix errors){

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

  //float chi = 0;
  //float chi2 = 0;

  // calculate chi2 (not currently used)
  //for (int i=1;i<7;i++){
  //  chi = (points(i,1) - intercept - slope*i)/(errors(i,1));
  //  chi2 = chi2 + chi*chi;
  //}

  return (intercept + slope*3);

}

//---------------------------------------------------------------------------------------
// Find the signal timing based on a weighted mean of the pulse.
// Function is meant to take the DetId and center strip number of a recHit and return
// the timing in units of time buckets (50ns)
//---------------------------------------------------------------------------------------

float CSCValidation::getTiming(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){

  float ADC[8];
  for (int i = 0; i < 8; i++){
    ADC[i] = 0;
  }
  float timing = 0;
  float peakADC = 0;
  int peakTime = 0;

  if (idRH.station() == 1 && idRH.ring() == 4){
    while(centerStrip> 16) centerStrip -= 16;
  }

  // Loop over strip digis responsible for this recHit and sum charge
  CSCStripDigiCollection::DigiRangeIterator gTstripIter;

  for (gTstripIter = stripdigis.begin(); gTstripIter != stripdigis.end(); gTstripIter++){
    CSCDetId id = (CSCDetId)(*gTstripIter).first;
    if (id == idRH){
      vector<CSCStripDigi>::const_iterator STiter = (*gTstripIter).second.first;
      vector<CSCStripDigi>::const_iterator lastS = (*gTstripIter).second.second;
      for ( ; STiter != lastS; ++STiter ) {
        int thisStrip = STiter->getStrip();
        if (thisStrip == (centerStrip)){
          float diff = 0;
          vector<int> myADCVals = STiter->getADCCounts();
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

  if (detailedAnalysis){
    float normADC;
    for (int i = 0; i < 8; i++){
      normADC = ADC[i]/ADC[peakTime];
      histos->fillProfileByChamber(i,normADC,"hSignalProfile","Normalized Signal Profile",idRH,8,-0.5,7.5,-0.1,1.1,"StripSignalProfile");
    }
    histos->fill1DHistByLayer(ADC[0],"hSigProPed","ADC in first time bin",idRH,400,-300,100,"StripSignalProfile");
    histos->fillProfile(chamberSerial(idRH),ADC[0],"hSigProPedProfile","ADC in first time bin",601,-0.5,600.5,-150,100,"StripSignalProfile");
  }


  timing = (ADC[2]*2 + ADC[3]*3 + ADC[4]*4 + ADC[5]*5 + ADC[6]*6)/(ADC[2] + ADC[3] + ADC[4] + ADC[5] + ADC[6]);

  return timing;

}

//----------------------------------------------------------------------------
// Calculate basic efficiencies for recHits and Segments
// Author: S. Stoynev
//----------------------------------------------------------------------------

void CSCValidation::doEfficiencies(edm::Handle<CSCWireDigiCollection> wires, edm::Handle<CSCStripDigiCollection> strips,
                                   edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments,
                                   edm::ESHandle<CSCGeometry> cscGeom){

  bool allWires[2][4][4][36][6];
  bool allStrips[2][4][4][36][6];
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
	    allWires[iE][iS][iR][iC][iL] = false;
	    allStrips[iE][iS][iR][iC][iL] = false;
            AllRecHits[iE][iS][iR][iC][iL] = false;
          }
        }
      }
    }
  }
  
  if (useDigis){
    // Wires
    for (CSCWireDigiCollection::DigiRangeIterator dWDiter=wires->begin(); dWDiter!=wires->end(); dWDiter++) {
      CSCDetId idrec = (CSCDetId)(*dWDiter).first;
      std::vector<CSCWireDigi>::const_iterator wireIter = (*dWDiter).second.first;
      std::vector<CSCWireDigi>::const_iterator lWire = (*dWDiter).second.second;
      for( ; wireIter != lWire; ++wireIter) {
        allWires[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;
        break;
      }
    }

    //---- STRIPS
    for (CSCStripDigiCollection::DigiRangeIterator dSDiter=strips->begin(); dSDiter!=strips->end(); dSDiter++) {
      CSCDetId idrec = (CSCDetId)(*dSDiter).first;
      std::vector<CSCStripDigi>::const_iterator stripIter = (*dSDiter).second.first;
      std::vector<CSCStripDigi>::const_iterator lStrip = (*dSDiter).second.second;
      for( ; stripIter != lStrip; ++stripIter) {
        std::vector<int> myADCVals = stripIter->getADCCounts();
        bool thisStripFired = false;
        float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
        float threshold = 13.3 ;
        float diff = 0.;
        for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
          diff = (float)myADCVals[iCount]-thisPedestal;
          if (diff > threshold) {
            thisStripFired = true;
   	    break;
          }
        }
        if(thisStripFired){
  	  allStrips[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;
          break;
        }
      }
    }
  }

  // Rechits
  for (CSCRecHit2DCollection::const_iterator recEffIt = recHits->begin(); recEffIt != recHits->end(); recEffIt++) {
    //CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    CSCDetId  idrec = (CSCDetId)(*recEffIt).cscDetId();
    AllRecHits[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;

  }

  std::vector <uint> seg_ME2(2,0) ;
  std::vector <uint> seg_ME3(2,0) ;
  std::vector < pair <CSCDetId, CSCSegment> > theSegments(4);
  // Segments
  for(CSCSegmentCollection::const_iterator segEffIt=cscSegments->begin(); segEffIt != cscSegments->end(); segEffIt++) {
    CSCDetId idseg  = (CSCDetId)(*segEffIt).cscDetId();
    //if(AllSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()]){
    //MultiSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()] = true;
    //}
    AllSegments[idseg.endcap() -1][idseg.station() -1][idseg.ring() -1][idseg.chamber() -1] = true;
    // "Intrinsic" efficiency measurement relies on "good" segment extrapolation - we need the pre-selection below
    // station 2 "good" segment will be used for testing efficiencies in ME1 and ME3
    // station 3 "good" segment will be used for testing efficiencies in ME2 and ME4
    if(2==idseg.station() || 3==idseg.station()){
      uint seg_tmp ; 
      if(2==idseg.station()){
	++seg_ME2[idseg.endcap() -1];
	seg_tmp = seg_ME2[idseg.endcap() -1];
      }
      else{
	++seg_ME3[idseg.endcap() -1];
	seg_tmp = seg_ME3[idseg.endcap() -1];
      }
      // is the segment good
      if(1== seg_tmp&& 6==(*segEffIt).nRecHits() && (*segEffIt).chi2()/(*segEffIt).degreesOfFreedom()<3.){
	pair <CSCDetId, CSCSegment> specSeg = make_pair( (CSCDetId)(*segEffIt).cscDetId(),*segEffIt);
	theSegments[2*(idseg.endcap()-1)+(idseg.station() -2)] = specSeg;
      }
    }
    /*
    if(2==idseg.station()){
	++seg_ME2[idseg.endcap() -1];
       if(1==seg_ME2[idseg.endcap() -1] && 6==(*segEffIt).nRecHits() && (*segEffIt).chi2()/(*segEffIt).degreesOfFreedom()<3.){
           pair <CSCDetId, CSCSegment> specSeg = make_pair( (CSCDetId)(*segEffIt).cscDetId(),*segEffIt);
           theSegments[2*(idseg.endcap()-1)+(idseg.station() -2)] = specSeg;
       }
    }
    else if(3==idseg.station()){
	++seg_ME3[idseg.endcap() -1];
	if(1==seg_ME3[idseg.endcap() -1] && 6==(*segEffIt).nRecHits() && (*segEffIt).chi2()/(*segEffIt).degreesOfFreedom()<3.){
         pair <CSCDetId, CSCSegment> specSeg = make_pair( (CSCDetId)(*segEffIt).cscDetId(),*segEffIt);
	 theSegments[2*(idseg.endcap()-1)+(idseg.station() -2)] = specSeg;
       }
    }
    */
    
  }
  // Simple efficiency calculations
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
          if (iS==0) bin = iR+1+(iE*10);
          else bin = (iS+1)*2 + (iR+1) + (iE*10);
          if(NumberOfLayers>1){
            //if(!(MultiSegments[iE][iS][iR][iC])){
            if(AllSegments[iE][iS][iR][iC]){
              //---- Efficient segment evenents
              hSSTE->AddBinContent(bin);
            }
            //---- All segment events (normalization)
            hSSTE->AddBinContent(20+bin);
            //}
          }
          if(AllSegments[iE][iS][iR][iC]){
            if(NumberOfLayers==6){
              //---- Efficient rechit events
              hRHSTE->AddBinContent(bin);;
            }
            //---- All rechit events (normalization)
            hRHSTE->AddBinContent(20+bin);;
          }
        }
      }
    }
  }

// pick a segment only if there are no others in the station
  std::vector < pair <CSCDetId, CSCSegment> * > theSeg;
  if(1==seg_ME2[0]) theSeg.push_back(&theSegments[0]);
  if(1==seg_ME3[0]) theSeg.push_back(&theSegments[1]);
  if(1==seg_ME2[1]) theSeg.push_back(&theSegments[2]);
  if(1==seg_ME3[1]) theSeg.push_back(&theSegments[3]);

  // Needed for plots
  // at the end the chamber types will be numbered as 1 to 18 
  // (ME-4/1, -ME3/2, -ME3/1, ..., +ME3/1, +ME3/2, ME+4/1 ) 
  std::map <std::string, float> chamberTypes;
  chamberTypes["ME1/a"] = 0.5;
  chamberTypes["ME1/b"] = 1.5;
  chamberTypes["ME1/2"] = 2.5;
  chamberTypes["ME1/3"] = 3.5;
  chamberTypes["ME2/1"] = 4.5;
  chamberTypes["ME2/2"] = 5.5;
  chamberTypes["ME3/1"] = 6.5;
  chamberTypes["ME3/2"] = 7.5;
  chamberTypes["ME4/1"] = 8.5;

  if(theSeg.size()){
    std::map <int , GlobalPoint> extrapolatedPoint;
    std::map <int , GlobalPoint>::iterator it;
    const std::vector<CSCChamber*> ChamberContainer = cscGeom->chambers();
    // Pick which chamber with which segment to test
    for(unsigned int nCh=0;nCh<ChamberContainer.size();nCh++){
      const CSCChamber *cscchamber = ChamberContainer[nCh];
      pair <CSCDetId, CSCSegment> * thisSegment = 0;
      for(uint iSeg =0;iSeg<theSeg.size();++iSeg ){
        if(cscchamber->id().endcap() == theSeg[iSeg]->first.endcap()){ 
          if(1==cscchamber->id().station() || 3==cscchamber->id().station() ){
	    if(2==theSeg[iSeg]->first.station()){
	      thisSegment = theSeg[iSeg];
	    }
	  }
	  else if (2==cscchamber->id().station() || 4==cscchamber->id().station()){
	    if(3==theSeg[iSeg]->first.station()){
	      thisSegment = theSeg[iSeg];
	    }
	  }
	}
      }
      // this chamber is to be tested with thisSegment
      if(thisSegment){
	CSCSegment * seg = &(thisSegment->second);
	const CSCChamber *segChamber = cscGeom->chamber(thisSegment->first);
	LocalPoint localCenter(0.,0.,0);
	GlobalPoint cscchamberCenter =  cscchamber->toGlobal(localCenter);
	// try to save some time (extrapolate a segment to a certain position only once)
	it = extrapolatedPoint.find(int(cscchamberCenter.z()));
	if(it==extrapolatedPoint.end()){
	  GlobalPoint segPos = segChamber->toGlobal(seg->localPosition());
	  GlobalVector segDir = segChamber->toGlobal(seg->localDirection());
	  double paramaterLine = lineParametrization(segPos.z(),cscchamberCenter.z() , segDir.z());
	  double xExtrapolated = extrapolate1D(segPos.x(),segDir.x(), paramaterLine);
	  double yExtrapolated = extrapolate1D(segPos.y(),segDir.y(), paramaterLine);
	  GlobalPoint globP (xExtrapolated, yExtrapolated, cscchamberCenter.z());
	  extrapolatedPoint[int(cscchamberCenter.z())] = globP;
	}
	// Where does the extrapolated point lie in the (tested) chamber local frame? Here: 
	LocalPoint extrapolatedPointLocal = cscchamber->toLocal(extrapolatedPoint[int(cscchamberCenter.z())]);
	const CSCLayer *layer_p = cscchamber->layer(1);//layer 1
	const CSCLayerGeometry *layerGeom = layer_p->geometry ();
	const std::vector<float> layerBounds = layerGeom->parameters ();
	float shiftFromEdge = 15.;//cm
	float shiftFromDeadZone = 10.;
	// is the extrapolated point within a sensitive region
	bool pass = withinSensitiveRegion(extrapolatedPointLocal, layerBounds, 
					  cscchamber->id().station(), cscchamber->id().ring(), 
					  shiftFromEdge, shiftFromDeadZone);
	if(pass){// the extrapolation point of the segment lies within sensitive region of that chamber
	  // how many rechit layers are there in the chamber?
	  // 0 - maybe the muon died or is deflected at large angle? do not use that case
	  // 1 - could be noise...
	  // 2 or more - this is promissing; this is our definition of a reliable signal; use it below
	  // is other definition better? 
	  int nRHLayers = 0;
	  for(int iL =0;iL<6;++iL){
	    if(AllRecHits[cscchamber->id().endcap()-1]
	       [cscchamber->id().station()-1]
	       [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
	      ++nRHLayers;
	    }
	  }
	  //std::cout<<" nRHLayers = "<<nRHLayers<<std::endl;
	  float verticalScale = chamberTypes[cscchamber->specs()->chamberTypeName()];
	  if(cscchamberCenter.z()<0){
	    verticalScale = - verticalScale;
	  } 
	  verticalScale +=9.5;
	  hSensitiveAreaEvt->Fill(float(cscchamber->id().chamber()),verticalScale);
	  if(nRHLayers>1){// this chamber contains a reliable signal
	    //chamberTypes[cscchamber->specs()->chamberTypeName()];
	    // "intrinsic" efficiencies
	    //std::cout<<" verticalScale = "<<verticalScale<<" chType = "<<cscchamber->specs()->chamberTypeName()<<std::endl;
	    // this is the denominator forr all efficiencies
	    hEffDenominator->Fill(float(cscchamber->id().chamber()),verticalScale);
	    // Segment efficiency
	    if(AllSegments[cscchamber->id().endcap()-1]
	       [cscchamber->id().station()-1]
	       [cscchamber->id().ring()-1][cscchamber->id().chamber()-1]){
	      hSSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale));
	    }
	  
	    for(int iL =0;iL<6;++iL){
	      float weight = 1./6.;
	      // one shold account for the weight in the efficiency...
	      // Rechit efficiency
	      if(AllRecHits[cscchamber->id().endcap()-1]
		 [cscchamber->id().station()-1]
		 [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
		hRHSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale),weight);
	      }
              if (useDigis){
	        // Wire efficiency
	        if(allWires[cscchamber->id().endcap()-1]
	  	  [cscchamber->id().station()-1]
		  [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
		  // one shold account for the weight in the efficiency...
	 	  hWireSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale),weight);
	        }
	        // Strip efficiency
	        if(allStrips[cscchamber->id().endcap()-1]
		  [cscchamber->id().station()-1]
		  [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
		  // one shold account for the weight in the efficiency...
		  hStripSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale),weight);
	        }
              }
	    }
	  }
	}
      }
    }
  }
  //
  
  
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
  for (int i=0;i<20;i++){
    Num = readHisto->GetBinContent(i+1);
    Den = readHisto->GetBinContent(i+21);
    getEfficiency(Num, Den, eff);
    Efficiency[i] = eff[0];
    EffError[i] = eff[1];
    writeHisto->SetBinContent(i+1, Efficiency[i]);
    writeHisto->SetBinError(i+1, EffError[i]);
  }
}

bool CSCValidation::withinSensitiveRegion(LocalPoint localPos, const std::vector<float> layerBounds, int station, int ring, float shiftFromEdge, float shiftFromDeadZone){
//---- check if it is in a good local region (sensitive area - geometrical and HV boundaries excluded) 
  bool pass = false;

  float y_center = 0.;
  double yUp = layerBounds[3] + y_center;
  double yDown = - layerBounds[3] + y_center;
  double xBound1Shifted = layerBounds[0] - shiftFromEdge;//
  double xBound2Shifted = layerBounds[1] - shiftFromEdge;//
  double lineSlope = (yUp - yDown)/(xBound2Shifted-xBound1Shifted);
  double lineConst = yUp - lineSlope*xBound2Shifted;
  double yBorder =  lineSlope*abs(localPos.x()) + lineConst;
      
  //bool withinChamberOnly = false;// false = "good region"; true - boundaries only
  std::vector <float> deadZoneCenter(6);
  float cutZone = shiftFromDeadZone;//cm
  //---- hardcoded... not good
  if(station>1 && station<5){
    if(2==ring){
      deadZoneCenter[0]= -162.48 ;
      deadZoneCenter[1] = -81.8744;
      deadZoneCenter[2] = -21.18165;
      deadZoneCenter[3] = 39.51105;
      deadZoneCenter[4] = 100.2939;
      deadZoneCenter[5] = 160.58;
      
      if(localPos.y() >yBorder &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone) ||
	  (localPos.y()> deadZoneCenter[3] + cutZone && localPos.y()< deadZoneCenter[4] - cutZone) ||
	  (localPos.y()> deadZoneCenter[4] + cutZone && localPos.y()< deadZoneCenter[5] - cutZone))){
	pass = true;
      }
    }
    else if(1==ring){
      if(2==station){
	deadZoneCenter[0]= -95.80 ;
	deadZoneCenter[1] = -27.47;
	deadZoneCenter[2] = 33.67;
	deadZoneCenter[3] = 90.85;
        }
      else if(3==station){
	deadZoneCenter[0]= -89.305 ;
	deadZoneCenter[1] = -39.705;
	deadZoneCenter[2] = 20.195;
	deadZoneCenter[3] = 77.395;
      }
      else if(4==station){
	deadZoneCenter[0]= -75.645;
	deadZoneCenter[1] = -26.055;
	deadZoneCenter[2] = 23.855;
	deadZoneCenter[3] = 70.575;
      }
      if(localPos.y() >yBorder &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone))){
	pass = true;
      }
    }
  }
  else if(1==station){
    if(3==ring){
      deadZoneCenter[0]= -83.155 ;
      deadZoneCenter[1] = -22.7401;
      deadZoneCenter[2] = 27.86665;
      deadZoneCenter[3] = 81.005;
      if(localPos.y() > yBorder &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone))){
	pass = true;
      }
    }
    else if(2==ring){
      deadZoneCenter[0]= -86.285 ;
      deadZoneCenter[1] = -32.88305;
      deadZoneCenter[2] = 32.867423;
      deadZoneCenter[3] = 88.205;
      if(localPos.y() > (yBorder) &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone))){
	pass = true;
      }
    }
    else{
      deadZoneCenter[0]= -81.0;
      deadZoneCenter[1] = 81.0;
      if(localPos.y() > (yBorder) &&
	 (localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone )){
	pass = true;
      }
    }
  }
  return pass;
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

//---------------------------------------------------------------------------------------
// Look at non-associated recHits
// Author: P. Jindal
//---------------------------------------------------------------------------------------

void CSCValidation::doNoiseHits(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments,
                                edm::ESHandle<CSCGeometry> cscGeom,  edm::Handle<CSCStripDigiCollection> strips){

  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {

    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();

    //Store the Rechits into a Map
    AllRechits.insert(pair<CSCDetId , CSCRecHit2D>(idrec,*recIt));

    // Find the strip containing this hit
    CSCRecHit2D::ChannelContainer hitstrips = (*recIt).channels();
    int nStrips     =  hitstrips.size();
    //std::cout << " no of strips in Rec Hit " << nStrips << std::endl;
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];

    float  rHsignal = getthisSignal(*strips, idrec, centerStrip);
    histos->fill1DHist(rHsignal,"hrHSignal", "Signal in the 4th time bin for centre strip",1100,-99,1000,"recHits");

  }

  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {

    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      LocalPoint lpRH = (*iRH).localPosition();
      float xrec = lpRH.x();
      float yrec = lpRH.y();
      float zrec = lpRH.z();
      bool RHalreadyinMap = false;
      //Store the rechits associated with segments into a Map
      multimap<CSCDetId , CSCRecHit2D>::iterator segRHit;
      segRHit = SegRechits.find(idRH);
      if (segRHit != SegRechits.end()){
	for( ; segRHit != SegRechits.upper_bound(idRH); ++segRHit){
	  //for( segRHit = SegRechits.begin(); segRHit != SegRechits.end() ;++segRHit){
	  LocalPoint lposRH = (segRHit->second).localPosition();
	  float xpos = lposRH.x();
	  float ypos = lposRH.y();
	  float zpos = lposRH.z();
	  if ( xrec == xpos && yrec == ypos && zrec == zpos){
	  RHalreadyinMap = true;
	  //std::cout << " Already exists " <<std ::endl;
	  break;}
	}
      }
      if(!RHalreadyinMap){ SegRechits.insert(pair<CSCDetId , CSCRecHit2D>(idRH,*iRH));}
    }
  }

  findNonAssociatedRecHits(cscGeom,strips);

}

//---------------------------------------------------------------------------------------
// Given  the list of all rechits and the rechits on a segment finds the rechits 
// not associated to a segment and stores in a list
//
//---------------------------------------------------------------------------------------

void CSCValidation::findNonAssociatedRecHits(edm::ESHandle<CSCGeometry> cscGeom,  edm::Handle<CSCStripDigiCollection> strips){
 
  for(multimap<CSCDetId , CSCRecHit2D>::iterator allRHiter =  AllRechits.begin();allRHiter != AllRechits.end(); ++allRHiter){
	CSCDetId idRH = allRHiter->first;
    LocalPoint lpRH = (allRHiter->second).localPosition();
    float xrec = lpRH.x();
    float yrec = lpRH.y();
    float zrec = lpRH.z();
    
    bool foundmatch = false;
    multimap<CSCDetId , CSCRecHit2D>::iterator segRHit;
    segRHit = SegRechits.find(idRH);
    if (segRHit != SegRechits.end()){
		for( ; segRHit != SegRechits.upper_bound(idRH); ++segRHit){
			
			LocalPoint lposRH = (segRHit->second).localPosition();
			float xpos = lposRH.x();
			float ypos = lposRH.y();
			float zpos = lposRH.z();

			if ( xrec == xpos && yrec == ypos && zrec == zpos){
				foundmatch = true;}
	  
			float d      = 0.;
			float dclose =1000.;

			if ( !foundmatch) {
				
				d = sqrt(pow(xrec-xpos,2)+pow(yrec-ypos,2)+pow(zrec-zpos,2));
				if (d < dclose) {
					dclose = d;
					if( distRHmap.find((allRHiter->second)) ==  distRHmap.end() ) { // entry for rechit does not yet exist, create one
						distRHmap.insert(make_pair(allRHiter->second,dclose) );
					}
					else {
						// we already have an entry for the detid.
						distRHmap.erase(allRHiter->second);
						distRHmap.insert(make_pair(allRHiter->second,dclose)); // fill rechits for the segment with the given detid
					}
				}
			} 	    
		}
    }
    if(!foundmatch){NonAssociatedRechits.insert(pair<CSCDetId , CSCRecHit2D>(idRH,allRHiter->second));}
  }

  for(map<CSCRecHit2D,float,ltrh>::iterator iter =  distRHmap.begin();iter != distRHmap.end(); ++iter){
    histos->fill1DHist(iter->second,"hdistRH","Distance of Non Associated RecHit from closest Segment RecHit",500,0.,100.,"NonAssociatedRechits");
  }

  for(multimap<CSCDetId , CSCRecHit2D>::iterator iter =  NonAssociatedRechits.begin();iter != NonAssociatedRechits.end(); ++iter){
    CSCDetId idrec = iter->first;
    int kEndcap  = idrec.endcap();
    int cEndcap  = idrec.endcap();
    if (kEndcap == 2)cEndcap = -1;
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    // Store rechit as a Local Point:
    LocalPoint rhitlocal = (iter->second).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();

    // Find the strip containing this hit
    CSCRecHit2D::ChannelContainer hitstrips = (iter->second).channels();
    int nStrips     =  hitstrips.size();
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];


    // Find the charge associated with this hit

    CSCRecHit2D::ADCContainer adcs = (iter->second).adcs();
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
    //float rHtime = (iter->second).tpeak();
    float rHtime = getTiming(*strips, idrec, centerStrip);

    // Get the width of this hit
    int rHwidth = getWidth(*strips, idrec, centerStrip);


    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx   =  rhitglobal.x();
    float grecy   =  rhitglobal.y();



   // Simple occupancy variables
    int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;

    //Fill the non-associated rechits parameters in histogram
    histos->fill1DHist(kCodeBroad,"hNARHCodeBroad","broad scope code for recHits",33,-16.5,16.5,"NonAssociatedRechits");
    if (kStation == 1) histos->fill1DHist(kCodeNarrow,"hNARHCodeNarrow1","narrow scope recHit code station 1",801,-400.5,400.5,"NonAssociatedRechits");
    if (kStation == 2) histos->fill1DHist(kCodeNarrow,"hNARHCodeNarrow2","narrow scope recHit code station 2",801,-400.5,400.5,"NonAssociatedRechits");
    if (kStation == 3) histos->fill1DHist(kCodeNarrow,"hNARHCodeNarrow3","narrow scope recHit code station 3",801,-400.5,400.5,"NonAssociatedRechits");
    if (kStation == 4) histos->fill1DHist(kCodeNarrow,"hNARHCodeNarrow4","narrow scope recHit code station 4",801,-400.5,400.5,"NonAssociatedRechits");
    histos->fill1DHistByType(kLayer,"hNARHLayer","RecHits per Layer",idrec,8,-0.5,7.5,"NonAssociatedRechits");
    histos->fill1DHistByType(xreco,"hNARHX","Local X of recHit",idrec,160,-80.,80.,"NonAssociatedRechits");
    histos->fill1DHistByType(yreco,"hNARHY","Local Y of recHit",idrec,60,-180.,180.,"NonAssociatedRechits");
    if (kStation == 1 && (kRing == 1 || kRing == 4)) histos->fill1DHistByType(rHSumQ,"hNARHSumQ","Sum 3x3 recHit Charge",idrec,250,0,4000,"NonAssociatedRechits");
    else histos->fill1DHistByType(rHSumQ,"hNARHSumQ","Sum 3x3 recHit Charge",idrec,250,0,2000,"NonAssociatedRechits");
    histos->fill1DHistByType(rHratioQ,"hNARHRatioQ","Ratio (Ql+Qr)/Qt)",idrec,120,-0.1,1.1,"NonAssociatedRechits");
    histos->fill1DHistByType(rHtime,"hNARHTiming","recHit Timing",idrec,100,0,10,"NonAssociatedRechits");
    histos->fill2DHistByStation(grecx,grecy,"hNARHGlobal","recHit Global Position",idrec,400,-800.,800.,400,-800.,800.,"NonAssociatedRechits");
    histos->fill1DHistByType(rHwidth,"hNARHwidth","width for Non associated recHit",idrec,21,-0.5,20.5,"NonAssociatedRechits");
    
  }

   for(multimap<CSCDetId , CSCRecHit2D>::iterator iter =  SegRechits.begin();iter != SegRechits.end(); ++iter){
	   CSCDetId idrec = iter->first;
	   int kEndcap  = idrec.endcap();
	   int cEndcap  = idrec.endcap();
	   if (kEndcap == 2)cEndcap = -1;
	   int kRing    = idrec.ring();
	   int kStation = idrec.station();
	   int kChamber = idrec.chamber();
	   int kLayer   = idrec.layer();

	   // Store rechit as a Local Point:
	   LocalPoint rhitlocal = (iter->second).localPosition();  
	   float xreco = rhitlocal.x();
	   float yreco = rhitlocal.y();

	   // Find the strip containing this hit
	   CSCRecHit2D::ChannelContainer hitstrips = (iter->second).channels();
	   int nStrips     =  hitstrips.size();
	   int centerid    =  nStrips/2 + 1;
	   int centerStrip =  hitstrips[centerid - 1];


	   // Find the charge associated with this hit
	   
	   CSCRecHit2D::ADCContainer adcs = (iter->second).adcs();
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
	   //float rHtime = (iter->second).tpeak();
	   float rHtime = getTiming(*strips, idrec, centerStrip);

	   // Get the width of this hit
	   int rHwidth = getWidth(*strips, idrec, centerStrip);


	   // Get pointer to the layer:
	   const CSCLayer* csclayer = cscGeom->layer( idrec );
	   
	   // Transform hit position from local chamber geometry to global CMS geom
	   GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
	   float grecx   =  rhitglobal.x();
	   float grecy   =  rhitglobal.y();

	   // Simple occupancy variables
	   int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
	   int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;

	   //Fill the non-associated rechits global position in histogram
           histos->fill1DHist(kCodeBroad,"hSegRHCodeBroad","broad scope code for recHits",33,-16.5,16.5,"AssociatedRechits");
           if (kStation == 1) histos->fill1DHist(kCodeNarrow,"hSegRHCodeNarrow1","narrow scope recHit code station 1",801,-400.5,400.5,"AssociatedRechits");
           if (kStation == 2) histos->fill1DHist(kCodeNarrow,"hSegRHCodeNarrow2","narrow scope recHit code station 2",801,-400.5,400.5,"AssociatedRechits");
           if (kStation == 3) histos->fill1DHist(kCodeNarrow,"hSegRHCodeNarrow3","narrow scope recHit code station 3",801,-400.5,400.5,"AssociatedRechits");
           if (kStation == 4) histos->fill1DHist(kCodeNarrow,"hSegRHCodeNarrow4","narrow scope recHit code station 4",801,-400.5,400.5,"AssociatedRechits");
           histos->fill1DHistByType(kLayer,"hSegRHLayer","RecHits per Layer",idrec,8,-0.5,7.5,"AssociatedRechits");
           histos->fill1DHistByType(xreco,"hSegRHX","Local X of recHit",idrec,160,-80.,80.,"AssociatedRechits");
           histos->fill1DHistByType(yreco,"hSegRHY","Local Y of recHit",idrec,60,-180.,180.,"AssociatedRechits");
           if (kStation == 1 && (kRing == 1 || kRing == 4)) histos->fill1DHistByType(rHSumQ,"hSegRHSumQ","Sum 3x3 recHit Charge",idrec,250,0,4000,"AssociatedRechits");
           else histos->fill1DHistByType(rHSumQ,"hSegRHSumQ","Sum 3x3 recHit Charge",idrec,250,0,2000,"AssociatedRechits");
           histos->fill1DHistByType(rHratioQ,"hSegRHRatioQ","Ratio (Ql+Qr)/Qt)",idrec,120,-0.1,1.1,"AssociatedRechits");
           histos->fill1DHistByType(rHtime,"hSegRHTiming","recHit Timing",idrec,100,0,10,"AssociatedRechits");
           histos->fill2DHistByStation(grecx,grecy,"hSegRHGlobal","recHit Global Position",idrec,400,-800.,800.,400,-800.,800.,"AssociatedRechits");
           histos->fill1DHistByType(rHwidth,"hSegRHwidth","width for Non associated recHit",idrec,21,-0.5,20.5,"AssociatedRechits");
	   
   }

   distRHmap.clear();
   AllRechits.clear();
   SegRechits.clear();
   NonAssociatedRechits.clear();
}



float CSCValidation::getthisSignal(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){
	// Loop over strip digis responsible for this recHit
	CSCStripDigiCollection::DigiRangeIterator sIt;
	float thisADC = 0.;
	bool foundRHid = false;
	// std::cout<<"iD   S/R/C/L = "<<idRH<<"    "<<idRH.station()<<"/"<<idRH.ring()<<"/"<<idRH.chamber()<<"/"<<idRH.layer()<<std::endl;
	for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
		CSCDetId id = (CSCDetId)(*sIt).first;
		//std::cout<<"STRIPS: id    S/R/C/L = "<<id<<"     "<<id.station()<<"/"<<id.ring()<<"/"<<id.chamber()<<"/"<<id.layer()<<std::endl;
		if (id == idRH){
			foundRHid = true;
			vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
			vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
			//if(digiItr == last ) {std::cout << " Attention1 :: Size of digi collection is zero " << std::endl;}
			int St = idRH.station();
			int Rg    = idRH.ring();
			if (St == 1 && Rg == 4){
				while(centerStrip> 16) centerStrip -= 16;
			}
			for ( ; digiItr != last; ++digiItr ) {
				int thisStrip = digiItr->getStrip();
				//std::cout<<" thisStrip = "<<thisStrip<<" centerStrip = "<<centerStrip<<std::endl;
				std::vector<int> myADCVals = digiItr->getADCCounts();
				float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
				float Signal = (float) myADCVals[3];
				if (thisStrip == (centerStrip)){
					thisADC = Signal-thisPedestal;
					//if(thisADC >= 0. && thisADC <2.) {std::cout << " Attention2 :: The Signal is equal to the pedestal " << std::endl;
					//}
					//if(thisADC < 0.) {std::cout << " Attention3 :: The Signal is less than the pedestal " << std::endl;
					//}
				}
				if (thisStrip == (centerStrip+1)){
					std::vector<int> myADCVals = digiItr->getADCCounts();
				}
				if (thisStrip == (centerStrip-1)){
					std::vector<int> myADCVals = digiItr->getADCCounts();
				}
			}
		}
	}
	//if(!foundRHid){std::cout << " Attention4 :: Did not find a matching RH id in the Strip Digi collection " << std::endl;}
	return thisADC;
}

//---------------------------------------------------------------------------------------
//
// Function is meant to take the DetId and center strip number of a recHit and return
// the width in terms of strips
//---------------------------------------------------------------------------------------

int CSCValidation::getWidth(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){

  int width = 1;
  int widthpos = 0;
  int widthneg = 0;

  // Loop over strip digis responsible for this recHit and sum charge
  CSCStripDigiCollection::DigiRangeIterator sIt;

  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
	  CSCDetId id = (CSCDetId)(*sIt).first;
	  if (id == idRH){
		  vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
		  vector<CSCStripDigi>::const_iterator first = (*sIt).second.first;
		  vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
		  vector<CSCStripDigi>::const_iterator it = (*sIt).second.first;
		  vector<CSCStripDigi>::const_iterator itr = (*sIt).second.first;
		  //std::cout << " IDRH " << id <<std::endl;
		  int St = idRH.station();
		  int Rg    = idRH.ring();
		  if (St == 1 && Rg == 4){
			  while(centerStrip> 16) centerStrip -= 16;
		  }
		  for ( ; digiItr != last; ++digiItr ) {
			  int thisStrip = digiItr->getStrip();
			  if (thisStrip == (centerStrip)){
				  it = digiItr;
				  for( ; it != last; ++it ) {
					  int strip = it->getStrip();
					  std::vector<int> myADCVals = it->getADCCounts();
					  float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
					  if(((float)myADCVals[3]-thisPedestal) < 6 || widthpos == 10 || it==last){break;}
					   if(strip != centerStrip){ widthpos += 1;
					   }
				  }
				  itr = digiItr;
				  for( ; itr != first; --itr) {
					  int strip = itr->getStrip();
					  std::vector<int> myADCVals = itr->getADCCounts();
					  float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
					  if(((float)myADCVals[3]-thisPedestal) < 6 || widthneg == 10 || itr==first){break;}	 
					  if(strip != centerStrip) {widthneg += 1 ; 
					  }
				  }
			  }
		  }
	  }
  }
  //std::cout << "Widthneg - " <<  widthneg << "Widthpos + " <<  widthpos << std::endl;
  width =  width + widthneg +  widthpos ;
  //std::cout << "Width " <<  width << std::endl;
  return width;
}


//---------------------------------------------------------------------------
// Module for looking at gas gains
// Author N. Terentiev
//---------------------------------------------------------------------------

void CSCValidation::doGasGain(const CSCWireDigiCollection& wirecltn, 
                              const CSCStripDigiCollection&   strpcltn,
                              const CSCRecHit2DCollection& rechitcltn) {
     float y;
     int channel=0,mult,wire,layer,idlayer,idchamber,ring;
     int wire_strip_rechit_present;
     string name,title,endcapstr;
     ostringstream ss;
     CSCIndexer indexer;
     std::map<int,int>::iterator intIt;

     m_single_wire_layer.clear();

  if(firstEvent) {

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

     if(wire_strip_rechit_present==7) {

//       std::cout<<"Event "<<nEventsAnalyzed<<std::endl;
//       std::cout<<std::endl;

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
             mult++;
          }     // end of digis loop in layer

          // select layers with single wire hit
          if(mult==1) {
            if(m_single_wire_layer.find(idlayer) == m_single_wire_layer.end())
              m_single_wire_layer[idlayer]=wire;
          } // end of if(mult==1)
       }   // end of cycle on detUnit

       // Looping thru rechit collection
       CSCRecHit2DCollection::const_iterator recIt;
       CSCRecHit2D::ADCContainer m_adc;
       CSCRecHit2D::ChannelContainer m_strip;
       for(recIt = rechitcltn.begin(); recIt != rechitcltn.end(); ++recIt) {
          CSCDetId id = (CSCDetId)(*recIt).cscDetId();
          idlayer=indexer.dbIndex(id, channel);
          idchamber=idlayer/10;
          layer=id.layer();
          // select layer with single wire rechit
          if(m_single_wire_layer.find(idlayer) != m_single_wire_layer.end()) {

            // getting strips comprising rechit
            m_strip=(CSCRecHit2D::ChannelContainer)(*recIt).channels(); 
            if(m_strip.size()==3)  {        
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
                float adc_3_3_sum=0.0;
                for(int j=binmx-1;j<=binmx+1;j++) {
                   adc_3_3_sum=adc_3_3_sum+adc_left[j]
                                          +adc_center[j]
                                          +adc_right[j];
                }

               if(adc_3_3_sum > 0.0 &&  adc_3_3_sum < 2000.0) {

                 // temporary fix for ME1/1a to avoid triple entries
                 int flag=0;
                 if(id.station()==1 && id.ring()==4 &&  m_strip[1]>16)  flag=1;
                 // end of temporary fix
                 if(flag==0) {

                 wire= m_single_wire_layer[idlayer];
                 int chambertype=id.iChamberType(id.station(),id.ring());
                 int hvsgmtnmb=m_wire_hvsegm[chambertype][wire];
                 int nmbofhvsegm=nmbhvsegm[chambertype-1];
                 int location= (layer-1)*nmbofhvsegm+hvsgmtnmb;
                 float x=location;
                
                 ss<<"gas_gain_rechit_adc_3_3_sum_location_ME_"<<idchamber;
                 name=ss.str(); ss.str("");
                 if(id.endcap()==1) endcapstr = "+";
                 ring=id.ring();
                 if(id.station()==1 && id.ring()==4) ring=1;
                 if(id.endcap()==2) endcapstr = "-"; 
                 ss<<"Gas Gain Rechit ADC3X3 Sum ME"<<endcapstr<<
                   id.station()<<"/"<<ring<<"/"<<id.chamber();
                 title=ss.str(); ss.str("");
                 x=location;
                 y=adc_3_3_sum;
                 histos->fill2DHist(x,y,name.c_str(),title.c_str(),30,1.0,31.0,50,0.0,2000.0,"GasGain");

                 /*
                   std::cout<<idchamber<<"   "<<id.station()<<" "<<id.ring()<<" "
                   <<id.chamber()<<"    "<<layer<<" "<< wire<<" "<<m_strip[1]<<" "<<
                   chambertype<<" "<< hvsgmtnmb<<" "<< nmbofhvsegm<<" "<< 
                   location<<"   "<<adc_3_3_sum<<std::endl;
                 */
               } // end of if flag==0
               } // end if(adcsum>0.0 && adcsum<2000.0)
            } // end of if if(m_strip.size()==3
          } // end of if single wire
        } // end of looping thru rechit collection
     }   // end of if wire and strip and rechit present 
}

//---------------------------------------------------------------------------
// Module for looking at AFEB Timing
// Author N. Terentiev
//---------------------------------------------------------------------------

void CSCValidation::doAFEBTiming(const CSCWireDigiCollection& wirecltn) {
     ostringstream ss;
     string name,title,endcapstr;
     float x,y;
     int wire,wiretbin,nmbwiretbin,layer,afeb,idlayer,idchamber;
     int channel=0; // for  CSCIndexer::dbIndex(id, channel); irrelevant here
     CSCIndexer indexer;

     if(wirecltn.begin() != wirecltn.end())  {

       //std::cout<<std::endl;
       //std::cout<<"Event "<<nEventsAnalyzed<<std::endl;
       //std::cout<<std::endl;

       // cycle on wire collection for all CSC
       CSCWireDigiCollection::DigiRangeIterator wiredetUnitIt;
       for(wiredetUnitIt=wirecltn.begin();wiredetUnitIt!=wirecltn.end();
          ++wiredetUnitIt) {
          const CSCDetId id = (*wiredetUnitIt).first;
          idlayer=indexer.dbIndex(id, channel);
          idchamber=idlayer/10;
          layer=id.layer();

          if (id.endcap() == 1) endcapstr = "+";
          if (id.endcap() == 2) endcapstr = "-";

          // looping in the layer of given CSC
 
          const CSCWireDigiCollection::Range& range = (*wiredetUnitIt).second;
          for(CSCWireDigiCollection::const_iterator digiIt =
             range.first; digiIt!=range.second; ++digiIt){
             wire=(*digiIt).getWireGroup();
             wiretbin=(*digiIt).getTimeBin();
             nmbwiretbin=(*digiIt).getTimeBinsOn().size();
             afeb=3*((wire-1)/8)+(layer+1)/2;
             
             // Anode wire group time bin vs afeb for each CSC
             x=afeb;
             y=wiretbin;
             ss<<"afeb_time_bin_vs_afeb_occupancy_ME_"<<idchamber;
             name=ss.str(); ss.str("");
             ss<<"Time Bin vs AFEB Occupancy ME"<<endcapstr<<id.station()<<"/"<<id.ring()<<"/"<< id.chamber();
             title=ss.str(); ss.str("");
             histos->fill2DHist(x,y,name.c_str(),title.c_str(),42,1.,43.,16,0.,16.,"AFEBTiming");

             // Number of anode wire group time bin vs afeb for each CSC
             x=afeb;
             y=nmbwiretbin;
             ss<<"nmb_afeb_time_bins_vs_afeb_ME_"<<idchamber;
             name=ss.str(); ss.str("");
             ss<<"Number of Time Bins vs AFEB ME"<<endcapstr<<id.station()<<"/"<<id.ring()<<"/"<< id.chamber();
             title=ss.str(); 
             ss.str("");
             histos->fill2DHist(x,y,name.c_str(),title.c_str(),42,1.,43.,16,0.,16.,"AFEBTiming");
             
          }     // end of digis loop in layer
       } // end of wire collection loop
     } // end of      if(wirecltn.begin() != wirecltn.end())
}

//---------------------------------------------------------------------------
// Module for looking at Comparitor Timing
// Author N. Terentiev
//---------------------------------------------------------------------------

void CSCValidation::doCompTiming(const CSCComparatorDigiCollection& compars) {

     ostringstream ss;      string name,title,endcap;
     float x,y;
     int strip,tbin,layer,cfeb,idlayer,idchamber,idum;
     int channel=0; // for  CSCIndexer::dbIndex(id, channel); irrelevant here
     CSCIndexer indexer;
                                                                                
     if(compars.begin() != compars.end())  {
                                                                                
       //std::cout<<std::endl;
       //std::cout<<"Event "<<nEventsAnalyzed<<std::endl;
       //std::cout<<std::endl;
                                                                                
       // cycle on comparators collection for all CSC
       CSCComparatorDigiCollection::DigiRangeIterator compdetUnitIt;
       for(compdetUnitIt=compars.begin();compdetUnitIt!=compars.end();
          ++compdetUnitIt) {
          const CSCDetId id = (*compdetUnitIt).first;
          idlayer=indexer.dbIndex(id, channel); // channel irrelevant here
          idchamber=idlayer/10;
          layer=id.layer();
                                                                                
          if (id.endcap() == 1) endcap = "+";
          if (id.endcap() == 2) endcap = "-";
          // looping in the layer of given CSC
          const CSCComparatorDigiCollection::Range& range = 
          (*compdetUnitIt).second;
          for(CSCComparatorDigiCollection::const_iterator digiIt =
             range.first; digiIt!=range.second; ++digiIt){
             strip=(*digiIt).getStrip();
          /*
          if(id.station()==1 && (id.ring()==1 || id.ring()==4))
             std::cout<<idchamber<<" "<<id.station()<<" "<<id.ring()<<" "
                      <<strip <<std::endl;  
          */
             idum=indexer.dbIndex(id, strip); // strips 1-16 of ME1/1a 
                                              // become strips 65-80 of ME1/1 
             tbin=(*digiIt).getTimeBin();
             cfeb=(strip-1)/16+1;
                                                                                
             // time bin vs cfeb for each CSC

             x=cfeb;
             y=tbin;
             ss<<"comp_time_bin_vs_cfeb_occupancy_ME_"<<idchamber;
             name=ss.str(); ss.str("");
             ss<<"Comparator Time Bin vs CFEB Occupancy ME"<<endcap<<
                 id.station()<<"/"<< id.ring()<<"/"<< id.chamber();             
             title=ss.str(); ss.str("");
             histos->fill2DHist(x,y,name.c_str(),title.c_str(),5,1.,6.,16,0.,16.,"CompTiming");

         }     // end of digis loop in layer
       } // end of collection loop
     } // end of      if(compars.begin() !=compars.end())
}

//---------------------------------------------------------------------------
// Module for looking at Strip Timing
// Author N. Terentiev
//---------------------------------------------------------------------------

void CSCValidation::doADCTiming(const CSCStripDigiCollection&   strpcltn,
                                const CSCRecHit2DCollection& rechitcltn) {
     float  adc_3_3_sum,adc_3_3_wtbin,x,y;
     int cfeb,idchamber,ring;

     string name,title,endcapstr;
     ostringstream ss;
     std::vector<float> zer(6,0.0);

     CSCIndexer indexer;
     std::map<int,int>::iterator intIt;

     if(rechitcltn.begin() != rechitcltn.end()) {

  //   std::cout<<"Event "<<nEventsAnalyzed <<std::endl;

       // Looping thru rechit collection
       CSCRecHit2DCollection::const_iterator recIt;
       CSCRecHit2D::ADCContainer m_adc;
       CSCRecHit2D::ChannelContainer m_strip;
       for(recIt = rechitcltn.begin(); recIt != rechitcltn.end(); ++recIt) {
          CSCDetId id = (CSCDetId)(*recIt).cscDetId();
          // getting strips comprising rechit
          m_strip=(CSCRecHit2D::ChannelContainer)(*recIt).channels();
          if(m_strip.size()==3) {
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

               adc_3_3_sum=0.0;
               for(int j=binmx-1;j<=binmx+1;j++) { 
                  adc_3_3_sum=adc_3_3_sum+adc_left[j]
                                          +adc_center[j]
                                          +adc_right[j];
               }

                // ADC weighted time bin
                if(adc_3_3_sum > 100.0) {
                  
                  int centerStrip=m_strip[1]; //take central from 3 strips;
                // temporary fix
                  int flag=0;
                  if(id.station()==1 && id.ring()==4 &&  centerStrip>16) flag=1;
                // end of temporary fix
                  if(flag==0) {
                  adc_3_3_wtbin=getTiming(strpcltn, id, centerStrip);
                  idchamber=indexer.dbIndex(id, centerStrip)/10; //strips 1-16 ME1/1a
                                              // become strips 65-80 ME1/1 !!!
                  /*
                  if(id.station()==1 && (id.ring()==1 || id.ring()==4))
                  std::cout<<idchamber<<" "<<id.station()<<" "<<id.ring()<<" "<<m_strip[1]<<" "<<
                      "      "<<centerStrip<<
                         " "<<adc_3_3_wtbin<<"     "<<adc_3_3_sum<<std::endl;    
                  */      
                 ss<<"adc_3_3_weight_time_bin_vs_cfeb_occupancy_ME_"<<idchamber;
                 name=ss.str(); ss.str("");

                 string endcapstr;
                 if(id.endcap() == 1) endcapstr = "+";
                 if(id.endcap() == 2) endcapstr = "-";
                 ring=id.ring(); if(id.ring()==4) ring=1;
                 ss<<"ADC 3X3 Weighted Time Bin vs CFEB Occupancy ME"
                   <<endcapstr<<id.station()<<"/"<<ring<<"/"<<id.chamber();
                 title=ss.str(); ss.str("");

                 cfeb=(centerStrip-1)/16+1;
                 x=cfeb; y=adc_3_3_wtbin;
                 histos->fill2DHist(x,y,name.c_str(),title.c_str(),5,1.,6.,40,0.,8.,"ADCTiming");                                     
                 } // end of if flag==0
                } // end of if (adc_3_3_sum > 100.0)
            } // end of if if(m_strip.size()==3
       } // end of the  pass thru CSCRecHit2DCollection
     }  // end of if (rechitcltn.begin() != rechitcltn.end())
}


void CSCValidation::endJob() {

     std::cout<<"Events in "<<nEventsAnalyzed<<std::endl;
}

DEFINE_FWK_MODULE(CSCValidation);

