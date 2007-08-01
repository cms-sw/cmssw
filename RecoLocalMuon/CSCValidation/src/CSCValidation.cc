/*
 *  simple validation package for CSC DIGIs, RECHITs and SEGMENTs.
 *
 *  Michael Schmitt, Northwestern University.
 */
#include "RecoLocalMuon/CSCValidation/interface/CSCValidation.h"

using namespace std;
using namespace edm;


// Constructor
CSCValidation::CSCValidation(const ParameterSet& pset){

  // Get the various input parameters
  rootFileName     = pset.getUntrackedParameter<string>("rootFileName");

  // set counter to zero
  nEventsAnalyzed = 0;
  
  // Create the root file for the histograms
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Create the root tree to hold position info
  rHTree  = new TTree("rHPositions","Local and Global reconstructed positions for recHits");
  segTree = new TTree("segPositions","Local and Global reconstructed positions for segments");

  // Create a branch on the tree
  rHTree->Branch("rHpos",&rHpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");
  segTree->Branch("segpos",&segpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");

  // Create sub-directories for digis/rechits/segments
  theFile->mkdir("Digis");
  theFile->mkdir("recHits");
  theFile->mkdir("Segments");
  theFile->cd();

  // Book the histograms

  printf("\n\n\n==book my histograms====\n\n\n");

  // wire digis
  hWireAll  = new TH1F("hWireAll","all wire group numbers",121,-0.5,120.5);
  hWireTBinAll  = new TH1F("hWireTBinAll","time bins all wires",21,-0.5,20.5);
  hWirenGroupsTotal = new TH1F("hWirenGroupsTotal","total number of wire groups",101,-0.5,100.5);
  hWireCodeBroad = new TH1F("hWireCodeBroad","broad scope code for wires",33,-16.5,16.5);
  hWireCodeNarrow1 = new TH1F("hWireCodeNarrow1","narrow scope wire code station 1",801,-400.5,400.5);
  hWireCodeNarrow2 = new TH1F("hWireCodeNarrow2","narrow scope wire code station 2",801,-400.5,400.5);
  hWireCodeNarrow3 = new TH1F("hWireCodeNarrow3","narrow scope wire code station 3",801,-400.5,400.5);
  hWireCodeNarrow4 = new TH1F("hWireCodeNarrow4","narrow scope wire code station 4",801,-400.5,400.5);
  hWireLayer1 = new TH1F("hWireLayer1","layer wire station 1",7,-0.5,6.5);
  hWireLayer2 = new TH1F("hWireLayer2","layer wire station 2",7,-0.5,6.5);
  hWireLayer3 = new TH1F("hWireLayer3","layer wire station 3",7,-0.5,6.5);
  hWireLayer4 = new TH1F("hWireLayer4","layer wire station 4",7,-0.5,6.5);
  hWireWire1  = new TH1F("hWireWire1","wire number station 1",113,-0.5,112.5);
  hWireWire2  = new TH1F("hWireWire2","wire number station 2",113,-0.5,112.5);
  hWireWire3  = new TH1F("hWireWire3","wire number station 3",113,-0.5,112.5);
  hWireWire4  = new TH1F("hWireWire4","wire number station 4",113,-0.5,112.5);

  // strip digis
  hStripAll = new TH1F("hStripAll","all strip numbers",81,-0.5,80.5);
  hStripADCAll   = new TH1F("hStripADCAll","all ADC values above cutoff",100,0.,1000.);
  hStripNFired = new TH1F("hStripNFired","total number of fired strips",601,-0.5,600.5);
  hStripCodeBroad = new TH1F("hStripCodeBroad","broad scope code for strips",33,-16.5,16.5);
  hStripCodeNarrow1 = new TH1F("hStripCodeNarrow1","narrow scope strip code station 1",801,-400.5,400.5);
  hStripCodeNarrow2 = new TH1F("hStripCodeNarrow2","narrow scope strip code station 2",801,-400.5,400.5);
  hStripCodeNarrow3 = new TH1F("hStripCodeNarrow3","narrow scope strip code station 3",801,-400.5,400.5);
  hStripCodeNarrow4 = new TH1F("hStripCodeNarrow4","narrow scope strip code station 4",801,-400.5,400.5);
  hStripLayer1 = new TH1F("hStripLayer1","layer strip station 1",7,-0.5,6.5);
  hStripLayer2 = new TH1F("hStripLayer2","layer strip station 2",7,-0.5,6.5);
  hStripLayer3 = new TH1F("hStripLayer3","layer strip station 3",7,-0.5,6.5);
  hStripLayer4 = new TH1F("hStripLayer4","layer strip station 4",7,-0.5,6.5);
  hStripStrip1  = new TH1F("hStripStrip1","strip number station 1",81,-0.5,80.5);
  hStripStrip2  = new TH1F("hStripStrip2","strip number station 2",81,-0.5,80.5);
  hStripStrip3  = new TH1F("hStripStrip3","strip number station 3",81,-0.5,80.5);
  hStripStrip4  = new TH1F("hStripStrip4","strip number station 4",81,-0.5,80.5);

  // recHits
  hRHCodeBroad = new TH1F("hRHCodeBroad","broad scope code for recHits",33,-16.5,16.5);
  hRHCodeNarrow1 = new TH1F("hRHCodeNarrow1","narrow scope recHit code station 1",801,-400.5,400.5);
  hRHCodeNarrow2 = new TH1F("hRHCodeNarrow2","narrow scope recHit code station 2",801,-400.5,400.5);
  hRHCodeNarrow3 = new TH1F("hRHCodeNarrow3","narrow scope recHit code station 3",801,-400.5,400.5);
  hRHCodeNarrow4 = new TH1F("hRHCodeNarrow4","narrow scope recHit code station 4",801,-400.5,400.5);
  hRHLayer1 = new TH1F("hRHLayer1","layer recHit station 1",7,-0.5,6.5);
  hRHLayer2 = new TH1F("hRHLayer2","layer recHit station 2",7,-0.5,6.5);
  hRHLayer3 = new TH1F("hRHLayer3","layer recHit station 3",7,-0.5,6.5);
  hRHLayer4 = new TH1F("hRHLayer4","layer recHit station 4",7,-0.5,6.5);
  hRHX1 = new TH1F("hRHX1","local X recHit station 1",120,-60.,60.);
  hRHX2 = new TH1F("hRHX2","local X recHit station 2",160,-80.,80.);
  hRHX3 = new TH1F("hRHX3","local X recHit station 3",160,-80.,80.);
  hRHX4 = new TH1F("hRHX4","local X recHit station 4",160,-80.,80.);
  hRHY1 = new TH1F("hRHY1","local Y recHit station 1",50,-100.,100.);
  hRHY2 = new TH1F("hRHY2","local Y recHit station 2",60,-180.,180.);
  hRHY3 = new TH1F("hRHY3","local Y recHit station 3",60,-180.,180.);
  hRHY4 = new TH1F("hRHY4","local Y recHit station 4",60,-180.,180.);
  hRHGlobal1 = new TH2F("hRHGlobal1","recHit global X,Y station 1",400,-800.,800.,400,-800.,800.);
  hRHGlobal2 = new TH2F("hRHGlobal2","recHit global X,Y station 2",400,-800.,800.,400,-800.,800.);
  hRHGlobal3 = new TH2F("hRHGlobal3","recHit global X,Y station 3",400,-800.,800.,400,-800.,800.);
  hRHGlobal4 = new TH2F("hRHGlobal4","recHit global X,Y station 4",400,-800.,800.,400,-800.,800.);

  // segments
  hSCodeBroad = new TH1F("hSCodeBroad","broad scope code for recHits",33,-16.5,16.5);
  hSCodeNarrow1 = new TH1F("hSCodeNarrow1","narrow scope recHit code station 1",801,-400.5,400.5);
  hSCodeNarrow2 = new TH1F("hSCodeNarrow2","narrow scope recHit code station 2",801,-400.5,400.5);
  hSCodeNarrow3 = new TH1F("hSCodeNarrow3","narrow scope recHit code station 3",801,-400.5,400.5);
  hSCodeNarrow4 = new TH1F("hSCodeNarrow4","narrow scope recHit code station 4",801,-400.5,400.5);
  hSnHits1 = new TH1F("hSnHits1","N hits on Segments in Station 1",7,-0.5,6.5);
  hSnHits2 = new TH1F("hSnHits2","N hits on Segments in Station 2",7,-0.5,6.5);
  hSnHits3 = new TH1F("hSnHits3","N hits on Segments in Station 3",7,-0.5,6.5);
  hSnHits4 = new TH1F("hSnHits4","N hits on Segments in Station 4",7,-0.5,6.5);
  hSTheta1 = new TH1F("hSTheta1","local theta segments in Station 1",128,-3.2,3.2);
  hSTheta2 = new TH1F("hSTheta2","local theta segments in Station 2",128,-3.2,3.2);
  hSTheta3 = new TH1F("hSTheta3","local theta segments in Station 3",128,-3.2,3.2);
  hSTheta4 = new TH1F("hSTheta4","local theta segments in Station 4",128,-3.2,3.2);
  hSGlobal1 = new TH2F("hSGlobal1","segment global X,Y station 1",400,-800.,800.,400,-800.,800.);
  hSGlobal2 = new TH2F("hSGlobal2","segment global X,Y station 2",400,-800.,800.,400,-800.,800.);
  hSGlobal3 = new TH2F("hSGlobal3","segment global X,Y station 3",400,-800.,800.,400,-800.,800.);
  hSGlobal4 = new TH2F("hSGlobal4","segment global X,Y station 4",400,-800.,800.,400,-800.,800.);
  hSnhits = new TH1F("hSnhits","N hits on Segments",7,-0.5,6.5);
  hSChiSqProb = new TH1F("hSChiSqProb","segments chi-squared probability",100,0.,1.);
  hSGlobalTheta = new TH1F("hSGlobalTheta","segment global theta",64,0,1.6);
  hSGlobalPhi   = new TH1F("hSGlobalPhi",  "segment global phi",  128,-3.2,3.2);
  hSnSegments   = new TH1F("hSnSegments","number of segments",11,-0.5,10.5);

}

// Destructor
CSCValidation::~CSCValidation(){
  
  // write out total number of events processed
  printf("\n\n======= write out my histograms ====\n");
  printf("\n\ttotal number of events processed: %i\n\n\n\n",nEventsAnalyzed);


  // Write the histos to file

  theFile->cd();

  // wire digis
  theFile->cd("Digis");
  hWireAll->Write();
  hWireTBinAll->Write();
  hWirenGroupsTotal->Write();
  hWireCodeBroad->Write();
  hWireCodeNarrow1->Write();
  hWireCodeNarrow2->Write();
  hWireCodeNarrow3->Write();
  hWireCodeNarrow4->Write();
  hWireLayer1->Write();
  hWireLayer2->Write();
  hWireLayer3->Write();
  hWireLayer4->Write();
  hWireWire1->Write();
  hWireWire2->Write();
  hWireWire3->Write();
  hWireWire4->Write();

  // strip digis
  hStripAll->Write();
  hStripADCAll->Write();
  hStripNFired->Write();
  hStripCodeBroad->Write();
  hStripCodeNarrow1->Write();
  hStripCodeNarrow2->Write();
  hStripCodeNarrow3->Write();
  hStripCodeNarrow4->Write();
  hStripLayer1->Write();
  hStripLayer2->Write();
  hStripLayer3->Write();
  hStripLayer4->Write();
  hStripStrip1->Write();
  hStripStrip2->Write();
  hStripStrip3->Write();
  hStripStrip4->Write();
  theFile->cd();

  // recHits
  theFile->cd("recHits");
  hRHCodeBroad->Write();
  hRHCodeNarrow1->Write();
  hRHCodeNarrow2->Write();
  hRHCodeNarrow3->Write();
  hRHCodeNarrow4->Write();
  hRHLayer1->Write();
  hRHLayer2->Write();
  hRHLayer3->Write();
  hRHLayer4->Write();
  hRHX1->Write();
  hRHX2->Write();
  hRHX3->Write();
  hRHX4->Write();
  hRHY1->Write();
  hRHY2->Write();
  hRHY3->Write();
  hRHY4->Write();
  hRHGlobal1->Write();
  hRHGlobal2->Write();
  hRHGlobal3->Write();
  hRHGlobal4->Write();
  rHTree->Write();
  theFile->cd();

  // segments
  theFile->cd("Segments");
  hSCodeBroad->Write();
  hSCodeNarrow1->Write();
  hSCodeNarrow2->Write();
  hSCodeNarrow3->Write();
  hSCodeNarrow4->Write();
  hSnHits1->Write();
  hSnHits2->Write();
  hSnHits3->Write();
  hSnHits4->Write();
  hSTheta1->Write();
  hSTheta2->Write();
  hSTheta3->Write();
  hSTheta4->Write();
  hSGlobal1->Write();
  hSGlobal2->Write();
  hSGlobal3->Write();
  hSGlobal4->Write();
  hSnhits->Write();
  hSChiSqProb->Write();
  hSGlobalTheta->Write();
  hSGlobalPhi->Write();
  hSnSegments->Write();
  segTree->Write();
  theFile->cd();

  //
  theFile->Close();

}

// The Analysis  (the main)
void CSCValidation::analyze(const Event & event, const EventSetup& eventSetup){
  
  // increment counter
  nEventsAnalyzed++;

  // printalot debug output
  bool printalot = (nEventsAnalyzed < 100);

  int iRun   = event.id().run();
  int iEvent = event.id().event();
  if (printalot) printf("\n==enter==CSCValidation===== run %i\tevent %i\tn Analyzed %i\n",iRun,iEvent,nEventsAnalyzed);

  //
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  if (printalot) printf("\tget handles for digi collections\n");
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCComparatorDigiCollection> comparators;
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCRPCDigiCollection> rpcs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts;
  
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //
  if (printalot) printf("\tpass handles\n");

  event.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
  event.getByLabel("muonCSCDigis","MuonCSCStripDigi",strips);
  /*
  event.getByLabel("muonCSCDigis","MuonCSCComparatorDigi",comparators);
  event.getByLabel("muonCSCDigis","MuonCSCALCTDigi",alcts);
  event.getByLabel("muonCSCDigis","MuonCSCCLCTDigi",clcts);
  event.getByLabel("muonCSCDigis","MuonCSCRPCDigi",rpcs);
  event.getByLabel("muonCSCDigis","MuonCSCCorrelatedLCTDigi",correlatedlcts);
  */

  // ==============================================
  //
  // look at DIGIs
  //
  // ===============================================

  //
  // WIRE GROUPS
  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myWire = digiItr->getWireGroup();
      int myTBin = digiItr->getTimeBin();
      hWireAll->Fill(myWire);
      hWireTBinAll->Fill(myTBin);
      nWireGroupsTotal++;
      int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
      int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;
      hWireCodeBroad->Fill(kCodeBroad);
      if (kStation == 1) {
	hWireCodeNarrow1->Fill(kCodeNarrow);
	hWireLayer1->Fill(kLayer);
	hWireWire1->Fill(myWire);
      }
      if (kStation == 2) {
	hWireCodeNarrow2->Fill(kCodeNarrow);
	hWireLayer2->Fill(kLayer);
	hWireWire2->Fill(myWire);
      }
      if (kStation == 3) {
	hWireCodeNarrow3->Fill(kCodeNarrow);
	hWireLayer3->Fill(kLayer);
	hWireWire3->Fill(myWire);
      }
      if (kStation == 4) {
	hWireCodeNarrow4->Fill(kCodeNarrow);
	hWireLayer4->Fill(kLayer);
	hWireWire4->Fill(myWire);
      }
    }
  }
  if (nWireGroupsTotal > 0) {hWirenGroupsTotal->Fill(nWireGroupsTotal);}
  

  //
  // STRIPS
  //
  int nStripsFired = 0;
  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
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
	if (thisStripFired) {
	  nStripsFired++;
	  hStripAll->Fill(myStrip);
	  hStripADCAll->Fill(diff);
	  int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
	  int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;
	  hStripCodeBroad->Fill(kCodeBroad);
	  if (kStation == 1) {
	    hStripCodeNarrow1->Fill(kCodeNarrow);
	    hStripLayer1->Fill(kLayer);
	    hStripStrip1->Fill(myStrip);
	  }
	  if (kStation == 2) {
	    hStripCodeNarrow2->Fill(kCodeNarrow);
	    hStripLayer2->Fill(kLayer);
	    hStripStrip2->Fill(myStrip);
	  }
	  if (kStation == 3) {
	    hStripCodeNarrow3->Fill(kCodeNarrow);
	    hStripLayer3->Fill(kLayer);
	    hStripStrip3->Fill(myStrip);
	  }
	  if (kStation == 4) {
	    hStripCodeNarrow4->Fill(kCodeNarrow);
	    hStripLayer4->Fill(kLayer);
	    hStripStrip4->Fill(myStrip);
	  }
	}
      }
    }
  }
  if (nStripsFired > 0) {hStripNFired->Fill(nStripsFired);}
    


  // ==============================================
  //
  // look at RECHITs
  //
  // ===============================================

  // Get the CSC Geometry :
  if (printalot) printf("\tget the CSC geometry.\n");
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);
  
  // Get the RecHits collection :
  if (printalot) printf("\tGet the recHits collection.\t");
  Handle<CSCRecHit2DCollection> recHits; 
  event.getByLabel("csc2DRecHits",recHits);  
  int nRecHits = recHits->size();
  if (printalot) printf("  The size is %i\n",nRecHits);
 
  // ---------------------
  // Loop over rechits 
  // ---------------------
  if (printalot) printf("\t...start loop over rechits...\n");
  int iHit = 0;

  // Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    iHit++;
    if (printalot) printf("\t\thit number %i\n",iHit);

    // Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();
    if (printalot) printf("\t\t\tendcap/ring/station/chamber/layer: %i/%i/%i/%i/%i\n",kEndcap,kRing,kStation,kChamber,kLayer);

    // Store reco hit as a Local Point:
    LocalPoint rhitlocal = (*recIt).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    float zreco = rhitlocal.z();
    LocalError rerrlocal = (*recIt).localPositionError();  
    float xxerr = rerrlocal.xx();
    float yyerr = rerrlocal.yy();
    float xyerr = rerrlocal.xy();

    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );
	
    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx = rhitglobal.x();
    float grecy = rhitglobal.y();
    float grecz = rhitglobal.z();

    if (printalot) printf("\t\t\tx,y,z: %f, %f, %f\texx,eey,exy: %f, %f, %f\tglobal x,y,z: %f, %f, %f \n",xreco,yreco,zreco,xxerr,yyerr,xyerr,grecx,grecy,grecz);

    // Fill the rechit position branch
    rHpos.localx  = xreco;
    rHpos.localy  = yreco;
    rHpos.globalx = grecx;
    rHpos.globaly = grecy;
    rHpos.endcap  = kEndcap;
    rHpos.ring    = kRing;
    rHpos.station = kStation;
    rHpos.chamber = kChamber;
    rHpos.layer   = kLayer;
    rHTree->Fill();

    // Fill some histograms
    int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;
    hRHCodeBroad->Fill(kCodeBroad);
    if (kStation == 1) {
      hRHCodeNarrow1->Fill(kCodeNarrow);
      hRHLayer1->Fill(kLayer);
      hRHX1->Fill(xreco);
      hRHY1->Fill(yreco);
      hRHGlobal1->Fill(grecx,grecy);
    }
    if (kStation == 2) {
      hRHCodeNarrow2->Fill(kCodeNarrow);
      hRHLayer2->Fill(kLayer);
      hRHX2->Fill(xreco);
      hRHY2->Fill(yreco);
      hRHGlobal2->Fill(grecx,grecy);
    }
    if (kStation == 3) {
      hRHCodeNarrow3->Fill(kCodeNarrow);
      hRHLayer3->Fill(kLayer);
      hRHX3->Fill(xreco);
      hRHY3->Fill(yreco);
      hRHGlobal3->Fill(grecx,grecy);
    }
    if (kStation == 4) {
      hRHCodeNarrow4->Fill(kCodeNarrow);
      hRHLayer4->Fill(kLayer);
      hRHX4->Fill(xreco);
      hRHY4->Fill(yreco);
      hRHGlobal4->Fill(grecx,grecy);
    }
    
    // get the channels in this recHit
    CSCRecHit2D::ChannelContainer chan = (CSCRecHit2D::ChannelContainer)(*recIt).channels();
    int nChan = chan.size();
    if (printalot) printf("\t\t\t\tn channels = %i :\t",nChan);
    for (unsigned int thisChan = 0; thisChan != chan.size(); thisChan++) {
      if (printalot) printf(" %i, ",chan[thisChan]);
    }
    if (printalot) printf("\n");

  }

  // ==============================================
  //
  // look at SEGMENTs
  //
  // ===============================================

  // get CSC segment collection
  if (printalot) printf("\tGet CSC segment collection...");
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);
  int nSegments = cscSegments->size();
  if (printalot) printf("  The size is %i\n",nSegments);

  // -----------------------
  // loop over segments
  // -----------------------
  int iSegment = 0;
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    iSegment++;
    //
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    if (printalot) printf("\tendcap/ring/station/chamber: %i %i %i %i\n",kEndcap,kRing,kStation,kChamber);
    //
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();
    int nDOF       = 2*nhits-4;
    double chisqProb = ChiSquaredProbability( (double)chisq, nDOF );
    LocalPoint localPos = (*it).localPosition();
    float segX     = localPos.x();
    float segY     = localPos.y();
    float segZ     = localPos.z();
    LocalVector segDir = (*it).localDirection();
    double theta   = segDir.theta();
    double phi     = segDir.phi();
    if (printalot) printf("\tlocal position: %f %f %f\ttheta,phi: %f %f\n",segX,segY,segZ,theta,phi);

    //
    // try to get the CSC recHits that contribute to this segment.
    if (printalot) printf("\tGet the recHits for this segment.\t");
    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    int nRH = (*it).nRecHits();
    if (printalot) printf("    nRH = %i\n",nRH);
    int jRH = 0;
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      int kEndcap  = idRH.endcap();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
      int kChamber = idRH.chamber();
      int kLayer   = idRH.layer();
      if (printalot) printf("\t%i RH\tendcap/station/ring/chamber/layer: %i %i %i %i %i\n",jRH,kEndcap,kRing,kStation,kChamber,kLayer);
    }

    //
    // global transformation: from Ingo Bloch
    float globX = 0.;
    float globY = 0.;
    float globZ = 0.;
    float globTheta = 0.;
    float globPhi   = 0.;
    const CSCChamber* cscchamber = cscGeom->chamber(id);
    if (cscchamber) {
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globX = globalPosition.x();
      globY = globalPosition.y();
      globZ = globalPosition.z();
      GlobalVector globalDirection = cscchamber->toGlobal(segDir);
      globTheta = globalDirection.theta();
      globPhi   = globalDirection.phi();
    } else {
      if (printalot) printf("\tFailed to get a local->global segment tranformation.\n");
    }
    //
    if (printalot) printf("\t\tsegment %i\tchisq,nhits: %f, %i\tx,y,z: %f, %f, %f\t%f, %f, %f\n",iSegment,chisq,nhits,segX,segY,segZ,globX,globY,globZ);
    if (printalot) printf("\t\ttheta,phi: %f %f\t%f %f\n",theta,phi,globTheta,globPhi);


    // Fill the segment position branch
    segpos.localx  = segX;
    segpos.localy  = segY;
    segpos.globalx = globX;
    segpos.globaly = globY;
    segpos.endcap  = kEndcap;
    segpos.ring    = kRing;
    segpos.station = kStation;
    segpos.chamber = kChamber;
    segpos.layer   = 0;
    segTree->Fill();


    // Fill some histograms
    int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;
    hSCodeBroad->Fill(kCodeBroad);
    if (kStation == 1) {
      hSCodeNarrow1->Fill(kCodeNarrow);
      hSnHits1->Fill(nhits);
      hSTheta1->Fill(theta);
      hSGlobal1->Fill(globX,globY);
    }
    if (kStation == 2) {
      hSCodeNarrow2->Fill(kCodeNarrow);
      hSnHits2->Fill(nhits);
      hSTheta2->Fill(theta);
      hSGlobal2->Fill(globX,globY);
    }
    if (kStation == 3) {
      hSCodeNarrow3->Fill(kCodeNarrow);
      hSnHits3->Fill(nhits);
      hSTheta3->Fill(theta);
      hSGlobal3->Fill(globX,globY);
    }
    if (kStation == 4) {
      hSCodeNarrow4->Fill(kCodeNarrow);
      hSnHits4->Fill(nhits);
      hSTheta4->Fill(theta);
      hSGlobal4->Fill(globX,globY);
    }
    hSnhits->Fill(nhits);
    hSChiSqProb->Fill(chisqProb);
    hSGlobalTheta->Fill(globTheta);
    hSGlobalPhi->Fill(globPhi);
  }
  hSnSegments->Fill(nSegments);



  // exit
  if (printalot) printf("==exit===CSCValidation===== run %i\tevent %i\n\n",iRun,iEvent);
}

DEFINE_FWK_MODULE(CSCValidation);

