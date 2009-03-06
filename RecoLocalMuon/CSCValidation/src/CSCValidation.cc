/*
 *  simple validation package for CSC DIGIs, RECHITs and SEGMENTs.
 *
 *  Michael Schmitt
 *  Andy Kubik
 *  Northwestern University
 */
#include "RecoLocalMuon/CSCValidation/interface/CSCValidation.h"

using namespace std;
using namespace edm;


// Constructor
CSCValidation::CSCValidation(const ParameterSet& pset){

  // Get the various input parameters
  rootFileName     = pset.getUntrackedParameter<string>("rootFileName");
  isSimulation     = pset.getUntrackedParameter<bool>("isSimulation");

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
  theFile->mkdir("Calib");
  theFile->cd();


  // calib
  hCalibGainsS = new TH1F("hCalibGainsS","Gains Slope",400,0,400);
  hCalibXtalkSL = new TH1F("hCalibXtalkSL","Xtalk Slope Left",400,0,400);
  hCalibXtalkSR = new TH1F("hCalibXtalkSR","Xtalk Slope Right",400,0,400);
  hCalibXtalkIL = new TH1F("hCalibXtalkIL","Xtalk Intercept Left",400,0,400);
  hCalibXtalkIR = new TH1F("hCalibXtalkIR","Xtalk Intercept Right",400,0,400);
  hCalibPedsP = new TH1F("hCalibPedsP","Peds",400,0,400);
  hCalibPedsR = new TH1F("hCalibPedsR","Peds RMS",400,0,400);
  hCalibNoise33 = new TH1F("hCalibNoise33","Noise Matrix 33",400,0,400);
  hCalibNoise34 = new TH1F("hCalibNoise34","Noise Matrix 34",400,0,400);
  hCalibNoise35 = new TH1F("hCalibNoise35","Noise Matrix 35",400,0,400);
  hCalibNoise44 = new TH1F("hCalibNoise44","Noise Matrix 44",400,0,400);
  hCalibNoise45 = new TH1F("hCalibNoise45","Noise Matrix 45",400,0,400);
  hCalibNoise46 = new TH1F("hCalibNoise46","Noise Matrix 46",400,0,400);
  hCalibNoise55 = new TH1F("hCalibNoise55","Noise Matrix 55",400,0,400);
  hCalibNoise56 = new TH1F("hCalibNoise56","Noise Matrix 56",400,0,400);
  hCalibNoise57 = new TH1F("hCalibNoise57","Noise Matrix 57",400,0,400);
  hCalibNoise66 = new TH1F("hCalibNoise66","Noise Matrix 66",400,0,400);
  hCalibNoise67 = new TH1F("hCalibNoise67","Noise Matrix 67",400,0,400);
  hCalibNoise77 = new TH1F("hCalibNoise77","Noise Matrix 77",400,0,400);




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

  // tmp efficiency histos
  hSSTE = new TH1F("hSSTE","hSSTE",20,0,20);
  hRHSTE = new TH1F("hRHSTE","hRHSTE",20,0,20);
  hSEff = new TH1F("hSEff","Segment Efficiency",10,0.5,10.5);
  hRHEff = new TH1F("hRHEff","recHit Efficiency",10,0.5,10.5);

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
  hRHResid11b = new TH1F("hRHResid11b","SimHitX - Reconstructed X (ME11b)",100,-1.0,1.0);
  hRHResid12 = new TH1F("hRHResid12","SimHitX - Reconstructed X (ME11)",100,-1.0,1.0);
  hRHResid13 = new TH1F("hRHResid13","SimHitX - Reconstructed X (ME11)",100,-1.0,1.0);
  hRHResid11a = new TH1F("hRHResid11a","SimHitX - Reconstructed X (ME11a)",100,-1.0,1.0);
  hRHResid21 = new TH1F("hRHResid21","SimHitX - Reconstructed X (ME21)",100,-1.0,1.0);
  hRHResid22 = new TH1F("hRHResid22","SimHitX - Reconstructed X (ME22)",100,-1.0,1.0);
  hRHResid31 = new TH1F("hRHResid31","SimHitX - Reconstructed X (ME31)",100,-1.0,1.0);
  hRHResid32 = new TH1F("hRHResid32","SimHitX - Reconstructed X (ME32)",100,-1.0,1.0);
  hRHResid41 = new TH1F("hRHResid41","SimHitX - Reconstructed X (ME41)",100,-1.0,1.0);
  hRHResid42 = new TH1F("hRHResid42","SimHitX - Reconstructed X (ME42)",100,-1.0,1.0);
  hRHSumQ11b = new TH1F("hRHSumQ11b","Sum 3x3 recHit Charge (ME11b)",250,0,2000);
  hRHSumQ12 = new TH1F("hRHSumQ12","Sum 3x3 recHit Charge (ME12)",250,0,2000);
  hRHSumQ13 = new TH1F("hRHSumQ13","Sum 3x3 recHit Charge (ME13)",250,0,2000);
  hRHSumQ11a = new TH1F("hRHSumQ11a","Sum 3x3 recHit Charge (ME11a)",250,0,2000);
  hRHSumQ21 = new TH1F("hRHSumQ21","Sum 3x3 recHit Charge (ME21)",250,0,2000);
  hRHSumQ22 = new TH1F("hRHSumQ22","Sum 3x3 recHit Charge (ME22)",250,0,2000);
  hRHSumQ31 = new TH1F("hRHSumQ31","Sum 3x3 recHit Charge (ME31)",250,0,2000);
  hRHSumQ32 = new TH1F("hRHSumQ32","Sum 3x3 recHit Charge (ME32)",250,0,2000);
  hRHSumQ41 = new TH1F("hRHSumQ41","Sum 3x3 recHit Charge (ME41)",250,0,2000);
  hRHSumQ42 = new TH1F("hRHSumQ42","Sum 3x3 recHit Charge (ME42)",250,0,2000);
  hRHRatioQ11b = new TH1F("hRHRatioQ11b","Ratio (Ql+Qr)/Qc (ME11b)",120,-0.1,1.1);
  hRHRatioQ12 = new TH1F("hRHRatioQ12","Ratio (Ql+Qr)/Qc (ME12)",120,-0.1,1.1);
  hRHRatioQ13 = new TH1F("hRHRatioQ13","Ratio (Ql+Qr)/Qc (ME13)",120,-0.1,1.1);
  hRHRatioQ11a = new TH1F("hRHRatioQ11a","Ratio (Ql+Qr)/Qc (ME11a)",120,-0.1,1.1);
  hRHRatioQ21 = new TH1F("hRHRatioQ21","Ratio (Ql+Qr)/Qc (ME21)",120,-0.1,1.1);
  hRHRatioQ22 = new TH1F("hRHRatioQ22","Ratio (Ql+Qr)/Qc (ME22)",120,-0.1,1.1);
  hRHRatioQ31 = new TH1F("hRHRatioQ31","Ratio (Ql+Qr)/Qc (ME31)",120,-0.1,1.1);
  hRHRatioQ32 = new TH1F("hRHRatioQ32","Ratio (Ql+Qr)/Qc (ME32)",120,-0.1,1.1);
  hRHRatioQ41 = new TH1F("hRHRatioQ41","Ratio (Ql+Qr)/Qc (ME41)",120,-0.1,1.1);
  hRHRatioQ42 = new TH1F("hRHRatioQ42","Ratio (Ql+Qr)/Qc (ME42)",120,-0.1,1.1);
  hRHTiming11a = new TH1F("hRHTiming11b","recHit Timing (ME11b)",100,0,10);
  hRHTiming12 = new TH1F("hRHTiming12","recHit Timing (ME12)",100,0,10);
  hRHTiming13 = new TH1F("hRHTiming13","recHit Timing (ME13)",100,0,10);
  hRHTiming11b = new TH1F("hRHTiming11a","recHit Timing (ME11a)",100,0,10);
  hRHTiming21 = new TH1F("hRHTiming21","recHit Timing (ME21)",100,0,10);
  hRHTiming22 = new TH1F("hRHTiming22","recHit Timing (ME22)",100,0,10);
  hRHTiming31 = new TH1F("hRHTiming31","recHit Timing (ME31)",100,0,10);
  hRHTiming32 = new TH1F("hRHTiming32","recHit Timing (ME32)",100,0,10);
  hRHTiming41 = new TH1F("hRHTiming41","recHit Timing (ME41)",100,0,10);
  hRHTiming42 = new TH1F("hRHTiming42","recHit Timing (ME42)",100,0,10);



  // segments
  hSCodeBroad = new TH1F("hSCodeBroad","broad scope code for recHits",33,-16.5,16.5);
  hSCodeNarrow1 = new TH1F("hSCodeNarrow1","narrow scope Segment code station 1",801,-400.5,400.5);
  hSCodeNarrow2 = new TH1F("hSCodeNarrow2","narrow scope Segment code station 2",801,-400.5,400.5);
  hSCodeNarrow3 = new TH1F("hSCodeNarrow3","narrow scope Segment code station 3",801,-400.5,400.5);
  hSCodeNarrow4 = new TH1F("hSCodeNarrow4","narrow scope Segment code station 4",801,-400.5,400.5);
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
  hSResid11b = new TH1F("hSResid11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME11b)",100,-0.5,0.5);
  hSResid12  = new TH1F("hSResid12","Fitted Position on Strip - Reconstructed for Layer 3 (ME12)",100,-0.5,0.5);
  hSResid13  = new TH1F("hSResid13","Fitted Position on Strip - Reconstructed for Layer 3 (ME13)",100,-0.5,0.5);
  hSResid11a = new TH1F("hSResid11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME11a)",100,-0.5,0.5);
  hSResid21  = new TH1F("hSResid21","Fitted Position on Strip - Reconstructed for Layer 3 (ME21)",100,-0.5,0.5);
  hSResid22  = new TH1F("hSResid22","Fitted Position on Strip - Reconstructed for Layer 3 (ME22)",100,-0.5,0.5);
  hSResid31  = new TH1F("hSResid31","Fitted Position on Strip - Reconstructed for Layer 3 (ME31)",100,-0.5,0.5);
  hSResid32  = new TH1F("hSResid32","Fitted Position on Strip - Reconstructed for Layer 3 (ME32)",100,-0.5,0.5);
  hSResid41  = new TH1F("hSResid41","Fitted Position on Strip - Reconstructed for Layer 3 (ME41)",100,-0.5,0.5);
  hSResid42  = new TH1F("hSResid42","Fitted Position on Strip - Reconstructed for Layer 3 (ME42)",100,-0.5,0.5);

}

// Destructor
CSCValidation::~CSCValidation(){
  

  // Write the histos to file

  theFile->cd();

  // calib
  theFile->cd("Calib");
  hCalibGainsS->Write();
  hCalibXtalkSL->Write();
  hCalibXtalkSR->Write();
  hCalibXtalkIL->Write();
  hCalibXtalkIR->Write();
  hCalibPedsP->Write();
  hCalibPedsR->Write();
  hCalibNoise33->Write();
  hCalibNoise34->Write();
  hCalibNoise35->Write();
  hCalibNoise44->Write();
  hCalibNoise45->Write();
  hCalibNoise46->Write();
  hCalibNoise55->Write();
  hCalibNoise56->Write();
  hCalibNoise57->Write();
  hCalibNoise66->Write();
  hCalibNoise67->Write();
  hCalibNoise77->Write();
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
  if (isSimulation){
    hRHResid11b->Write();
    hRHResid12->Write();
    hRHResid13->Write();
    hRHResid11a->Write();
    hRHResid21->Write();
    hRHResid22->Write();
    hRHResid31->Write();
    hRHResid32->Write();
    hRHResid41->Write();
    hRHResid42->Write();
  }
  hSResid11b->Write();
  hSResid12->Write();
  hSResid13->Write();
  hSResid11a->Write();
  hSResid21->Write();
  hSResid22->Write();
  hSResid31->Write();
  hSResid32->Write();
  hSResid41->Write();
  hSResid42->Write();
  hRHSumQ11b->Write();
  hRHSumQ12->Write();
  hRHSumQ13->Write();
  hRHSumQ11a->Write();
  hRHSumQ21->Write();
  hRHSumQ22->Write();
  hRHSumQ31->Write();
  hRHSumQ32->Write();
  hRHSumQ41->Write();
  hRHSumQ42->Write();
  hRHRatioQ11b->Write();
  hRHRatioQ12->Write();
  hRHRatioQ13->Write();
  hRHRatioQ11a->Write();
  hRHRatioQ21->Write();
  hRHRatioQ22->Write();
  hRHRatioQ31->Write();
  hRHRatioQ32->Write();
  hRHRatioQ41->Write();
  hRHRatioQ42->Write();
  hRHTiming11a->Write();
  hRHTiming12->Write();
  hRHTiming13->Write();
  hRHTiming11b->Write();
  hRHTiming21->Write();
  hRHTiming22->Write();
  hRHTiming31->Write();
  hRHTiming32->Write();
  hRHTiming41->Write();
  hRHTiming42->Write();
  histoEfficiency(hRHSTE,hRHEff);
  hRHEff->Write();
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
  histoEfficiency(hSSTE,hSEff);
  hSEff->Write();
  segTree->Write();
  theFile->cd();

  //
  theFile->Close();

}

// The Analysis  (the main)
void CSCValidation::analyze(const Event & event, const EventSetup& eventSetup){
  
  // increment counter
  nEventsAnalyzed++;


  int iRun   = event.id().run();
  int iEvent = event.id().event();

  //
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
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
  // look at Calibrations
  //
  // ==============================================

  // Only do this for the first event
  if (nEventsAnalyzed == 1){
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

    for (int i = 0; i < 400; i++){
      hCalibGainsS->SetBinContent(i+1,pGains->gains[i].gain_slope);
      hCalibXtalkSL->SetBinContent(i+1,pCrosstalk->crosstalk[i].xtalk_slope_left);
      hCalibXtalkSR->SetBinContent(i+1,pCrosstalk->crosstalk[i].xtalk_slope_right);
      hCalibXtalkIL->SetBinContent(i+1,pCrosstalk->crosstalk[i].xtalk_intercept_left);
      hCalibXtalkIR->SetBinContent(i+1,pCrosstalk->crosstalk[i].xtalk_intercept_right);
      hCalibPedsP->SetBinContent(i+1,pPedestals->pedestals[i].ped);
      hCalibPedsR->SetBinContent(i+1,pPedestals->pedestals[i].rms);
      hCalibNoise33->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem33);
      hCalibNoise34->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem34);
      hCalibNoise35->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem35);
      hCalibNoise44->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem44);
      hCalibNoise45->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem45);
      hCalibNoise46->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem46);
      hCalibNoise55->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem55);
      hCalibNoise56->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem56);
      hCalibNoise57->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem57);
      hCalibNoise66->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem66);
      hCalibNoise67->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem67);
      hCalibNoise77->SetBinContent(i+1,pNoiseMatrix->matrix[i].elem77);
    }  

  } // end calib
  
  // ==============================================
  //
  // look at DIGIs
  //
  // ==============================================

  //
  // WIRE GROUPS
  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    if (kEndcap == 2) kEndcap = -1;
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
    if (kEndcap == 2) kEndcap = -1;
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
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);
  
  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits; 
  event.getByLabel("csc2DRecHits",recHits);  
  int nRecHits = recHits->size();

  // Get the SimHits (if applicable)
  Handle<PSimHitContainer> simHits;
  if (isSimulation) event.getByLabel("g4SimHits", "MuonCSCHits", simHits);
 
 
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
    if (kEndcap == 2) kEndcap = -1;
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    // Store reco hit as a Local Point:
    LocalPoint rhitlocal = (*recIt).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    float zreco = rhitlocal.z();
    float phireco = rhitlocal.phi();
    LocalError rerrlocal = (*recIt).localPositionError();  
    float xxerr = rerrlocal.xx();
    float yyerr = rerrlocal.yy();
    float xyerr = rerrlocal.xy();

    // Find the charge associated with this hit
    CSCRecHit2D::ChannelContainer hitstrips = (*recIt).channels();
    int nStrips     =  hitstrips.size();
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];
    HepMatrix rHcharge = getCharge3x3(*strips, idrec, centerStrip);    
    float rHSumQ = rHcharge(1,1) + rHcharge(1,2) + rHcharge(1,3) +
                   rHcharge(2,1) + rHcharge(2,2) + rHcharge(2,3) +
                   rHcharge(3,1) + rHcharge(3,2) + rHcharge(3,3);
    float rHratioQ = (rHcharge(1,1) + rHcharge(1,2) + rHcharge(1,3)  +
                      rHcharge(3,1) + rHcharge(3,2) + rHcharge(3,3)) /
                     (rHcharge(2,1) + rHcharge(2,2) + rHcharge(2,3));

    // Get the signal timing of this hit
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



    float simHitXres = -99;
    float simHitYres = -99;
    float mindiffX   = 99;
    float mindiffY   = 10;
    // If MC, find closest muon simHit to check resolution:
    if (isSimulation){
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
    }

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
      if (kRing == 1){
        if (isSimulation) hRHResid11b->Fill(simHitXres);
        hRHSumQ11b->Fill(rHSumQ);
        hRHRatioQ11b->Fill(rHratioQ);
        hRHTiming11b->Fill(rHtime);
      }
      if (kRing == 2){
        if (isSimulation) hRHResid12->Fill(simHitXres);
        hRHSumQ12->Fill(rHSumQ);
        hRHRatioQ12->Fill(rHratioQ);
        hRHTiming12->Fill(rHtime);
      }
      if (kRing == 3){
        if (isSimulation) hRHResid13->Fill(simHitXres);
        hRHSumQ13->Fill(rHSumQ);
        hRHRatioQ13->Fill(rHratioQ);
        hRHTiming13->Fill(rHtime);
      }
      if (kRing == 4){
        if (isSimulation) hRHResid11a->Fill(simHitXres);
        hRHSumQ11a->Fill(rHSumQ);
        hRHRatioQ11a->Fill(rHratioQ);
        hRHTiming11a->Fill(rHtime);
      }
    }
    if (kStation == 2) {
      hRHCodeNarrow2->Fill(kCodeNarrow);
      hRHLayer2->Fill(kLayer);
      hRHX2->Fill(xreco);
      hRHY2->Fill(yreco);
      hRHGlobal2->Fill(grecx,grecy);
      if (kRing == 1){
        if (isSimulation) hRHResid21->Fill(simHitXres);
        hRHSumQ21->Fill(rHSumQ);
        hRHRatioQ21->Fill(rHratioQ);
        hRHTiming21->Fill(rHtime);
      }
      if (kRing == 2){
        if (isSimulation) hRHResid22->Fill(simHitXres);
        hRHSumQ22->Fill(rHSumQ);
        hRHRatioQ22->Fill(rHratioQ);
        hRHTiming22->Fill(rHtime);
      }
    }
    if (kStation == 3) {
      hRHCodeNarrow3->Fill(kCodeNarrow);
      hRHLayer3->Fill(kLayer);
      hRHX3->Fill(xreco);
      hRHY3->Fill(yreco);
      hRHGlobal3->Fill(grecx,grecy);
      if (kRing == 1){
        if (isSimulation) hRHResid31->Fill(simHitXres);
        hRHSumQ31->Fill(rHSumQ);
        hRHRatioQ31->Fill(rHratioQ);
        hRHTiming31->Fill(rHtime);
      }
      if (kRing == 2){
        if (isSimulation) hRHResid32->Fill(simHitXres);
        hRHSumQ32->Fill(rHSumQ);
        hRHRatioQ32->Fill(rHratioQ);
        hRHTiming32->Fill(rHtime);
      }
    }
    if (kStation == 4) {
      hRHCodeNarrow4->Fill(kCodeNarrow);
      hRHLayer4->Fill(kLayer);
      hRHX4->Fill(xreco);
      hRHY4->Fill(yreco);
      hRHGlobal4->Fill(grecx,grecy);
      if (kRing == 1){
        if (isSimulation) hRHResid41->Fill(simHitXres);
        hRHSumQ41->Fill(rHSumQ);
        hRHRatioQ41->Fill(rHratioQ);
        hRHTiming41->Fill(rHtime);
      }
      if (kRing == 2){
        if (isSimulation) hRHResid42->Fill(simHitXres);
        hRHSumQ42->Fill(rHSumQ);
        hRHRatioQ42->Fill(rHratioQ);
        hRHTiming42->Fill(rHtime);
      }
    }

  }

  // ==============================================
  //
  // look at SEGMENTs
  //
  // ===============================================

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);
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
    if (kEndcap == 2) kEndcap = -1;
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

      // If this segment has 6 hits, find the position of each hit on the strip in units of stripwidth and store values
      if (nRH == 6){
        const CSCLayer* csclayer = cscGeom->layer( idRH );
        const CSCLayerGeometry *layerGeom = csclayer->geometry();
        LocalPoint rhlp = (*iRH).localPosition();
        float swidth = layerGeom->stripPitch(rhlp);
        se(kLayer,1) = sqrt((*iRH).localPositionError().xx())/swidth;
        // Take into account half-strip staggering of layers (ME1/1 has no staggering)
        if (kStation == 1 && (kRing == 1 || kRing == 4)) sp(kLayer,1) = layerGeom->strip(rhlp);
        else{
          if (kLayer == 1 || kLayer == 3 || kLayer == 5) sp(kLayer,1) = layerGeom->strip(rhlp);
          if (kLayer == 2 || kLayer == 4 || kLayer == 6) sp(kLayer,1) = layerGeom->strip(rhlp) - 0.5;
        }
      }

    }

    float residual = -99;

    // Fit all points except layer 3, then compare expected value for layer 3 to reconstructed value
    if (nRH == 6){
      float expected = fitX(sp,se);
      residual = expected - sp(3,1);
    }

    //
    // global transformation: from Ingo Bloch
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
    //


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
      if (kRing == 1){
        hSResid11b->Fill(residual);
      }
      if (kRing == 2){
        hSResid12->Fill(residual);
      }
      if (kRing == 3){
        hSResid13->Fill(residual);
      }
      if (kRing == 4){
        hSResid11a->Fill(residual);
      }
    }
    if (kStation == 2) {
      hSCodeNarrow2->Fill(kCodeNarrow);
      hSnHits2->Fill(nhits);
      hSTheta2->Fill(theta);
      hSGlobal2->Fill(globX,globY);
      if (kRing == 1){
        hSResid21->Fill(residual);
      }
      if (kRing == 2){
        hSResid22->Fill(residual);
      }
    }
    if (kStation == 3) {
      hSCodeNarrow3->Fill(kCodeNarrow);
      hSnHits3->Fill(nhits);
      hSTheta3->Fill(theta);
      hSGlobal3->Fill(globX,globY);
      if (kRing == 1){
        hSResid31->Fill(residual);
      }
      if (kRing == 2){
        hSResid32->Fill(residual);
      }
    }
    if (kStation == 4) {
      hSCodeNarrow4->Fill(kCodeNarrow);
      hSnHits4->Fill(nhits);
      hSTheta4->Fill(theta);
      hSGlobal4->Fill(globX,globY);
      if (kRing == 1){
        hSResid41->Fill(residual);
      }
      if (kRing == 2){
        hSResid42->Fill(residual);
      }
    }
    hSnhits->Fill(nhits);
    hSChiSqProb->Fill(chisqProb);
    hSGlobalTheta->Fill(globTheta);
    hSGlobalPhi->Fill(globPhi);
  }
  hSnSegments->Fill(nSegments);


  // do Efficiency
  doEfficiencies(recHits, cscSegments);


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
          float peakADC = -1;
          vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
          for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
            diff = (float)myADCVals[iCount]-thisPedestal;
            ADC[iCount] = diff;
            if (diff > peakADC){
              peakADC = diff;
            }
          }
        }
      }

    }

  }

  timing = (ADC[2]*2 + ADC[3]*3 + ADC[4]*4 + ADC[5]*5 + ADC[6]*6)/(ADC[2] + ADC[3] + ADC[4] + ADC[5] + ADC[6]);
  return timing;


}


//---------------------------------------------------------------------------------------
// Given a set of digis, the CSCDetId, and the central strip of your choosing, returns
// the 3 time bin x 3 strip charge in the form of a matrix.  The charge matrix is centered
// on the peak charge of the center strip Will return 0's for strip if no digi is present
// (i.e. to the left of the leftmost strip in a chamber).  Charge is ped subtracted.
//---------------------------------------------------------------------------------------

HepMatrix CSCValidation::getCharge3x3(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){

  float ADC[8];
  int peakTime = -1;
  HepMatrix bcharge(3,3);
  bcharge(1,1) = 0;
  bcharge(1,2) = 0;
  bcharge(1,3) = 0;
  bcharge(2,1) = 0;
  bcharge(2,2) = 0;
  bcharge(2,3) = 0;
  bcharge(3,1) = 0;
  bcharge(3,2) = 0;
  bcharge(3,3) = 0;

  // Loop over strip digis responsible for this recHit and sum charge
  CSCStripDigiCollection::DigiRangeIterator sIt;
  CSCStripDigiCollection::DigiRangeIterator sIt2;

  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idRH){

      // First, find the peak charge in the center strip
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
          float diff = 0;
          float peakADC = -1;
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
          bcharge(2,1) = ADC[peakTime-1];
          bcharge(2,2) = ADC[peakTime];
          bcharge(2,3) = ADC[peakTime+1];
        }
      }

      // Then get the charge on the neighboring strips
      vector<CSCStripDigi>::const_iterator digiItr2 = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last2 = (*sIt).second.second;
      for ( ; digiItr2 != last2; ++digiItr2 ) {
        int thisStrip = digiItr2->getStrip();
        if (thisStrip == (centerStrip - 1)){
          float diff = 0;
          vector<int> myADCVals = digiItr2->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
          for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
            diff = (float)myADCVals[iCount]-thisPedestal;
            ADC[iCount] = diff;
          }
          bcharge(1,1) = ADC[peakTime-1];
          bcharge(1,2) = ADC[peakTime];
          bcharge(1,3) = ADC[peakTime+1];
        }

        if (thisStrip == (centerStrip + 1)){
          float diff = 0;
          vector<int> myADCVals = digiItr2->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
          for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
            diff = (float)myADCVals[iCount]-thisPedestal;
            ADC[iCount] = diff;
          }
          bcharge(3,1) = ADC[peakTime-1];
          bcharge(3,2) = ADC[peakTime];
          bcharge(3,3) = ADC[peakTime+1];
        }
      }
    }
  }

  return bcharge;
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
    AllRecHits[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()][idrec.layer() -1] = true;

  }
   

  for(CSCSegmentCollection::const_iterator segIt=cscSegments->begin(); segIt != cscSegments->end(); segIt++) {
    CSCDetId idseg  = (CSCDetId)(*segIt).cscDetId();
    //if(AllSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()]){
    //MultiSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()] = true;
    //}
    AllSegments[idseg.endcap() -1][idseg.station() -1][idseg.ring() -1][idseg.chamber()] = true;
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
              hRHSTE->AddBinContent(bin);
            }
            //---- All rechit events (normalization)
            hRHSTE->AddBinContent(10+bin);
          }
        }
      }
    }
  }

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
//
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
DEFINE_FWK_MODULE(CSCValidation);

