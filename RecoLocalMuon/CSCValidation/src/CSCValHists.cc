/*
 *  class to manage histograms for RecoLocalMuon/CSCValidation package
 *
 *  Andy Kubik
 *  Northwestern University
 */
#include "RecoLocalMuon/CSCValidation/interface/CSCValHists.h"

CSCValHists::CSCValHists(){

}

CSCValHists::~CSCValHists(){

}

void CSCValHists::bookHists(){

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
      hWireLayer11b = new TH1F("hWireLayer11b","layer wire ME11b",7,-0.5,6.5);
      hWireLayer12 = new TH1F("hWireLayer12","layer wire ME12",7,-0.5,6.5);
      hWireLayer13 = new TH1F("hWireLayer13","layer wire ME13",7,-0.5,6.5);
      hWireLayer11a = new TH1F("hWireLayer11a","layer wire ME11a",7,-0.5,6.5);
      hWireLayer21 = new TH1F("hWireLayer21","layer wire ME21",7,-0.5,6.5);
      hWireLayer22 = new TH1F("hWireLayer22","layer wire ME22",7,-0.5,6.5);
      hWireLayer31 = new TH1F("hWireLayer31","layer wire ME31",7,-0.5,6.5);
      hWireLayer32 = new TH1F("hWireLayer32","layer wire ME32",7,-0.5,6.5);
      hWireLayer41 = new TH1F("hWireLayer41","layer wire ME41",7,-0.5,6.5);
      hWireLayer42 = new TH1F("hWireLayer42","layer wire ME42",7,-0.5,6.5);
      hWireWire11b  = new TH1F("hWireWire11b","wire number ME11b",113,-0.5,112.5);
      hWireWire12  = new TH1F("hWireWire12","wire number ME12",113,-0.5,112.5);
      hWireWire13  = new TH1F("hWireWire13","wire number ME13",113,-0.5,112.5);
      hWireWire11a  = new TH1F("hWireWire11a","wire number ME11a",113,-0.5,112.5);
      hWireWire21  = new TH1F("hWireWire21","wire number ME21",113,-0.5,112.5);
      hWireWire22  = new TH1F("hWireWire22","wire number ME22",113,-0.5,112.5);
      hWireWire31  = new TH1F("hWireWire31","wire number ME31",113,-0.5,112.5);
      hWireWire32  = new TH1F("hWireWire32","wire number ME32",113,-0.5,112.5);
      hWireWire41  = new TH1F("hWireWire41","wire number ME41",113,-0.5,112.5);
      hWireWire42  = new TH1F("hWireWire42","wire number ME42",113,-0.5,112.5);

      // strip digis
      hStripAll = new TH1F("hStripAll","all strip numbers",81,-0.5,80.5);
      hStripNFired = new TH1F("hStripNFired","total number of fired strips",601,-0.5,600.5);
      hStripCodeBroad = new TH1F("hStripCodeBroad","broad scope code for strips",33,-16.5,16.5);
      hStripCodeNarrow1 = new TH1F("hStripCodeNarrow1","narrow scope strip code station 1",801,-400.5,400.5);
      hStripCodeNarrow2 = new TH1F("hStripCodeNarrow2","narrow scope strip code station 2",801,-400.5,400.5);
      hStripCodeNarrow3 = new TH1F("hStripCodeNarrow3","narrow scope strip code station 3",801,-400.5,400.5);
      hStripCodeNarrow4 = new TH1F("hStripCodeNarrow4","narrow scope strip code station 4",801,-400.5,400.5);
      hStripLayer11b = new TH1F("hStripLayer11b","layer strip ME11b",7,-0.5,6.5);
      hStripLayer12 = new TH1F("hStripLayer12","layer strip ME12",7,-0.5,6.5);
      hStripLayer13 = new TH1F("hStripLayer13","layer strip ME13",7,-0.5,6.5);
      hStripLayer11a = new TH1F("hStripLayer11a","layer strip ME11a",7,-0.5,6.5);
      hStripLayer21 = new TH1F("hStripLayer21","layer strip ME21",7,-0.5,6.5);
      hStripLayer22 = new TH1F("hStripLayer22","layer strip ME22",7,-0.5,6.5);
      hStripLayer31 = new TH1F("hStripLayer31","layer strip ME31",7,-0.5,6.5);
      hStripLayer32 = new TH1F("hStripLayer32","layer strip ME32",7,-0.5,6.5);
      hStripLayer41 = new TH1F("hStripLayer41","layer strip ME41",7,-0.5,6.5);
      hStripLayer42 = new TH1F("hStripLayer42","layer strip ME42",7,-0.5,6.5);
      hStripStrip11b  = new TH1F("hStripStrip11b","strip number ME11b",81,-0.5,80.5);
      hStripStrip12  = new TH1F("hStripStrip12","strip number ME12",81,-0.5,80.5);
      hStripStrip13  = new TH1F("hStripStrip13","strip number ME13",81,-0.5,80.5);
      hStripStrip11a  = new TH1F("hStripStrip11a","strip number ME11a",81,-0.5,80.5);
      hStripStrip21  = new TH1F("hStripStrip21","strip number ME21",81,-0.5,80.5);
      hStripStrip22  = new TH1F("hStripStrip22","strip number ME22",81,-0.5,80.5);
      hStripStrip31  = new TH1F("hStripStrip31","strip number ME31",81,-0.5,80.5);
      hStripStrip32  = new TH1F("hStripStrip32","strip number ME32",81,-0.5,80.5);
      hStripStrip41  = new TH1F("hStripStrip41","strip number ME41",81,-0.5,80.5);
      hStripStrip42  = new TH1F("hStripStrip42","strip number ME42",81,-0.5,80.5);

      //Pedestal Noise Plots
      hStripPed = new TH1F("hStripPed","Pedestal Noise Distribution",50,-25.,25.);
      hStripPedME11 = new TH1F("hStripPedME11","Pedestal Noise Distribution Chamber ME11 ",50,-25.,25.);
      hStripPedME12 = new TH1F("hStripPedME12","Pedestal Noise Distribution Chamber ME12 ",50,-25.,25.);
      hStripPedME13 = new TH1F("hStripPedME13","Pedestal Noise Distribution Chamber ME13 ",50,-25.,25.);
      hStripPedME21 = new TH1F("hStripPedME21","Pedestal Noise Distribution Chamber ME21 ",50,-25.,25.);
      hStripPedME22 = new TH1F("hStripPedME22","Pedestal Noise Distribution Chamber ME22 ",50,-25.,25.);
      hStripPedME31 = new TH1F("hStripPedME31","Pedestal Noise Distribution Chamber ME31 ",50,-25.,25.);
      hStripPedME32 = new TH1F("hStripPedME32","Pedestal Noise Distribution Chamber ME32 ",50,-25.,25.);
      hStripPedME41 = new TH1F("hStripPedME41","Pedestal Noise Distribution Chamber ME41 ",50,-25.,25.);
      hStripPedME42 = new TH1F("hStripPedME42","Pedestal Noise Distribution Chamber ME42 ",50,-25.,25.);


      // efficiency
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
      hRHLayer11b = new TH1F("hRHLayer11b","layer recHit ME11b",7,-0.5,6.5);
      hRHLayer12 = new TH1F("hRHLayer12","layer recHit ME12",7,-0.5,6.5);
      hRHLayer13 = new TH1F("hRHLayer13","layer recHit ME13",7,-0.5,6.5);
      hRHLayer11a = new TH1F("hRHLayer11a","layer recHit ME11a",7,-0.5,6.5);
      hRHLayer21 = new TH1F("hRHLayer21","layer recHit ME21",7,-0.5,6.5);
      hRHLayer22 = new TH1F("hRHLayer22","layer recHit ME22",7,-0.5,6.5);
      hRHLayer31 = new TH1F("hRHLayer31","layer recHit ME31",7,-0.5,6.5);
      hRHLayer32 = new TH1F("hRHLayer32","layer recHit ME32",7,-0.5,6.5);
      hRHLayer41 = new TH1F("hRHLayer41","layer recHit ME41",7,-0.5,6.5);
      hRHLayer42 = new TH1F("hRHLayer42","layer recHit ME42",7,-0.5,6.5);
      hRHX11b = new TH1F("hRHX11b","local X recHit ME11b",120,-60.,60.);
      hRHX12 = new TH1F("hRHX12","local X recHit ME12",120,-60.,60.);
      hRHX13 = new TH1F("hRHX13","local X recHit ME13",120,-60.,60.);
      hRHX11a = new TH1F("hRHX11a","local X recHit ME11a",120,-60.,60.);
      hRHX21 = new TH1F("hRHX21","local X recHit ME21",160,-80.,80.);
      hRHX22 = new TH1F("hRHX22","local X recHit ME22",160,-80.,80.);
      hRHX31 = new TH1F("hRHX31","local X recHit ME31",160,-80.,80.);
      hRHX32 = new TH1F("hRHX32","local X recHit ME32",160,-80.,80.);
      hRHX41 = new TH1F("hRHX41","local X recHit ME41",160,-80.,80.);
      hRHX42 = new TH1F("hRHX42","local X recHit ME42",160,-80.,80.);
      hRHY11b = new TH1F("hRHY11b","local Y recHit ME11b",50,-100.,100.);
      hRHY12 = new TH1F("hRHY12","local Y recHit ME12",50,-100.,100.);
      hRHY13 = new TH1F("hRHY13","local Y recHit ME13",50,-100.,100.);
      hRHY11a = new TH1F("hRHY11a","local Y recHit ME11a",50,-100.,100.);
      hRHY21 = new TH1F("hRHY21","local Y recHit ME21",60,-180.,180.);
      hRHY22 = new TH1F("hRHY22","local Y recHit ME22",60,-180.,180.);
      hRHY31 = new TH1F("hRHY31","local Y recHit ME31",60,-180.,180.);
      hRHY32 = new TH1F("hRHY32","local Y recHit ME32",60,-180.,180.);
      hRHY41 = new TH1F("hRHY41","local Y recHit ME41",60,-180.,180.);
      hRHY42 = new TH1F("hRHY42","local Y recHit ME42",60,-180.,180.);
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
      hRHRatioQ11b = new TH1F("hRHRatioQ11b","Ratio (Ql+Qr)/Qt (ME11b)",120,-0.1,1.1);
      hRHRatioQ12 = new TH1F("hRHRatioQ12","Ratio (Ql+Qr)/Qt (ME12)",120,-0.1,1.1);
      hRHRatioQ13 = new TH1F("hRHRatioQ13","Ratio (Ql+Qr)/Qt (ME13)",120,-0.1,1.1);
      hRHRatioQ11a = new TH1F("hRHRatioQ11a","Ratio (Ql+Qr)/Qt (ME11a)",120,-0.1,1.1);
      hRHRatioQ21 = new TH1F("hRHRatioQ21","Ratio (Ql+Qr)/Qt (ME21)",120,-0.1,1.1);
      hRHRatioQ22 = new TH1F("hRHRatioQ22","Ratio (Ql+Qr)/Qt (ME22)",120,-0.1,1.1);
      hRHRatioQ31 = new TH1F("hRHRatioQ31","Ratio (Ql+Qr)/Qt (ME31)",120,-0.1,1.1);
      hRHRatioQ32 = new TH1F("hRHRatioQ32","Ratio (Ql+Qr)/Qt (ME32)",120,-0.1,1.1);
      hRHRatioQ41 = new TH1F("hRHRatioQ41","Ratio (Ql+Qr)/Qt (ME41)",120,-0.1,1.1);
      hRHRatioQ42 = new TH1F("hRHRatioQ42","Ratio (Ql+Qr)/Qt (ME42)",120,-0.1,1.1);
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
      hRHnrechits = new TH1F("hRHnrechits","recHits per Event (all chambers)",50,0,50);


      // segments
      hSCodeBroad = new TH1F("hSCodeBroad","broad scope code for recHits",33,-16.5,16.5);
      hSCodeNarrow1 = new TH1F("hSCodeNarrow1","narrow scope Segment code station 1",801,-400.5,400.5);
      hSCodeNarrow2 = new TH1F("hSCodeNarrow2","narrow scope Segment code station 2",801,-400.5,400.5);
      hSCodeNarrow3 = new TH1F("hSCodeNarrow3","narrow scope Segment code station 3",801,-400.5,400.5);
      hSCodeNarrow4 = new TH1F("hSCodeNarrow4","narrow scope Segment code station 4",801,-400.5,400.5);
      hSnHits11b = new TH1F("hSnHits11b","N hits on Segments ME11b",7,-0.5,6.5);
      hSnHits12 = new TH1F("hSnHits12","N hits on Segments ME12",7,-0.5,6.5);
      hSnHits13 = new TH1F("hSnHits13","N hits on Segments ME13",7,-0.5,6.5);
      hSnHits11a = new TH1F("hSnHits11a","N hits on Segments ME11a",7,-0.5,6.5);
      hSnHits21 = new TH1F("hSnHits21","N hits on Segments ME21",7,-0.5,6.5);
      hSnHits22 = new TH1F("hSnHits22","N hits on Segments ME22",7,-0.5,6.5);
      hSnHits31 = new TH1F("hSnHits31","N hits on Segments ME31",7,-0.5,6.5);
      hSnHits32 = new TH1F("hSnHits32","N hits on Segments ME32",7,-0.5,6.5);
      hSnHits41 = new TH1F("hSnHits41","N hits on Segments ME41",7,-0.5,6.5);
      hSnHits42 = new TH1F("hSnHits42","N hits on Segments ME42",7,-0.5,6.5);
      hSTheta11b = new TH1F("hSTheta11b","local theta segments ME11b",128,-3.2,3.2);
      hSTheta12 = new TH1F("hSTheta12","local theta segments ME12",128,-3.2,3.2);
      hSTheta13 = new TH1F("hSTheta13","local theta segments ME13",128,-3.2,3.2);
      hSTheta11a = new TH1F("hSTheta11a","local theta segments ME11a",128,-3.2,3.2);
      hSTheta21 = new TH1F("hSTheta21","local theta segments in ME21",128,-3.2,3.2);
      hSTheta22 = new TH1F("hSTheta22","local theta segments in ME22",128,-3.2,3.2);
      hSTheta31 = new TH1F("hSTheta31","local theta segments in ME31",128,-3.2,3.2);
      hSTheta32 = new TH1F("hSTheta32","local theta segments in ME32",128,-3.2,3.2);
      hSTheta41 = new TH1F("hSTheta41","local theta segments in ME41",128,-3.2,3.2);
      hSTheta42 = new TH1F("hSTheta42","local theta segments in ME42",128,-3.2,3.2);
      hSGlobal1 = new TH2F("hSGlobal1","segment global X,Y station 1",400,-800.,800.,400,-800.,800.);
      hSGlobal2 = new TH2F("hSGlobal2","segment global X,Y station 2",400,-800.,800.,400,-800.,800.);
      hSGlobal3 = new TH2F("hSGlobal3","segment global X,Y station 3",400,-800.,800.,400,-800.,800.);
      hSGlobal4 = new TH2F("hSGlobal4","segment global X,Y station 4",400,-800.,800.,400,-800.,800.);
      hSnhits = new TH1F("hSnhits","N hits on Segments",7,-0.5,6.5);
      hSChiSqProb = new TH1F("hSChiSqProb","segments chi-squared probability",100,0.,1.);
      hSGlobalTheta = new TH1F("hSGlobalTheta","segment global theta",64,0,1.6);
      hSGlobalPhi   = new TH1F("hSGlobalPhi",  "segment global phi",  128,-3.2,3.2);
      hSnSegments   = new TH1F("hSnSegments","number of segments per event",11,-0.5,10.5);
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

      //occupancy plots
      hOWires = new TH2I("hOWires","Wire Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOStrips = new TH2I("hOStrips","Strip Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
      hORecHits = new TH2I("hORecHits","RecHit Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOSegments = new TH2I("hOSegments","Segment Occupancy",36,0.5,36.5,20,0.5,20.5);

}

void CSCValHists::writeHists(TFile* theFile, bool isSimulation){

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
      hWireLayer11b->Write();
      hWireLayer12->Write();
      hWireLayer13->Write();
      hWireLayer11a->Write();
      hWireLayer21->Write();
      hWireLayer22->Write();
      hWireLayer31->Write();
      hWireLayer32->Write();
      hWireLayer41->Write();
      hWireLayer42->Write();
      hWireWire11b->Write();
      hWireWire12->Write();
      hWireWire13->Write();
      hWireWire11a->Write();
      hWireWire21->Write();
      hWireWire22->Write();
      hWireWire31->Write();
      hWireWire32->Write();
      hWireWire41->Write();
      hWireWire42->Write();
      hOWires->Write();

      // strip digis
      hStripAll->Write();
      hStripNFired->Write();
      hStripCodeBroad->Write();
      hStripCodeNarrow1->Write();
      hStripCodeNarrow2->Write();
      hStripCodeNarrow3->Write();
      hStripCodeNarrow4->Write();
      hStripLayer11b->Write();
      hStripLayer12->Write();
      hStripLayer13->Write();
      hStripLayer11a->Write();
      hStripLayer21->Write();
      hStripLayer22->Write();
      hStripLayer31->Write();
      hStripLayer32->Write();
      hStripLayer41->Write();
      hStripLayer42->Write();
      hStripStrip11b->Write();
      hStripStrip12->Write();
      hStripStrip13->Write();
      hStripStrip11a->Write();
      hStripStrip21->Write();
      hStripStrip22->Write();
      hStripStrip31->Write();
      hStripStrip32->Write();
      hStripStrip41->Write();
      hStripStrip42->Write();
      hOStrips->Write();
      theFile->cd();

      //Pedestal Noise
      theFile->cd("PedestalNoise");
      hStripPed->Write();
      hStripPedME11->Write();
      hStripPedME12->Write();
      hStripPedME13->Write();
      hStripPedME21->Write();
      hStripPedME22->Write();
      hStripPedME31->Write();
      hStripPedME32->Write();
      hStripPedME41->Write();
      hStripPedME42->Write();
      theFile->cd();

      // recHits
      theFile->cd("recHits");
      hRHCodeBroad->Write();
      hRHCodeNarrow1->Write();
      hRHCodeNarrow2->Write();
      hRHCodeNarrow3->Write();
      hRHCodeNarrow4->Write();
      hRHLayer11a->Write();
      hRHLayer12->Write();
      hRHLayer13->Write();
      hRHLayer11b->Write();
      hRHLayer21->Write();
      hRHLayer22->Write();
      hRHLayer31->Write();
      hRHLayer32->Write();
      hRHLayer41->Write();
      hRHLayer42->Write();
      hRHX11b->Write();
      hRHX12->Write();
      hRHX13->Write();
      hRHX11a->Write();
      hRHX21->Write();
      hRHX22->Write();
      hRHX31->Write();
      hRHX32->Write();
      hRHX41->Write();
      hRHX42->Write();
      hRHY11b->Write();
      hRHY12->Write();
      hRHY13->Write();
      hRHY11a->Write();
      hRHY21->Write();
      hRHY22->Write();
      hRHY31->Write();
      hRHY32->Write();
      hRHY41->Write();
      hRHY42->Write();
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
      hRHnrechits->Write();
      rHTree->Write();
      hORecHits->Write();
      theFile->cd();

      // segments
      theFile->cd("Segments");
      hSCodeBroad->Write();
      hSCodeNarrow1->Write();
      hSCodeNarrow2->Write();
      hSCodeNarrow3->Write();
      hSCodeNarrow4->Write();
      hSnHits11b->Write();
      hSnHits12->Write();
      hSnHits13->Write();
      hSnHits11a->Write();
      hSnHits21->Write();
      hSnHits22->Write();
      hSnHits31->Write();
      hSnHits32->Write();
      hSnHits41->Write();
      hSnHits42->Write();
      hSTheta11b->Write();
      hSTheta12->Write();
      hSTheta13->Write();
      hSTheta11a->Write();
      hSTheta21->Write();
      hSTheta22->Write();
      hSTheta31->Write();
      hSTheta32->Write();
      hSTheta41->Write();
      hSTheta42->Write();
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
      hOSegments->Write();
      segTree->Write();
      theFile->cd();

      //
      theFile->Close();

}

void CSCValHists::setupTrees(){

        // Create the root tree to hold position info
      rHTree  = new TTree("rHPositions","Local and Global reconstructed positions for recHits");
      segTree = new TTree("segPositions","Local and Global reconstructed positions for segments");

      // Create a branch on the tree
      rHTree->Branch("rHpos",&rHpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");
      segTree->Branch("segpos",&segpos,"endcap/I:station/I:ring/I:chamber/I:layer/I:localx/F:localy/F:globalx/F:globaly/F");

}

void CSCValHists::fillWireHistos(int wire, int TB, int codeN, int codeB,
                                 int en, int st, int ri, int ch, int la){

      hWireTBinAll->Fill(TB);
      hWireAll->Fill(wire);
      hWireCodeBroad->Fill(codeB);
      if (st == 1){
        hWireCodeNarrow1->Fill(codeN);
        if (ri == 1){
          hWireLayer11b->Fill(la);
          hWireWire11b->Fill(wire);
        }
        if (ri == 2){
          hWireLayer12->Fill(la);
          hWireWire12->Fill(wire);
        }
        if (ri == 3){
          hWireLayer13->Fill(la);
          hWireWire13->Fill(wire);
        }
        if (ri == 4){
          hWireLayer11a->Fill(la);
          hWireWire11a->Fill(wire);
        }
      }
      if (st == 2){
        hWireCodeNarrow2->Fill(codeN);
        if (ri == 1){
          hWireLayer21->Fill(la);
          hWireWire21->Fill(wire);
        }
        if (ri == 2){
          hWireLayer22->Fill(la);
          hWireWire22->Fill(wire);
        }
      }
      if (st == 3){
        hWireCodeNarrow3->Fill(codeN);
        if (ri == 1){
          hWireLayer31->Fill(la);
          hWireWire31->Fill(wire);
        }
        if (ri == 2){
          hWireLayer32->Fill(la);
          hWireWire32->Fill(wire);
        }
      }
      if (st == 4){
        hWireCodeNarrow4->Fill(codeN);
        if (ri == 1){
          hWireLayer41->Fill(la);
          hWireWire41->Fill(wire);
        }
        if (ri == 2){
          hWireLayer42->Fill(la);
          hWireWire42->Fill(wire);
        }
      }
}


void CSCValHists::fillStripHistos(int strip, int codeN, int codeB,
                                  int en, int st, int ri, int ch, int la){

      hStripAll->Fill(strip);
      hStripCodeBroad->Fill(codeB);
      if (st == 1){
        hStripCodeNarrow1->Fill(codeN);
        if (ri == 1){
          hStripLayer11b->Fill(la);
          hStripStrip11b->Fill(strip);
        }
        if (ri == 2){
          hStripLayer12->Fill(la);
          hStripStrip12->Fill(strip);
        }
        if (ri == 3){
          hStripLayer13->Fill(la);
          hStripStrip13->Fill(strip);
        }
        if (ri == 4){
          hStripLayer11a->Fill(la);
          hStripStrip11a->Fill(strip);
        }
      }
      if (st == 2){
        if (ri == 1){
          hStripLayer21->Fill(la);
          hStripStrip21->Fill(strip);
        }
        if (ri == 2){
          hStripLayer22->Fill(la);
          hStripStrip22->Fill(strip);
        }
      }
      if (st == 3){
        if (ri == 1){
          hStripLayer31->Fill(la);
          hStripStrip31->Fill(strip);
        }
        if (ri == 2){
          hStripLayer32->Fill(la);
          hStripStrip32->Fill(strip);
        }
      }
      if (st == 4){
        if (ri == 1){
          hStripLayer41->Fill(la);
          hStripStrip41->Fill(strip);
        }
        if (ri == 2){
          hStripLayer42->Fill(la);
          hStripStrip42->Fill(strip);
        }
      }

}

void CSCValHists::fillNoiseHistos(float ADC, int  globalStrip, int kStation, int kRing){

	hStripPed->Fill(ADC);
	if (kStation == 1 && kRing == 1 ) {
	  hStripPedME11->Fill(ADC);
	}
	if (kStation == 1 && kRing == 2) {
	  hStripPedME12->Fill(ADC);
	}
	if (kStation == 1 && kRing == 3) {
	  hStripPedME13->Fill(ADC);
	}
	if (kStation == 2 && kRing == 1) {
	  hStripPedME21->Fill(ADC);
	}
	if (kStation == 2 && kRing == 2) {
	  hStripPedME22->Fill(ADC);
	}
        if (kStation == 3 && kRing == 1) {
	  hStripPedME31->Fill(ADC);
	}
	if (kStation == 3 && kRing == 2) {
	  hStripPedME32->Fill(ADC);
	}
        if (kStation == 4 && kRing == 1) {
	  hStripPedME41->Fill(ADC);
	}
	if (kStation == 4 && kRing == 2) {
	  hStripPedME42->Fill(ADC);
	}

}

void CSCValHists::fillCalibHistos(float gain, float xl, float xr, float xil, float xir, float pedp,
                                  float pedr, float n33, float n34, float n35, float n44, float n45,
                                  float n46, float n55, float n56, float n57, float n66, float n67,
                                  float n77, int bin){

      hCalibGainsS->SetBinContent(bin,gain);
      hCalibXtalkSL->SetBinContent(bin,xl);
      hCalibXtalkSR->SetBinContent(bin,xr);
      hCalibXtalkIL->SetBinContent(bin,xil);
      hCalibXtalkIR->SetBinContent(bin,xir);
      hCalibPedsP->SetBinContent(bin,pedp);
      hCalibPedsR->SetBinContent(bin,pedr);
      hCalibNoise33->SetBinContent(bin,n33);
      hCalibNoise34->SetBinContent(bin,n34);
      hCalibNoise35->SetBinContent(bin,n35);
      hCalibNoise44->SetBinContent(bin,n44);
      hCalibNoise45->SetBinContent(bin,n45);
      hCalibNoise46->SetBinContent(bin,n46);
      hCalibNoise55->SetBinContent(bin,n55);
      hCalibNoise56->SetBinContent(bin,n56);
      hCalibNoise57->SetBinContent(bin,n57);
      hCalibNoise66->SetBinContent(bin,n66);
      hCalibNoise67->SetBinContent(bin,n67);
      hCalibNoise77->SetBinContent(bin,n77);

}


void CSCValHists::fillRechitHistos(int codeN, int codeB, float x, float y, float gx, float gy,
                                   float sQ, float rQ, float time, float simres,
                                   int en, int st, int ri, int ch, int la){

      hRHCodeBroad->Fill(codeB);
      if (st == 1){
        hRHCodeNarrow1->Fill(codeN);
        hRHGlobal1->Fill(gx,gy);
        if (ri == 1){
          hRHLayer11b->Fill(la);
          hRHX11b->Fill(x);
          hRHY11b->Fill(y);
          hRHResid11b->Fill(simres);
          hRHSumQ11b->Fill(sQ);
          hRHRatioQ11b->Fill(rQ);
          hRHTiming11b->Fill(time);
        }
        if (ri == 2){
          hRHLayer12->Fill(la);
          hRHX12->Fill(x);
          hRHY12->Fill(y);
          hRHResid12->Fill(simres);
          hRHSumQ12->Fill(sQ);
          hRHRatioQ12->Fill(rQ);
          hRHTiming12->Fill(time);
        }
        if (ri == 3){
          hRHLayer13->Fill(la);
          hRHX13->Fill(x);
          hRHY13->Fill(y);
          hRHResid13->Fill(simres);
          hRHSumQ13->Fill(sQ);
          hRHRatioQ13->Fill(rQ);
          hRHTiming13->Fill(time);
        }
        if (ri == 4){
          hRHLayer11a->Fill(la);
          hRHX11a->Fill(x);
          hRHY11a->Fill(y);
          hRHResid11a->Fill(simres);
          hRHSumQ11a->Fill(sQ);
          hRHRatioQ11a->Fill(rQ);
          hRHTiming11a->Fill(time);
        }
      }
      if (st == 2){
        hRHCodeNarrow2->Fill(codeN);
        hRHGlobal2->Fill(gx,gy);
        if (ri == 1){
          hRHLayer21->Fill(la);
          hRHX21->Fill(x);
          hRHY21->Fill(y);
          hRHResid21->Fill(simres);
          hRHSumQ21->Fill(sQ);
          hRHRatioQ21->Fill(rQ);
          hRHTiming21->Fill(time);
        }
        if (ri == 2){
          hRHLayer22->Fill(la);
          hRHX22->Fill(x);
          hRHY22->Fill(y);
          hRHResid22->Fill(simres);
          hRHSumQ22->Fill(sQ);
          hRHRatioQ22->Fill(rQ);
          hRHTiming22->Fill(time);
        }
      }
      if (st == 3){
        hRHCodeNarrow3->Fill(codeN);
        hRHGlobal3->Fill(gx,gy);
        if (ri == 1){
          hRHLayer31->Fill(la);
          hRHX31->Fill(x);
          hRHY31->Fill(y);
          hRHResid31->Fill(simres);
          hRHSumQ31->Fill(sQ);
          hRHRatioQ31->Fill(rQ);
          hRHTiming31->Fill(time);
        }
        if (ri == 2){
          hRHLayer32->Fill(la);
          hRHX32->Fill(x);
          hRHY32->Fill(y);
          hRHResid32->Fill(simres);
          hRHSumQ32->Fill(sQ);
          hRHRatioQ32->Fill(rQ);
          hRHTiming32->Fill(time);
        }
      }
      if (st == 4){
        hRHCodeNarrow4->Fill(codeN);
        hRHGlobal4->Fill(gx,gy);
        if (ri == 1){
          hRHLayer41->Fill(la);
          hRHX41->Fill(x);
          hRHY41->Fill(y);
          hRHResid41->Fill(simres);
          hRHSumQ41->Fill(sQ);
          hRHRatioQ41->Fill(rQ);
          hRHTiming41->Fill(time);
        }
        if (ri == 2){
          hRHLayer42->Fill(la);
          hRHX42->Fill(x);
          hRHY42->Fill(y);
          hRHResid42->Fill(simres);
          hRHSumQ42->Fill(sQ);
          hRHRatioQ42->Fill(rQ);
          hRHTiming42->Fill(time);
        }
      }

}


void CSCValHists::fillRechitTree(float x, float y, float gx, float gy,
                                 int en, int st, int ri, int ch, int la){
      // Fill the rechit position branch
      rHpos.localx  = x;
      rHpos.localy  = y;
      rHpos.globalx = gx;
      rHpos.globaly = gy;
      rHpos.endcap  = en;
      rHpos.ring    = ri;
      rHpos.station = st;
      rHpos.chamber = ch;
      rHpos.layer   = la;
      rHTree->Fill();

}


void CSCValHists::fillSegmentHistos(int codeN, int codeB, int nhits, float theta, float gx, float gy,
                                    float resid, float chi2p, float gTheta, float gPhi,
                                    int en, int st, int ri, int ch){

      hSCodeBroad->Fill(codeB);
      hSChiSqProb->Fill(chi2p);
      hSGlobalTheta->Fill(gTheta);
      hSGlobalPhi->Fill(gPhi);
      hSnhits->Fill(nhits);
      if (st == 1){
        hSCodeNarrow1->Fill(codeN);
        hSGlobal1->Fill(gx,gy);
        if (ri == 1){
          hSnHits11b->Fill(nhits);
          hSTheta11b->Fill(theta);
          hSResid11b->Fill(resid);
        }
        if (ri == 2){
          hSnHits12->Fill(nhits);
          hSTheta12->Fill(theta);
          hSResid12->Fill(resid);
        }
        if (ri == 3){
          hSnHits13->Fill(nhits);
          hSTheta13->Fill(theta);
          hSResid13->Fill(resid);
        }
        if (ri == 4){
          hSnHits11a->Fill(nhits);
          hSTheta11a->Fill(theta);
          hSResid11a->Fill(resid);
        }
      }
      if (st == 2){
        hSCodeNarrow2->Fill(codeN);
        hSGlobal2->Fill(gx,gy);
        if (ri == 1){
          hSnHits21->Fill(nhits);
          hSTheta21->Fill(theta);
          hSResid21->Fill(resid);
        }
        if (ri == 2){
          hSnHits22->Fill(nhits);
          hSTheta22->Fill(theta);
          hSResid22->Fill(resid);
        }
      }
      if (st == 3){
        hSCodeNarrow3->Fill(codeN);
        hSGlobal3->Fill(gx,gy);
        if (ri == 1){
          hSnHits31->Fill(nhits);
          hSTheta31->Fill(theta);
          hSResid31->Fill(resid);
        }
        if (ri == 2){
          hSnHits32->Fill(nhits);
          hSTheta32->Fill(theta);
          hSResid32->Fill(resid);
        }
      }
      if (st == 4){
        hSCodeNarrow4->Fill(codeN);
        hSGlobal4->Fill(gx,gy);
        if (ri == 1){
          hSnHits41->Fill(nhits);
          hSTheta41->Fill(theta);
          hSResid41->Fill(resid);
        }
        if (ri == 2){
          hSnHits42->Fill(nhits);
          hSTheta42->Fill(theta);
          hSResid42->Fill(resid);
        }
      }

}


void CSCValHists::fillSegmentTree(float x, float y, float gx, float gy, int en,
                                  int st, int ri, int ch){

      // Fill the segment position branch
      segpos.localx  = x;
      segpos.localy  = y;
      segpos.globalx = gx;
      segpos.globaly = gy;
      segpos.endcap  = en;
      segpos.ring    = ri;
      segpos.station = st;
      segpos.chamber = ch;
      segpos.layer   = 0;
      segTree->Fill();

}

void CSCValHists::fillEventHistos(int nWire, int nStrip, int nrH, int nSeg){

      if (nWire != 0) hWirenGroupsTotal->Fill(nWire);
      if (nStrip != 0) hStripNFired->Fill(nStrip);
      if (nSeg != 0) hSnSegments->Fill(nSeg);
      if (nrH != 0) hRHnrechits->Fill(nrH);

}

void CSCValHists::fillOccupancyHistos(const bool wo[2][4][4][36], const bool sto[2][4][4][36],
                                      const bool ro[2][4][4][36], const bool so[2][4][4][36]){

  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          int type = 0;
          if ((s+1) == 1) type = (r+1);
          else type = (s+1)*2 + (r+1);
          if ((e+1) == 1) type = type + 10;
          if ((e+1) == 2) type = 11 - type;
          //int bin = hOWires->GetBin(chamber,type);
          //hOWires->AddBinContent(bin);
          if (wo[e][s][r][c]) hOWires->Fill((c+1),type);
          if (sto[e][s][r][c]) hOStrips->Fill((c+1),type);
          if (ro[e][s][r][c]) hORecHits->Fill((c+1),type);
          if (so[e][s][r][c]) hOSegments->Fill((c+1),type);
        }
      }
    }
  }

}


void CSCValHists::fillEfficiencyHistos(int bin, int flag){

      if (flag == 1) hSSTE->AddBinContent(bin);
      if (flag == 2) hRHSTE->AddBinContent(bin);

}

void CSCValHists::getEfficiency(float bin, float Norm, std::vector<float> &eff){
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

void CSCValHists::histoEfficiency(TH1F *readHisto, TH1F *writeHisto){
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

