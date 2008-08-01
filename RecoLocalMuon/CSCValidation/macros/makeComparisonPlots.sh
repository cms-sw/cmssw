#!/bin/bash

# to run this script, do
# ./makeComparisonPlots.sh <filepath_new> <filepath_ref> <data_type>
# where <filepath_new> and <filepath_ref> are the paths to the output root files from CSCValiation
# data_type is an int (1 = data ; 2 = MC)

# example:  ./makeComparisonPlots.sh CMSSW_1_8_0_pre8/src/RecoLocalMuon/CSCValidation/test/ CMSSW_1_7_4/src/RecoLocalMuon/CSCValidation/test/ 2

ARG1=$1
ARG2=$2
ARG3=$3

MACRO=makePlots.C
cat > ${MACRO}<<EOF

{
  gROOT->Reset();
  gROOT->ProcessLine(".L myFunctions.C");

  std::string newReleasePath = "${ARG1}";
  std::string refReleasePath = "${ARG2}";
  int datatype = ${ARG3};              // 1 = data, 2 = mc

  TFile *f1;
  TFile *f2;

  f1 = OpenFiles(refReleasePath,datatype);
  f2 = OpenFiles(newReleasePath,datatype);

  //procuce wire and strip digi comparison plots
  Compare1DPlots2("Digis/hStripAll","Digis/hWireAll",f1,f2,"Strip Numbers Fired (All Chambers)","Wire Groups Fired (All Chambers)","digi_stripswires_all.gif");
  Compare1DPlots2("Digis/hStripCodeBroad","Digis/hWireCodeBroad",f1,f2,"hStripCodeBroad","hWireCodeBroad","digi_stripswires_hCodeBroad.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow1","Digis/hWireCodeNarrow1",f1,f2,"hStripCodeNarrow (Station 1)","hWireCodeNarrow (Station 1)","digi_stripswires_hCodeNarrow1.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow2","Digis/hWireCodeNarrow2",f1,f2,"hStripCodeNarrow (Station 2)","hWireCodeNarrow (Station 2)","digi_stripswires_hCodeNarrow2.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow3","Digis/hWireCodeNarrow3",f1,f2,"hStripCodeNarrow (Station 3)","hWireCodeNarrow (Station 3)","digi_stripswires_hCodeNarrow3.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow4","Digis/hWireCodeNarrow4",f1,f2,"hStripCodeNarrow (Station 4)","hWireCodeNarrow (Station 4)","digi_stripswires_hCodeNarrow4.gif");
  Compare1DPlots2("Digis/hStripLayer1","Digis/hWireLayer1",f1,f2,"Strips Fired per Layer(Station 1)","Wires Fired per Layer(Station 1)","digi_stripswires_perlayer1.gif");
  Compare1DPlots2("Digis/hStripLayer2","Digis/hWireLayer2",f1,f2,"Strips Fired per Layer(Station 2)","Wires Fired per Layer(Station 2)","digi_stripswires_perlayer2.gif");
  Compare1DPlots2("Digis/hStripLayer3","Digis/hWireLayer3",f1,f2,"Strips Fired per Layer(Station 3)","Wires Fired per Layer(Station 3)","digi_stripswires_perlayer3.gif");
  Compare1DPlots2("Digis/hStripLayer4","Digis/hWireLayer4",f1,f2,"Strips Fired per Layer(Station 4)","Wires Fired per Layer(Station 4)","digi_stripswires_perlayer4.gif");
  Compare1DPlots2("Digis/hStripNFired","Digis/hWirenGroupsTotal",f1,f2,"Number of Fired Strips per Event","Number of Fired Wiregroups per Event","digi_stripswires_perevent.gif");
  Compare1DPlots2("Digis/hStripStrip1","Digis/hWireWire1",f1,f2,"Strip Numbers Fired (Station 1)","Wiregroup Numbers Fired (Station 1)","digi_stripswires_1.gif");
  Compare1DPlots2("Digis/hStripStrip2","Digis/hWireWire2",f1,f2,"Strip Numbers Fired (Station 2)","Wiregroup Numbers Fired (Station 2)","digi_stripswires_2.gif");
  Compare1DPlots2("Digis/hStripStrip3","Digis/hWireWire3",f1,f2,"Strip Numbers Fired (Station 3)","Wiregroup Numbers Fired (Station 3)","digi_stripswires_3.gif");
  Compare1DPlots2("Digis/hStripStrip4","Digis/hWireWire4",f1,f2,"Strip Numbers Fired (Station 4)","Wiregroup Numbers Fired (Station 4)","digi_stripswires_4.gif");
  Compare1DPlots1("Digis/hStripADCAll",f1,f2,"All ADC Values Above Cutoff","digi_stripadcs.gif");
  Compare1DPlots1("Digis/hWireTBinAll",f1,f2,"Signal Time Bin for All Wires","digi_wireTB.gif");


  //produce rechit comparison plots
  Compare1DPlots1("recHits/hRHCodeBroad",f1,f2,"hRHCodeBroad","rH_hRHCodeBroad.gif"); 
  Compare1DPlots2("recHits/hRHCodeNarrow1","recHits/hRHCodeNarrow2",f1,f2,"hRHCodeNarrow (Station 1)","hRHCodeNarrow (Station 2)","rH_hRHCodeNarrow_1a2.gif");
  Compare1DPlots2("recHits/hRHCodeNarrow3","recHits/hRHCodeNarrow4",f1,f2,"hRHCodeNarrow (Station 3)","hRHCodeNarrow (Station 4)","rH_hRHCodeNarrow_3a4.gif");
  Compare1DPlots2("recHits/hRHX1","recHits/hRHY1",f1,f2,"recHits, LocalX, Station 1","recHits, LocalY, Station 1","rH_local_pos_station1.gif");
  Compare1DPlots2("recHits/hRHX2","recHits/hRHY2",f1,f2,"recHits, LocalX, Station 2","recHits, LocalY, Station 2","rH_local_pos_station2.gif");
  Compare1DPlots2("recHits/hRHX3","recHits/hRHY3",f1,f2,"recHits, LocalX, Station 3","recHits, LocalY, Station 3","rH_local_pos_station3.gif");
  Compare1DPlots2("recHits/hRHX4","recHits/hRHY4",f1,f2,"recHits, LocalX, Station 4","recHits, LocalY, Station 4","rH_local_pos_station4.gif");
  if (datatype == 2){
    Compare1DPlots2("recHits/hRHResid11a","recHits/hRHResid11b",f1,f2,"SimHit X - Reco X (ME11a)","SimHit X - Reco X (ME11b)","rH_resid_ME11.gif");
    Compare1DPlots2("recHits/hRHResid12","recHits/hRHResid13",f1,f2,"SimHit X - Reco X (ME12)","SimHit X - Reco X (ME13)","rH_sH_resid_ME12_ME13.gif"); 
    Compare1DPlots2("recHits/hRHResid21","recHits/hRHResid22",f1,f2,"SimHit X - Reco X (ME21)","SimHit X - Reco X (ME22)","rH_sH_resid_ME21_ME22.gif"); 
    Compare1DPlots2("recHits/hRHResid31","recHits/hRHResid32",f1,f2,"SimHit X - Reco X (ME31)","SimHit X - Reco X (ME32)","rH_sH_resid_ME31_ME32.gif"); 
    Compare1DPlots2("recHits/hRHResid41","recHits/hRHResid42",f1,f2,"SimHit X - Reco X (ME41)","SimHit X - Reco X (ME42)","rH_sH_resid_ME41_ME42.gif"); 
  }
  Compare1DPlots2("recHits/hRHLayer1","recHits/hRHLayer2",f1,f2,"recHits in a Layer, Station 1","recHits in a Layer, Station 2","rH_per_layer_stations1and2.gif");
  Compare1DPlots2("recHits/hRHLayer3","recHits/hRHLayer4",f1,f2,"recHits in a Layer, Station 3","recHits in a Layer, Station 4","rH_per_layer_stations3and4.gif");
  Compare1DPlots2("recHits/hSResid11a","recHits/hSResid11b",f1,f2,"Fitted Position on Strip - Reco Pos (ME11a)","Fitted Position on Strip - Reco Pos (ME11b)","rH_fit_resid_ME11.gif");
  Compare1DPlots2("recHits/hSResid12","recHits/hSResid13",f1,f2,"Fitted Position on Strip - Reco Pos (ME12)","Fitted Position on Strip - Reco Pos (ME13)","rH_fit_resid_ME12_ME13.gif");
  Compare1DPlots2("recHits/hSResid21","recHits/hSResid22",f1,f2,"Fitted Position on Strip - Reco Pos (ME21)","Fitted Position on Strip - Reco Pos (ME22)","rH_fit_resid_ME21_ME22.gif");
  Compare1DPlots2("recHits/hSResid31","recHits/hSResid32",f1,f2,"Fitted Position on Strip - Reco Pos (ME31)","Fitted Position on Strip - Reco Pos (ME32)","rH_fit_resid_ME31_ME32.gif");
  Compare1DPlots2("recHits/hSResid41","recHits/hSResid42",f1,f2,"Fitted Position on Strip - Reco Pos (ME41)","Fitted Position on Strip - Reco Pos (ME42)","rH_fit_resid_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHSumQ11a","recHits/hRHSumQ11b",f1,f2,"Sum 3 strip x 3 time bin charge (ME11a)","Sum 3 strip x 3 time bin charge (ME11b)","rH_sumQ_ME11.gif");
  Compare1DPlots2("recHits/hRHSumQ12","recHits/hRHSumQ13",f1,f2,"Sum 3 strip x 3 time bin charge (ME12)","Sum 3 strip x 3 time bin charge (ME13)","rH_sumQ_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHSumQ21","recHits/hRHSumQ22",f1,f2,"Sum 3 strip x 3 time bin charge (ME21)","Sum 3 strip x 3 time bin charge (ME22)","rH_sumQ_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHSumQ31","recHits/hRHSumQ32",f1,f2,"Sum 3 strip x 3 time bin charge (ME31)","Sum 3 strip x 3 time bin charge (ME32)","rH_sumQ_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHSumQ41","recHits/hRHSumQ42",f1,f2,"Sum 3 strip x 3 time bin charge (ME41)","Sum 3 strip x 3 time bin charge (ME42)","rH_sumQ_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHRatioQ11a","recHits/hRHRatioQ11b",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME11a)","Charge Ratio (Ql+Qr)/Qc (ME11b)","rH_ratioQ_ME11.gif");
  Compare1DPlots2("recHits/hRHRatioQ12","recHits/hRHRatioQ13",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME12)","Charge Ratio (Ql+Qr)/Qc (ME13)","rH_ratioQ_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHRatioQ21","recHits/hRHRatioQ22",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME21)","Charge Ratio (Ql+Qr)/Qc (ME22)","rH_ratioQ_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHRatioQ31","recHits/hRHRatioQ32",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME31)","Charge Ratio (Ql+Qr)/Qc (ME32)","rH_ratioQ_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHRatioQ41","recHits/hRHRatioQ42",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME41)","Charge Ratio (Ql+Qr)/Qc (ME42)","rH_ratioQ_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHTiming11a","recHits/hRHTiming11b",f1,f2,"recHit Timing from Strip (ME11a)","recHit Timing from Strip (ME11b)","rH_timing_ME11.gif");
  Compare1DPlots2("recHits/hRHTiming12","recHits/hRHTiming13",f1,f2,"recHit Timing from Strip (ME12)","recHit Timing from Strip (ME13)","rH_timing_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHTiming21","recHits/hRHTiming22",f1,f2,"recHit Timing from Strip (ME21)","recHit Timing from Strip (ME22)","rH_timing_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHTiming31","recHits/hRHTiming32",f1,f2,"recHit Timing from Strip (ME31)","recHit Timing from Strip (ME32)","rH_timing_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHTiming41","recHits/hRHTiming42",f1,f2,"recHit Timing from Strip (ME41)","recHit Timing from Strip (ME42)","rH_timing_ME41_ME42.gif");


  //produce segment comparison plots
  Compare1DPlots2("Segments/hSCodeNarrow1","Segments/hSCodeNarrow2",f1,f2,"hSCodeNarrow (Station 1)","hSCodeNarrow (Station 2)","seg_hSCodeNarrow_1a2.gif");
  Compare1DPlots2("Segments/hSCodeNarrow3","Segments/hSCodeNarrow4",f1,f2,"hSCodeNarrow (Station 3)","hSCodeNarrow (Station 4)","seg_hSCodeNarrow_3a4.gif");
  Compare1DPlots1("Segments/hSCodeBroad",f1,f2,"hSCodeBroad","seg_hSCodeBroad.gif");
  Compare1DPlots2("Segments/hSGlobalPhi","Segments/hSGlobalTheta",f1,f2,"Segment Global Phi (all stations)","Segment Global Theta (all stations)","seg_globthetaphi.gif");
  Compare1DPlots2("Segments/hSTheta1","Segments/hSTheta2",f1,f2,"Segment Local Theta (Station 1)","Segment Local Theta (Station 2)","seg_localtheta_1a2.gif");
  Compare1DPlots2("Segments/hSTheta3","Segments/hSTheta4",f1,f2,"Segment Local Theta (Station 3)","Segment Local Theta (Station 4)","seg_localtheta_3a4.gif");
  Compare1DPlots2("Segments/hSnHits1","Segments/hSnHits2",f1,f2,"recHits per Segment (Station 1)","recHits per Segment (Station 2)","seg_nhits_1a2.gif");
  Compare1DPlots2("Segments/hSnHits3","Segments/hSnHits4",f1,f2,"recHits per Segment (Station 3)","recHits per Segment (Station 4)","seg_nhits_3a4.gif");
  Compare1DPlots2("Segments/hSnSegments","Segments/hSnhits",f1,f2,"Segments per Event","recHits per Segment (all stations)","seg_nhits_all.gif");

 
  //produce efficiency plots
  Compare1DPlots2("recHits/hRHEff","Segments/hSEff",f1,f2,"recHit Efficiency","Segment Efficiency","efficiency.gif");


  //Make global position graphs from trees
  GlobalrHPosfromTree("Global recHit positions (Station 1)",f1,f2,1,"rH_global_pos_station1.gif");
  GlobalrHPosfromTree("Global recHit positions (Station 2)",f1,f2,2,"rH_global_pos_station2.gif");
  GlobalrHPosfromTree("Global recHit positions (Station 3)",f1,f2,3,"rH_global_pos_station3.gif");
  GlobalrHPosfromTree("Global recHit positions (Station 4)",f1,f2,4,"rH_global_pos_station4.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 1)",f1,f2,1,"seg_global_pos_station1.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 2)",f1,f2,2,"seg_global_pos_station2.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 3)",f1,f2,3,"seg_global_pos_station3.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 4)",f1,f2,4,"seg_global_pos_station4.gif");


}

EOF

root -l -q ${MACRO}

rm makePlots.C

