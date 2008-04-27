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

  Compare1DPlots2("Digis/hStripAll","Digis/hWireAll",f1,f2,"Strip Numbers Fired (All Chambers)","Wire Groups Fired (All Chambers)","digi_stripswires_all.gif");
  Compare1DPlots2("Digis/hStripCodeBroad","Digis/hWireCodeBroad",f1,f2,"hStripCodeBroad","hWireCodeBroad","digi_stripswires_hCodeBroad.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow1","Digis/hWireCodeNarrow1",f1,f2,"hStripCodeNarrow (Station 1)","hWireCodeNarrow (Station 1)","digi_stripswires_hCodeNarrow1.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow2","Digis/hWireCodeNarrow2",f1,f2,"hStripCodeNarrow (Station 2)","hWireCodeNarrow (Station 2)","digi_stripswires_hCodeNarrow2.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow3","Digis/hWireCodeNarrow3",f1,f2,"hStripCodeNarrow (Station 3)","hWireCodeNarrow (Station 3)","digi_stripswires_hCodeNarrow3.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow4","Digis/hWireCodeNarrow4",f1,f2,"hStripCodeNarrow (Station 4)","hWireCodeNarrow (Station 4)","digi_stripswires_hCodeNarrow4.gif");
  Compare1DPlots2("Digis/hStripLayer+14","Digis/hWireLayer+14",f1,f2,"Strips Fired per Layer(ME 11a)","Wires Fired per Layer(ME 11a)","digi_stripswires_perlayer11a.gif");
  Compare1DPlots2("Digis/hStripLayer+11","Digis/hWireLayer+11",f1,f2,"Strips Fired per Layer(ME 11b)","Wires Fired per Layer(ME 11b)","digi_stripswires_perlayer11b.gif");
  Compare1DPlots2("Digis/hStripLayer+12","Digis/hWireLayer+12",f1,f2,"Strips Fired per Layer(ME 12)","Wires Fired per Layer(ME 12)","digi_stripswires_perlayer12.gif");
  Compare1DPlots2("Digis/hStripLayer+13","Digis/hWireLayer+13",f1,f2,"Strips Fired per Layer(ME 13)","Wires Fired per Layer(ME 13)","digi_stripswires_perlayer13.gif");
  Compare1DPlots2("Digis/hStripLayer+21","Digis/hWireLayer+21",f1,f2,"Strips Fired per Layer(ME 21)","Wires Fired per Layer(ME 21)","digi_stripswires_perlayer21.gif");
  Compare1DPlots2("Digis/hStripLayer+22","Digis/hWireLayer+22",f1,f2,"Strips Fired per Layer(ME 22)","Wires Fired per Layer(ME 22)","digi_stripswires_perlayer22.gif");
  Compare1DPlots2("Digis/hStripLayer+31","Digis/hWireLayer+31",f1,f2,"Strips Fired per Layer(ME 31)","Wires Fired per Layer(ME 31)","digi_stripswires_perlayer31.gif");
  Compare1DPlots2("Digis/hStripLayer+32","Digis/hWireLayer+32",f1,f2,"Strips Fired per Layer(ME 32)","Wires Fired per Layer(ME 32)","digi_stripswires_perlayer32.gif");
  Compare1DPlots2("Digis/hStripLayer+41","Digis/hWireLayer+41",f1,f2,"Strips Fired per Layer(ME 41)","Wires Fired per Layer(ME 41)","digi_stripswires_perlayer41.gif");
  Compare1DPlots2("Digis/hStripLayer+42","Digis/hWireLayer+42",f1,f2,"Strips Fired per Layer(ME 42)","Wires Fired per Layer(ME 42)","digi_stripswires_perlayer42.gif");
  Compare1DPlots2("Digis/hStripNFired","Digis/hWirenGroupsTotal",f1,f2,"Number of Fired Strips per Event","Number of Fired Wiregroups per Event","digi_stripswires_perevent.gif");
  Compare1DPlots2("Digis/hStripStrip+14","Digis/hWireWire+14",f1,f2,"Strip Numbers Fired(ME 11a)","Wiregroup Numbers Fired (ME 11a)","digi_stripswires_11a.gif");
  Compare1DPlots2("Digis/hStripStrip+11","Digis/hWireWire+11",f1,f2,"Strip Numbers Fired(ME 11b)","Wiregroup Numbers Fired (ME 11b)","digi_stripswires_11b.gif");
  Compare1DPlots2("Digis/hStripStrip+12","Digis/hWireWire+12",f1,f2,"Strip Numbers Fired(ME 12)","Wiregroup Numbers Fired (ME 12)","digi_stripswires_12.gif");
  Compare1DPlots2("Digis/hStripStrip+13","Digis/hWireWire+13",f1,f2,"Strip Numbers Fired(ME 13)","Wiregroup Numbers Fired (ME 13)","digi_stripswires_13.gif");
  Compare1DPlots2("Digis/hStripStrip+21","Digis/hWireWire+21",f1,f2,"Strip Numbers Fired(ME 21)","Wiregroup Numbers Fired (ME 21)","digi_stripswires_21.gif");
  Compare1DPlots2("Digis/hStripStrip+22","Digis/hWireWire+22",f1,f2,"Strip Numbers Fired(ME 22)","Wiregroup Numbers Fired (ME 22)","digi_stripswires_22.gif");
  Compare1DPlots2("Digis/hStripStrip+31","Digis/hWireWire+31",f1,f2,"Strip Numbers Fired(ME 31)","Wiregroup Numbers Fired (ME 31)","digi_stripswires_31.gif");
  Compare1DPlots2("Digis/hStripStrip+32","Digis/hWireWire+32",f1,f2,"Strip Numbers Fired(ME 32)","Wiregroup Numbers Fired (ME 32)","digi_stripswires_32.gif");
  Compare1DPlots2("Digis/hStripStrip+41","Digis/hWireWire+41",f1,f2,"Strip Numbers Fired(ME 41)","Wiregroup Numbers Fired (ME 41)","digi_stripswires_41.gif");
  Compare1DPlots2("Digis/hStripStrip+42","Digis/hWireWire+42",f1,f2,"Strip Numbers Fired(ME 42)","Wiregroup Numbers Fired (ME 42)","digi_stripswires_42.gif");
  Compare1DPlots2("Digis/hWireTBinAll+14","Digis/hWireTBinAll+11",f1,f2,"Wire Signal Time Bin (ME11a)","Wire Signal Time Bin (ME11b)","digi_wireTB_11.gif");
  Compare1DPlots2("Digis/hWireTBinAll+12","Digis/hWireTBinAll+13",f1,f2,"Wire Signal Time Bin (ME12)","Wire Signal Time Bin (ME13)","digi_wireTB_12_13.gif");
  Compare1DPlots2("Digis/hWireTBinAll+21","Digis/hWireTBinAll+22",f1,f2,"Wire Signal Time Bin (ME21)","Wire Signal Time Bin (ME22)","digi_wireTB_2.gif");
  Compare1DPlots2("Digis/hWireTBinAll+31","Digis/hWireTBinAll+32",f1,f2,"Wire Signal Time Bin (ME31)","Wire Signal Time Bin (ME32)","digi_wireTB_3.gif");
  Compare1DPlots2("Digis/hWireTBinAll+41","Digis/hWireTBinAll+42",f1,f2,"Wire Signal Time Bin (ME41)","Wire Signal Time Bin (ME42)","digi_wireTB_4.gif");

  //produce pedestal noise plots
  Compare1DPlots2("PedestalNoise/hStripPedME+11","PedestalNoise/hStripPedME+11",f1,f2,"Pedestal Noise Distribution Chamber ME11","Pedestal Noise Distribution Chamber ME11","noise_ME11.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME+12","PedestalNoise/hStripPedME+13",f1,f2,"Pedestal Noise Distribution Chamber ME12","Pedestal Noise Distribution Chamber ME13","noise_ME12_ME13.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME+21","PedestalNoise/hStripPedME+22",f1,f2,"Pedestal Noise Distribution Chamber ME21","Pedestal Noise Distribution Chamber ME22","noise_ME21_ME22.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME+31","PedestalNoise/hStripPedME+32",f1,f2,"Pedestal Noise Distribution Chamber ME31","Pedestal Noise Distribution Chamber ME32","noise_ME31_ME32.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME+41","PedestalNoise/hStripPedME+42",f1,f2,"Pedestal Noise Distribution Chamber ME41","Pedestal Noise Distribution Chamber ME42","noise_ME41_ME42.gif");


  //produce rechit comparison plots
  Compare1DPlots1("recHits/hRHCodeBroad",f1,f2,"hRHCodeBroad","rH_hRHCodeBroad.gif");
  Compare1DPlots2("recHits/hRHCodeNarrow1","recHits/hRHCodeNarrow2",f1,f2,"hRHCodeNarrow (Station 1)","hRHCodeNarrow (Station 2)","rH_hRHCodeNarrow_1a2.gif");
  Compare1DPlots2("recHits/hRHCodeNarrow3","recHits/hRHCodeNarrow4",f1,f2,"hRHCodeNarrow (Station 3)","hRHCodeNarrow (Station 4)","rH_hRHCodeNarrow_3a4.gif");
  Compare1DPlots2("recHits/hRHX+14","recHits/hRHX+11",f1,f2,"recHits, LocalX, ME 11a","recHits, LocalX, ME 11b","rH_local_X_ME11.gif");
  Compare1DPlots2("recHits/hRHY+14","recHits/hRHY+11",f1,f2,"recHits, LocalY, ME 11a","recHits, LocalY, ME 11b","rH_local_Y_ME11.gif");
  Compare1DPlots2("recHits/hRHX+12","recHits/hRHX+13",f1,f2,"recHits, LocalX, ME 12","recHits, LocalX, ME 13","rH_local_X_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHY+12","recHits/hRHY+13",f1,f2,"recHits, LocalY, ME 12","recHits, LocalY, ME 13","rH_local_Y_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHX+21","recHits/hRHX+22",f1,f2,"recHits, LocalX, ME 21","recHits, LocalX, ME 22","rH_local_X_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHY+21","recHits/hRHY+22",f1,f2,"recHits, LocalY, ME 21","recHits, LocalY, ME 22","rH_local_Y_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHX+31","recHits/hRHX+32",f1,f2,"recHits, LocalX, ME 31","recHits, LocalX, ME 32","rH_local_X_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHY+31","recHits/hRHY+32",f1,f2,"recHits, LocalY, ME 31","recHits, LocalY, ME 32","rH_local_Y_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHX+41","recHits/hRHX+42",f1,f2,"recHits, LocalX, ME 41","recHits, LocalX, ME 42","rH_local_X_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHY+41","recHits/hRHY+42",f1,f2,"recHits, LocalY, ME 41","recHits, LocalY, ME 42","rH_local_Y_ME41_ME42.gif");
  if (datatype == 2){
    Compare1DPlots2("recHits/hRHResid+14","recHits/hRHResid+11",f1,f2,"SimHit X - Reco X (ME11a)","SimHit X - Reco X (ME11b)","rH_resid_ME11.gif");
    Compare1DPlots2("recHits/hRHResid+12","recHits/hRHResid+13",f1,f2,"SimHit X - Reco X (ME12)","SimHit X - Reco X (ME13)","rH_sH_resid_ME12_ME13.gif");
    Compare1DPlots2("recHits/hRHResid+21","recHits/hRHResid+22",f1,f2,"SimHit X - Reco X (ME21)","SimHit X - Reco X (ME22)","rH_sH_resid_ME21_ME22.gif");
    Compare1DPlots2("recHits/hRHResid+31","recHits/hRHResid+32",f1,f2,"SimHit X - Reco X (ME31)","SimHit X - Reco X (ME32)","rH_sH_resid_ME31_ME32.gif");
    Compare1DPlots2("recHits/hRHResid+41","recHits/hRHResid+42",f1,f2,"SimHit X - Reco X (ME41)","SimHit X - Reco X (ME42)","rH_sH_resid_ME41_ME42.gif");
  }
  Compare1DPlots2("recHits/hRHLayer+14","recHits/hRHLayer+11",f1,f2,"recHits in a Layer, ME 11a","recHits in a Layer, ME 11b","rH_per_layer_ME11.gif");
  Compare1DPlots2("recHits/hRHLayer+12","recHits/hRHLayer+13",f1,f2,"recHits in a Layer, ME 12","recHits in a Layer, ME 13","rH_per_layer_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHLayer+21","recHits/hRHLayer+22",f1,f2,"recHits in a Layer, ME 21","recHits in a Layer, ME 22","rH_per_layer_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHLayer+31","recHits/hRHLayer+32",f1,f2,"recHits in a Layer, ME 31","recHits in a Layer, ME 32","rH_per_layer_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHLayer+41","recHits/hRHLayer+42",f1,f2,"recHits in a Layer, ME 41","recHits in a Layer, ME 42","rH_per_layer_ME41_ME42.gif");
  Compare1DPlots2("recHits/hSResid+14","recHits/hSResid+11",f1,f2,"Fitted Position on Strip - Reco Pos (ME11a)","Fitted Position on Strip - Reco Pos (ME11b)","rH_fit_resid_ME11.gif");
  Compare1DPlots2("recHits/hSResid+12","recHits/hSResid+13",f1,f2,"Fitted Position on Strip - Reco Pos (ME12)","Fitted Position on Strip - Reco Pos (ME13)","rH_fit_resid_ME12_ME13.gif");
  Compare1DPlots2("recHits/hSResid+21","recHits/hSResid+22",f1,f2,"Fitted Position on Strip - Reco Pos (ME21)","Fitted Position on Strip - Reco Pos (ME22)","rH_fit_resid_ME21_ME22.gif");
  Compare1DPlots2("recHits/hSResid+31","recHits/hSResid+32",f1,f2,"Fitted Position on Strip - Reco Pos (ME31)","Fitted Position on Strip - Reco Pos (ME32)","rH_fit_resid_ME31_ME32.gif");
  Compare1DPlots2("recHits/hSResid+41","recHits/hSResid+42",f1,f2,"Fitted Position on Strip - Reco Pos (ME41)","Fitted Position on Strip - Reco Pos (ME42)","rH_fit_resid_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHSumQ+14","recHits/hRHSumQ+11",f1,f2,"Sum 3 strip x 3 time bin charge (ME11a)","Sum 3 strip x 3 time bin charge (ME11b)","rH_sumQ_ME11.gif");
  Compare1DPlots2("recHits/hRHSumQ+12","recHits/hRHSumQ+13",f1,f2,"Sum 3 strip x 3 time bin charge (ME12)","Sum 3 strip x 3 time bin charge (ME13)","rH_sumQ_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHSumQ+21","recHits/hRHSumQ+22",f1,f2,"Sum 3 strip x 3 time bin charge (ME21)","Sum 3 strip x 3 time bin charge (ME22)","rH_sumQ_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHSumQ+31","recHits/hRHSumQ+32",f1,f2,"Sum 3 strip x 3 time bin charge (ME31)","Sum 3 strip x 3 time bin charge (ME32)","rH_sumQ_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHSumQ+41","recHits/hRHSumQ+42",f1,f2,"Sum 3 strip x 3 time bin charge (ME41)","Sum 3 strip x 3 time bin charge (ME42)","rH_sumQ_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHRatioQ+14","recHits/hRHRatioQ+11",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME11a)","Charge Ratio (Ql+Qr)/Qc (ME11b)","rH_ratioQ_ME11.gif");
  Compare1DPlots2("recHits/hRHRatioQ+12","recHits/hRHRatioQ+13",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME12)","Charge Ratio (Ql+Qr)/Qc (ME13)","rH_ratioQ_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHRatioQ+21","recHits/hRHRatioQ+22",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME21)","Charge Ratio (Ql+Qr)/Qc (ME22)","rH_ratioQ_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHRatioQ+31","recHits/hRHRatioQ+32",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME31)","Charge Ratio (Ql+Qr)/Qc (ME32)","rH_ratioQ_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHRatioQ+41","recHits/hRHRatioQ+42",f1,f2,"Charge Ratio (Ql+Qr)/Qc (ME41)","Charge Ratio (Ql+Qr)/Qc (ME42)","rH_ratioQ_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHTiming+14","recHits/hRHTiming+11",f1,f2,"recHit Timing from Strip (ME11a)","recHit Timing from Strip (ME11b)","rH_timing_ME11.gif");
  Compare1DPlots2("recHits/hRHTiming+12","recHits/hRHTiming+13",f1,f2,"recHit Timing from Strip (ME12)","recHit Timing from Strip (ME13)","rH_timing_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHTiming+21","recHits/hRHTiming+22",f1,f2,"recHit Timing from Strip (ME21)","recHit Timing from Strip (ME22)","rH_timing_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHTiming+31","recHits/hRHTiming+32",f1,f2,"recHit Timing from Strip (ME31)","recHit Timing from Strip (ME32)","rH_timing_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHTiming+41","recHits/hRHTiming+42",f1,f2,"recHit Timing from Strip (ME41)","recHit Timing from Strip (ME42)","rH_timing_ME41_ME42.gif");
  Compare1DPlots1("recHits/hRHnrechits",f1,f2,"Number of RecHits per Event","rH_nrH.gif");

  //produce segment comparison plots
  Compare1DPlots2("Segments/hSCodeNarrow1","Segments/hSCodeNarrow2",f1,f2,"hSCodeNarrow (Station 1)","hSCodeNarrow (Station 2)","seg_hSCodeNarrow_1a2.gif");
  Compare1DPlots2("Segments/hSCodeNarrow3","Segments/hSCodeNarrow4",f1,f2,"hSCodeNarrow (Station 3)","hSCodeNarrow (Station 4)","seg_hSCodeNarrow_3a4.gif");
  Compare1DPlots1("Segments/hSCodeBroad",f1,f2,"hSCodeBroad","seg_hSCodeBroad.gif");
  Compare1DPlots2("Segments/hSGlobalPhi","Segments/hSGlobalTheta",f1,f2,"Segment Global Phi (all stations)","Segment Global Theta (all stations)","seg_globthetaphi.gif");
  Compare1DPlots2("Segments/hSnHits+14","Segments/hSnHits+11",f1,f2,"recHits per Segment (ME 11a)","recHits per Segment (ME 11b)","seg_nhits_ME11.gif");
  Compare1DPlots2("Segments/hSnHits+12","Segments/hSnHits+13",f1,f2,"recHits per Segment (ME 12)","recHits per Segment (ME 13)","seg_nhits_ME12_ME13.gif");
  Compare1DPlots2("Segments/hSnHits+21","Segments/hSnHits+22",f1,f2,"recHits per Segment (ME 21)","recHits per Segment (ME 22)","seg_nhits_ME21_ME22.gif");
  Compare1DPlots2("Segments/hSnHits+31","Segments/hSnHits+32",f1,f2,"recHits per Segment (ME 31)","recHits per Segment (ME 32)","seg_nhits_ME31_ME32.gif");
  Compare1DPlots2("Segments/hSnHits+41","Segments/hSnHits+42",f1,f2,"recHits per Segment (ME 41)","recHits per Segment (ME 42)","seg_nhits_ME41_ME42.gif");
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

root -l -q -b ${MACRO}

rm makePlots.C

