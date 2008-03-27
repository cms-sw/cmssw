#!/bin/bash

# this script will take output histos from CSCValidation and make 'nice' looking .gifs
#
# to run this script, do
# ./makePlots.sh <filepath> <data_type>
# where <filepath> is the paths to the output root files from CSCValiation
# data_type is an int (1 = data ; 2 = MC)

# example:  ./makeComparisonPlots.sh CMSSW_1_8_0_pre8/src/RecoLocalMuon/CSCValidation/test/ 2

ARG1=$1
ARG2=$2

MACRO=makePlots.C
cat > ${MACRO}<<EOF

{
  gROOT->Reset();
  gROOT->ProcessLine(".L myFunctions_nocompare.C");

  std::string Path = "${ARG1}";
  int datatype = ${ARG2};              // 1 = data, 2 = mc

  TFile *f1;
  TFile *f2;

  f1 = OpenFiles(Path,datatype);
  f2 = OpenFiles(Path,datatype);

  
  //procuce wire and strip digi comparison plots
  Compare1DPlots2("Digis/hStripAll","Digis/hWireAll",f1,f2,"Strip Numbers Fired (All Chambers)","Wire Groups Fired (All Chambers)","digi_stripswires_all.gif");
  Compare1DPlots2("Digis/hStripCodeBroad","Digis/hWireCodeBroad",f1,f2,"hStripCodeBroad","hWireCodeBroad","digi_stripswires_hCodeBroad.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow1","Digis/hWireCodeNarrow1",f1,f2,"hStripCodeNarrow (Station 1)","hWireCodeNarrow (Station 1)","digi_stripswires_hCodeNarrow1.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow2","Digis/hWireCodeNarrow2",f1,f2,"hStripCodeNarrow (Station 2)","hWireCodeNarrow (Station 2)","digi_stripswires_hCodeNarrow2.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow3","Digis/hWireCodeNarrow3",f1,f2,"hStripCodeNarrow (Station 3)","hWireCodeNarrow (Station 3)","digi_stripswires_hCodeNarrow3.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow4","Digis/hWireCodeNarrow4",f1,f2,"hStripCodeNarrow (Station 4)","hWireCodeNarrow (Station 4)","digi_stripswires_hCodeNarrow4.gif");
  Compare1DPlots2("Digis/hStripLayer11a","Digis/hWireLayer11a",f1,f2,"Strips Fired per Layer(ME 11a)","Wires Fired per Layer(ME 11a)","digi_stripswires_perlayer11a.gif");
  Compare1DPlots2("Digis/hStripLayer11b","Digis/hWireLayer11b",f1,f2,"Strips Fired per Layer(ME 11b)","Wires Fired per Layer(ME 11b)","digi_stripswires_perlayer11b.gif");
  Compare1DPlots2("Digis/hStripLayer12","Digis/hWireLayer12",f1,f2,"Strips Fired per Layer(ME 12)","Wires Fired per Layer(ME 12)","digi_stripswires_perlayer12.gif");
  Compare1DPlots2("Digis/hStripLayer13","Digis/hWireLayer13",f1,f2,"Strips Fired per Layer(ME 13)","Wires Fired per Layer(ME 13)","digi_stripswires_perlayer13.gif");
  Compare1DPlots2("Digis/hStripLayer21","Digis/hWireLayer21",f1,f2,"Strips Fired per Layer(ME 21)","Wires Fired per Layer(ME 21)","digi_stripswires_perlayer21.gif");
  Compare1DPlots2("Digis/hStripLayer22","Digis/hWireLayer22",f1,f2,"Strips Fired per Layer(ME 22)","Wires Fired per Layer(ME 22)","digi_stripswires_perlayer22.gif");
  Compare1DPlots2("Digis/hStripLayer31","Digis/hWireLayer31",f1,f2,"Strips Fired per Layer(ME 31)","Wires Fired per Layer(ME 31)","digi_stripswires_perlayer31.gif");
  Compare1DPlots2("Digis/hStripLayer32","Digis/hWireLayer32",f1,f2,"Strips Fired per Layer(ME 32)","Wires Fired per Layer(ME 32)","digi_stripswires_perlayer32.gif");
  Compare1DPlots2("Digis/hStripLayer41","Digis/hWireLayer41",f1,f2,"Strips Fired per Layer(ME 41)","Wires Fired per Layer(ME 41)","digi_stripswires_perlayer41.gif");
  Compare1DPlots2("Digis/hStripLayer42","Digis/hWireLayer42",f1,f2,"Strips Fired per Layer(ME 42)","Wires Fired per Layer(ME 42)","digi_stripswires_perlayer42.gif");
  Compare1DPlots2("Digis/hStripNFired","Digis/hWirenGroupsTotal",f1,f2,"Number of Fired Strips per Event","Number of Fired Wiregroups per Event","digi_stripswires_perevent.gif");
  Compare1DPlots2("Digis/hStripStrip11a","Digis/hWireWire11a",f1,f2,"Strip Numbers Fired(ME 11a)","Wiregroup Numbers Fired (ME 11a)","digi_stripswires_11a.gif");
  Compare1DPlots2("Digis/hStripStrip11b","Digis/hWireWire11b",f1,f2,"Strip Numbers Fired(ME 11b)","Wiregroup Numbers Fired (ME 11b)","digi_stripswires_11b.gif");
  Compare1DPlots2("Digis/hStripStrip12","Digis/hWireWire12",f1,f2,"Strip Numbers Fired(ME 12)","Wiregroup Numbers Fired (ME 12)","digi_stripswires_12.gif");
  Compare1DPlots2("Digis/hStripStrip13","Digis/hWireWire13",f1,f2,"Strip Numbers Fired(ME 13)","Wiregroup Numbers Fired (ME 13)","digi_stripswires_13.gif");
  Compare1DPlots2("Digis/hStripStrip21","Digis/hWireWire21",f1,f2,"Strip Numbers Fired(ME 21)","Wiregroup Numbers Fired (ME 21)","digi_stripswires_21.gif");
  Compare1DPlots2("Digis/hStripStrip22","Digis/hWireWire22",f1,f2,"Strip Numbers Fired(ME 22)","Wiregroup Numbers Fired (ME 22)","digi_stripswires_22.gif");
  Compare1DPlots2("Digis/hStripStrip31","Digis/hWireWire31",f1,f2,"Strip Numbers Fired(ME 31)","Wiregroup Numbers Fired (ME 31)","digi_stripswires_31.gif");
  Compare1DPlots2("Digis/hStripStrip32","Digis/hWireWire32",f1,f2,"Strip Numbers Fired(ME 32)","Wiregroup Numbers Fired (ME 32)","digi_stripswires_32.gif");
  Compare1DPlots2("Digis/hStripStrip41","Digis/hWireWire41",f1,f2,"Strip Numbers Fired(ME 41)","Wiregroup Numbers Fired (ME 41)","digi_stripswires_41.gif");
  Compare1DPlots2("Digis/hStripStrip42","Digis/hWireWire42",f1,f2,"Strip Numbers Fired(ME 42)","Wiregroup Numbers Fired (ME 42)","digi_stripswires_42.gif");
  Compare1DPlots1("Digis/hWireTBinAll",f1,f2,"Signal Time Bin for All Wires","digi_wireTB.gif");

  //produce pedestal noise plots
  Compare1DPlots2("PedestalNoise/hStripPedME11","PedestalNoise/hStripPedME11",f1,f2,"Pedestal Noise Distribution Chamber ME11","Pedestal Noise Distribution Chamber ME11","noise_ME11.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME12","PedestalNoise/hStripPedME13",f1,f2,"Pedestal Noise Distribution Chamber ME12","Pedestal Noise Distribution Chamber ME13","noise_ME12_ME13.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME21","PedestalNoise/hStripPedME22",f1,f2,"Pedestal Noise Distribution Chamber ME21","Pedestal Noise Distribution Chamber ME22","noise_ME21_ME22.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME31","PedestalNoise/hStripPedME32",f1,f2,"Pedestal Noise Distribution Chamber ME31","Pedestal Noise Distribution Chamber ME32","noise_ME31_ME32.gif");
  Compare1DPlots2("PedestalNoise/hStripPedME41","PedestalNoise/hStripPedME42",f1,f2,"Pedestal Noise Distribution Chamber ME41","Pedestal Noise Distribution Chamber ME42","noise_ME41_ME42.gif");


  //produce rechit comparison plots
  Compare1DPlots1("recHits/hRHCodeBroad",f1,f2,"hRHCodeBroad","rH_hRHCodeBroad.gif"); 
  Compare1DPlots2("recHits/hRHCodeNarrow1","recHits/hRHCodeNarrow2",f1,f2,"hRHCodeNarrow (Station 1)","hRHCodeNarrow (Station 2)","rH_hRHCodeNarrow_1a2.gif");
  Compare1DPlots2("recHits/hRHCodeNarrow3","recHits/hRHCodeNarrow4",f1,f2,"hRHCodeNarrow (Station 3)","hRHCodeNarrow (Station 4)","rH_hRHCodeNarrow_3a4.gif");
  Compare1DPlots2("recHits/hRHX11a","recHits/hRHX11b",f1,f2,"recHits, LocalX, ME 11a","recHits, LocalX, ME 11b","rH_local_X_ME11.gif");
  Compare1DPlots2("recHits/hRHY11a","recHits/hRHY11b",f1,f2,"recHits, LocalY, ME 11a","recHits, LocalY, ME 11b","rH_local_Y_ME11.gif");
  Compare1DPlots2("recHits/hRHX12","recHits/hRHX13",f1,f2,"recHits, LocalX, ME 12","recHits, LocalX, ME 13","rH_local_X_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHY12","recHits/hRHY13",f1,f2,"recHits, LocalY, ME 12","recHits, LocalY, ME 13","rH_local_Y_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHX21","recHits/hRHX22",f1,f2,"recHits, LocalX, ME 21","recHits, LocalX, ME 22","rH_local_X_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHY21","recHits/hRHY22",f1,f2,"recHits, LocalY, ME 21","recHits, LocalY, ME 22","rH_local_Y_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHX31","recHits/hRHX32",f1,f2,"recHits, LocalX, ME 31","recHits, LocalX, ME 32","rH_local_X_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHY31","recHits/hRHY32",f1,f2,"recHits, LocalY, ME 31","recHits, LocalY, ME 32","rH_local_Y_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHX41","recHits/hRHX42",f1,f2,"recHits, LocalX, ME 41","recHits, LocalX, ME 42","rH_local_X_ME41_ME42.gif");
  Compare1DPlots2("recHits/hRHY41","recHits/hRHY42",f1,f2,"recHits, LocalY, ME 41","recHits, LocalY, ME 42","rH_local_Y_ME41_ME42.gif");
  if (datatype == 2){
    Compare1DPlots2("recHits/hRHResid11a","recHits/hRHResid11b",f1,f2,"SimHit X - Reco X (ME11a)","SimHit X - Reco X (ME11b)","rH_resid_ME11.gif");
    Compare1DPlots2("recHits/hRHResid12","recHits/hRHResid13",f1,f2,"SimHit X - Reco X (ME12)","SimHit X - Reco X (ME13)","rH_sH_resid_ME12_ME13.gif"); 
    Compare1DPlots2("recHits/hRHResid21","recHits/hRHResid22",f1,f2,"SimHit X - Reco X (ME21)","SimHit X - Reco X (ME22)","rH_sH_resid_ME21_ME22.gif"); 
    Compare1DPlots2("recHits/hRHResid31","recHits/hRHResid32",f1,f2,"SimHit X - Reco X (ME31)","SimHit X - Reco X (ME32)","rH_sH_resid_ME31_ME32.gif"); 
    Compare1DPlots2("recHits/hRHResid41","recHits/hRHResid42",f1,f2,"SimHit X - Reco X (ME41)","SimHit X - Reco X (ME42)","rH_sH_resid_ME41_ME42.gif"); 
  }
  Compare1DPlots2("recHits/hRHLayer11a","recHits/hRHLayer11b",f1,f2,"recHits in a Layer, ME 11a","recHits in a Layer, ME 11b","rH_per_layer_ME11.gif");
  Compare1DPlots2("recHits/hRHLayer12","recHits/hRHLayer13",f1,f2,"recHits in a Layer, ME 12","recHits in a Layer, ME 13","rH_per_layer_ME12_ME13.gif");
  Compare1DPlots2("recHits/hRHLayer21","recHits/hRHLayer22",f1,f2,"recHits in a Layer, ME 21","recHits in a Layer, ME 22","rH_per_layer_ME21_ME22.gif");
  Compare1DPlots2("recHits/hRHLayer31","recHits/hRHLayer32",f1,f2,"recHits in a Layer, ME 31","recHits in a Layer, ME 32","rH_per_layer_ME31_ME32.gif");
  Compare1DPlots2("recHits/hRHLayer41","recHits/hRHLayer42",f1,f2,"recHits in a Layer, ME 41","recHits in a Layer, ME 42","rH_per_layer_ME41_ME42.gif");
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
  Compare1DPlots1("recHits/hRHnrechits",f1,f2,"Number of RecHits per Event","rH_nrH.gif");

  //produce segment comparison plots
  Compare1DPlots2("Segments/hSCodeNarrow1","Segments/hSCodeNarrow2",f1,f2,"hSCodeNarrow (Station 1)","hSCodeNarrow (Station 2)","seg_hSCodeNarrow_1a2.gif");
  Compare1DPlots2("Segments/hSCodeNarrow3","Segments/hSCodeNarrow4",f1,f2,"hSCodeNarrow (Station 3)","hSCodeNarrow (Station 4)","seg_hSCodeNarrow_3a4.gif");
  Compare1DPlots1("Segments/hSCodeBroad",f1,f2,"hSCodeBroad","seg_hSCodeBroad.gif");
  Compare1DPlots2("Segments/hSGlobalPhi","Segments/hSGlobalTheta",f1,f2,"Segment Global Phi (all stations)","Segment Global Theta (all stations)","seg_globthetaphi.gif");
  Compare1DPlots2("Segments/hSnHits11a","Segments/hSnHits11b",f1,f2,"recHits per Segment (ME 11a)","recHits per Segment (ME 11b)","seg_nhits_ME11.gif");
  Compare1DPlots2("Segments/hSnHits12","Segments/hSnHits13",f1,f2,"recHits per Segment (ME 12)","recHits per Segment (ME 13)","seg_nhits_ME12_ME13.gif");
  Compare1DPlots2("Segments/hSnHits21","Segments/hSnHits22",f1,f2,"recHits per Segment (ME 21)","recHits per Segment (ME 22)","seg_nhits_ME21_ME22.gif");
  Compare1DPlots2("Segments/hSnHits31","Segments/hSnHits32",f1,f2,"recHits per Segment (ME 31)","recHits per Segment (ME 32)","seg_nhits_ME31_ME32.gif");
  Compare1DPlots2("Segments/hSnHits41","Segments/hSnHits42",f1,f2,"recHits per Segment (ME 41)","recHits per Segment (ME 42)","seg_nhits_ME41_ME42.gif");
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



  Draw2DTempPlot("Digis/hOWires", f1, "wire_occupancy.gif");
  Draw2DTempPlot("Digis/hOStrips", f1, "strip_occupancy.gif");
  Draw2DTempPlot("recHits/hORecHits", f1, "rechit_occupancy.gif");
  Draw2DTempPlot("Segments/hOSegments", f1, "segment_occupancy.gif");


}

EOF

root -l -q -b ${MACRO}

rm makePlots.C

