#!/bin/bash

# this script will take output histos from CSCValidation and make 'nice' looking .gifs
#
# to run this script, do
# ./makePlots.sh <newfilepath> <reffilepath>
# where <filepath> is the paths to the output root files from CSCValiation

# example:  ./makePlots.sh CMSSW_1_8_0_pre8/src/RecoLocalMuon/CSCValidation/test/validationHists.root CMSSW_2_1_0/src/RecoLocalMuon/CSCValidation/test/validationHists.root

ARG1=$1
ARG2=$2

MACRO=makePlots.C
cat > ${MACRO}<<EOF

{
  gROOT->Reset();
  gROOT->ProcessLine(".L myFunctions.C");

  std::string Path1 = "${ARG1}";
  std::string Path2 = "${ARG2}";

  TFile *f1;
  TFile *f2;
  f1 = OpenFiles(Path1);
  f2 = OpenFiles(Path2);

  //Make global position graphs from trees
  GlobalPosfromTreeCompare("Global recHit positions ME+1", f1, f2, 1, 1, "rechit", "rHglobal_station_+1.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME+2", f1, f2, 1, 2, "rechit", "rHglobal_station_+2.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME+3", f1, f2, 1, 3, "rechit", "rHglobal_station_+3.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME+4", f1, f2, 1, 4, "rechit", "rHglobal_station_+4.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME-1", f1, f2, 2, 1, "rechit", "rHglobal_station_-1.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME-2", f1, f2, 2, 2, "rechit", "rHglobal_station_-2.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME-3", f1, f2, 2, 3, "rechit", "rHglobal_station_-3.gif");
  GlobalPosfromTreeCompare("Global recHit positions ME-4", f1, f2, 2, 4, "rechit", "rHglobal_station_-4.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME+1", f1, f2, 1, 1, "segment", "Sglobal_station_+1.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME+2", f1, f2, 1, 2, "segment", "Sglobal_station_+2.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME+3", f1, f2, 1, 3, "segment", "Sglobal_station_+3.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME+4", f1, f2, 1, 4, "segment", "Sglobal_station_+4.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME-1", f1, f2, 2, 1, "segment", "Sglobal_station_-1.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME-2", f1, f2, 2, 2, "segment", "Sglobal_station_-2.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME-3", f1, f2, 2, 3, "segment", "Sglobal_station_-3.gif");
  GlobalPosfromTreeCompare("Global Segment positions ME-4", f1, f2, 2, 4, "segment", "Sglobal_station_-4.gif");


  //produce number of X per event plots
  compare1DPlot("Digis/hStripNFired",f1,f2,"Fired Strips per Event", 1110, "Digis_hStripNFired.gif");
  compare1DPlot("Digis/hWirenGroupsTotal",f1,f2,"Fired Wires per Event", 1110, "Digis_hWirenGroupsTotal.gif");
  compare1DPlot("recHits/hRHnrechits",f1,f2,"RecHits per Event", 1110, "recHits_hRHnrechits.gif");
  compare1DPlot("Segments/hSnSegments",f1,f2,"Segments per Event", 1110, "Segments_hSnSegments.gif");

  //efficiency plots
  compareEffGif("Efficiency/hRHEff", f1,f2, "RecHit Efficiecy", "Efficiency_hRHEff.gif");
  compareEffGif("Efficiency/hSEff", f1,f2, "Segment Efficiecy", "Efficiency_hSEff.gif");

  
  //produce wire timing plots
  compare1DPlot("Digis/hWireTBin+11",f1,f2,"Wire TimeBin Fired ME+1/1", 1110,"Digis_hWireTBin+11.gif");
  compare1DPlot("Digis/hWireTBin+12",f1,f2,"Wire TimeBin Fired ME+1/2", 1110,"Digis_hWireTBin+12.gif");
  compare1DPlot("Digis/hWireTBin+13",f1,f2,"Wire TimeBin Fired ME+1/3", 1110,"Digis_hWireTBin+13.gif");
  compare1DPlot("Digis/hWireTBin+21",f1,f2,"Wire TimeBin Fired ME+2/1", 1110,"Digis_hWireTBin+21.gif");
  compare1DPlot("Digis/hWireTBin+22",f1,f2,"Wire TimeBin Fired ME+2/2", 1110,"Digis_hWireTBin+22.gif");
  compare1DPlot("Digis/hWireTBin+31",f1,f2,"Wire TimeBin Fired ME+3/1", 1110,"Digis_hWireTBin+31.gif");
  compare1DPlot("Digis/hWireTBin+32",f1,f2,"Wire TimeBin Fired ME+3/2", 1110,"Digis_hWireTBin+32.gif");
  compare1DPlot("Digis/hWireTBin+41",f1,f2,"Wire TimeBin Fired ME+4/1", 1110,"Digis_hWireTBin+41.gif");
  compare1DPlot("Digis/hWireTBin-11",f1,f2,"Wire TimeBin Fired ME-1/1", 1110,"Digis_hWireTBin-11.gif");
  compare1DPlot("Digis/hWireTBin-12",f1,f2,"Wire TimeBin Fired ME-1/2", 1110,"Digis_hWireTBin-12.gif");
  compare1DPlot("Digis/hWireTBin-13",f1,f2,"Wire TimeBin Fired ME-1/3", 1110,"Digis_hWireTBin-13.gif");
  compare1DPlot("Digis/hWireTBin-21",f1,f2,"Wire TimeBin Fired ME-2/1", 1110,"Digis_hWireTBin-21.gif");
  compare1DPlot("Digis/hWireTBin-22",f1,f2,"Wire TimeBin Fired ME-2/2", 1110,"Digis_hWireTBin-22.gif");
  compare1DPlot("Digis/hWireTBin-31",f1,f2,"Wire TimeBin Fired ME-3/1", 1110,"Digis_hWireTBin-31.gif");
  compare1DPlot("Digis/hWireTBin-32",f1,f2,"Wire TimeBin Fired ME-3/2", 1110,"Digis_hWireTBin-32.gif");
  compare1DPlot("Digis/hWireTBin-41",f1,f2,"Wire TimeBin Fired ME-4/1", 1110,"Digis_hWireTBin-41.gif");


  //produce pedestal noise plots
  compare1DPlot("PedestalNoise/hStripPedME+11",f1,f2,"Pedestal Noise Distribution ME+1/1", 1110,"PedestalNoise_hStripPedME+11.gif");
  compare1DPlot("PedestalNoise/hStripPedME+12",f1,f2,"Pedestal Noise Distribution ME+1/2", 1110,"PedestalNoise_hStripPedME+12.gif");
  compare1DPlot("PedestalNoise/hStripPedME+13",f1,f2,"Pedestal Noise Distribution ME+1/3", 1110,"PedestalNoise_hStripPedME+13.gif");
  compare1DPlot("PedestalNoise/hStripPedME+21",f1,f2,"Pedestal Noise Distribution ME+2/1", 1110,"PedestalNoise_hStripPedME+21.gif");
  compare1DPlot("PedestalNoise/hStripPedME+22",f1,f2,"Pedestal Noise Distribution ME+2/2", 1110,"PedestalNoise_hStripPedME+22.gif");
  compare1DPlot("PedestalNoise/hStripPedME+31",f1,f2,"Pedestal Noise Distribution ME+3/1", 1110,"PedestalNoise_hStripPedME+31.gif");
  compare1DPlot("PedestalNoise/hStripPedME+32",f1,f2,"Pedestal Noise Distribution ME+3/2", 1110,"PedestalNoise_hStripPedME+32.gif");
  compare1DPlot("PedestalNoise/hStripPedME+41",f1,f2,"Pedestal Noise Distribution ME+4/1", 1110,"PedestalNoise_hStripPedME+41.gif");
  compare1DPlot("PedestalNoise/hStripPedME-11",f1,f2,"Pedestal Noise Distribution ME-1/1", 1110,"PedestalNoise_hStripPedME-11.gif");
  compare1DPlot("PedestalNoise/hStripPedME-12",f1,f2,"Pedestal Noise Distribution ME-1/2", 1110,"PedestalNoise_hStripPedME-12.gif");
  compare1DPlot("PedestalNoise/hStripPedME-13",f1,f2,"Pedestal Noise Distribution ME-1/3", 1110,"PedestalNoise_hStripPedME-13.gif");
  compare1DPlot("PedestalNoise/hStripPedME-21",f1,f2,"Pedestal Noise Distribution ME-2/1", 1110,"PedestalNoise_hStripPedME-21.gif");
  compare1DPlot("PedestalNoise/hStripPedME-22",f1,f2,"Pedestal Noise Distribution ME-2/2", 1110,"PedestalNoise_hStripPedME-22.gif");
  compare1DPlot("PedestalNoise/hStripPedME-31",f1,f2,"Pedestal Noise Distribution ME-3/1", 1110,"PedestalNoise_hStripPedME-31.gif");
  compare1DPlot("PedestalNoise/hStripPedME-32",f1,f2,"Pedestal Noise Distribution ME-3/2", 1110,"PedestalNoise_hStripPedME-32.gif");
  compare1DPlot("PedestalNoise/hStripPedME-41",f1,f2,"Pedestal Noise Distribution ME-4/1", 1110,"PedestalNoise_hStripPedME-41.gif");

  // resolution
  compare1DPlot("Resolution/hSResid+11",f1,f2,"Expected Position from Fit - Reconstructed, ME+1/1", 1110,"Resolution_hSResid+11.gif");
  compare1DPlot("Resolution/hSResid+12",f1,f2,"Expected Position from Fit - Reconstructed, ME+1/2", 1110,"Resolution_hSResid+12.gif");
  compare1DPlot("Resolution/hSResid+13",f1,f2,"Expected Position from Fit - Reconstructed, ME+1/3", 1110,"Resolution_hSResid+13.gif");
  compare1DPlot("Resolution/hSResid+21",f1,f2,"Expected Position from Fit - Reconstructed, ME+2/1", 1110,"Resolution_hSResid+21.gif");
  compare1DPlot("Resolution/hSResid+22",f1,f2,"Expected Position from Fit - Reconstructed, ME+2/2", 1110,"Resolution_hSResid+22.gif");
  compare1DPlot("Resolution/hSResid+31",f1,f2,"Expected Position from Fit - Reconstructed, ME+3/1", 1110,"Resolution_hSResid+31.gif");
  compare1DPlot("Resolution/hSResid+32",f1,f2,"Expected Position from Fit - Reconstructed, ME+3/2", 1110,"Resolution_hSResid+32.gif");
  compare1DPlot("Resolution/hSResid+41",f1,f2,"Expected Position from Fit - Reconstructed, ME+4/1", 1110,"Resolution_hSResid+41.gif");
  compare1DPlot("Resolution/hSResid-11",f1,f2,"Expected Position from Fit - Reconstructed, ME-1/1", 1110,"Resolution_hSResid-11.gif");
  compare1DPlot("Resolution/hSResid-12",f1,f2,"Expected Position from Fit - Reconstructed, ME-1/2", 1110,"Resolution_hSResid-12.gif");
  compare1DPlot("Resolution/hSResid-13",f1,f2,"Expected Position from Fit - Reconstructed, ME-1/3", 1110,"Resolution_hSResid-13.gif");
  compare1DPlot("Resolution/hSResid-21",f1,f2,"Expected Position from Fit - Reconstructed, ME-2/1", 1110,"Resolution_hSResid-21.gif");
  compare1DPlot("Resolution/hSResid-22",f1,f2,"Expected Position from Fit - Reconstructed, ME-2/2", 1110,"Resolution_hSResid-22.gif");
  compare1DPlot("Resolution/hSResid-31",f1,f2,"Expected Position from Fit - Reconstructed, ME-3/1", 1110,"Resolution_hSResid-31.gif");
  compare1DPlot("Resolution/hSResid-32",f1,f2,"Expected Position from Fit - Reconstructed, ME-3/2", 1110,"Resolution_hSResid-32.gif");
  compare1DPlot("Resolution/hSResid-41",f1,f2,"Expected Position from Fit - Reconstructed, ME-4/1", 1110,"Resolution_hSResid-41.gif");


  // rechit timing
  compare1DPlot("recHits/hRHTiming+11",f1,f2,"RecHit Timing ME+1/1", 1110,"recHits_hRHTiming+11.gif");
  compare1DPlot("recHits/hRHTiming+12",f1,f2,"RecHit Timing ME+1/2", 1110,"recHits_hRHTiming+12.gif");
  compare1DPlot("recHits/hRHTiming+13",f1,f2,"RecHit Timing ME+1/3", 1110,"recHits_hRHTiming+13.gif");
  compare1DPlot("recHits/hRHTiming+21",f1,f2,"RecHit Timing ME+2/1", 1110,"recHits_hRHTiming+21.gif");
  compare1DPlot("recHits/hRHTiming+22",f1,f2,"RecHit Timing ME+2/2", 1110,"recHits_hRHTiming+22.gif");
  compare1DPlot("recHits/hRHTiming+31",f1,f2,"RecHit Timing ME+3/1", 1110,"recHits_hRHTiming+31.gif");
  compare1DPlot("recHits/hRHTiming+32",f1,f2,"RecHit Timing ME+3/2", 1110,"recHits_hRHTiming+32.gif");
  compare1DPlot("recHits/hRHTiming+41",f1,f2,"RecHit Timing ME+4/1", 1110,"recHits_hRHTiming+41.gif");
  compare1DPlot("recHits/hRHTiming-11",f1,f2,"RecHit Timing ME-1/1", 1110,"recHits_hRHTiming-11.gif");
  compare1DPlot("recHits/hRHTiming-12",f1,f2,"RecHit Timing ME-1/2", 1110,"recHits_hRHTiming-12.gif");
  compare1DPlot("recHits/hRHTiming-13",f1,f2,"RecHit Timing ME-1/3", 1110,"recHits_hRHTiming-13.gif");
  compare1DPlot("recHits/hRHTiming-21",f1,f2,"RecHit Timing ME-2/1", 1110,"recHits_hRHTiming-21.gif");
  compare1DPlot("recHits/hRHTiming-22",f1,f2,"RecHit Timing ME-2/2", 1110,"recHits_hRHTiming-22.gif");
  compare1DPlot("recHits/hRHTiming-31",f1,f2,"RecHit Timing ME-3/1", 1110,"recHits_hRHTiming-31.gif");
  compare1DPlot("recHits/hRHTiming-32",f1,f2,"RecHit Timing ME-3/2", 1110,"recHits_hRHTiming-32.gif");
  compare1DPlot("recHits/hRHTiming-41",f1,f2,"RecHit Timing ME-4/1", 1110,"recHits_hRHTiming-41.gif");

  // rechit charge
  compare1DPlot("recHits/hRHSumQ+11",f1,f2,"Sum 3x3 RecHit Charge ME+1/1", 1110,"recHits_hRHSumQ+11.gif");
  compare1DPlot("recHits/hRHSumQ+12",f1,f2,"Sum 3x3 RecHit Charge ME+1/2", 1110,"recHits_hRHSumQ+12.gif");
  compare1DPlot("recHits/hRHSumQ+13",f1,f2,"Sum 3x3 RecHit Charge ME+1/3", 1110,"recHits_hRHSumQ+13.gif");
  compare1DPlot("recHits/hRHSumQ+21",f1,f2,"Sum 3x3 RecHit Charge ME+2/1", 1110,"recHits_hRHSumQ+21.gif");
  compare1DPlot("recHits/hRHSumQ+22",f1,f2,"Sum 3x3 RecHit Charge ME+2/2", 1110,"recHits_hRHSumQ+22.gif");
  compare1DPlot("recHits/hRHSumQ+31",f1,f2,"Sum 3x3 RecHit Charge ME+3/1", 1110,"recHits_hRHSumQ+31.gif");
  compare1DPlot("recHits/hRHSumQ+32",f1,f2,"Sum 3x3 RecHit Charge ME+3/2", 1110,"recHits_hRHSumQ+32.gif");
  compare1DPlot("recHits/hRHSumQ+41",f1,f2,"Sum 3x3 RecHit Charge ME+4/1", 1110,"recHits_hRHSumQ+41.gif");
  compare1DPlot("recHits/hRHSumQ-11",f1,f2,"Sum 3x3 RecHit Charge ME-1/1", 1110,"recHits_hRHSumQ-11.gif");
  compare1DPlot("recHits/hRHSumQ-12",f1,f2,"Sum 3x3 RecHit Charge ME-1/2", 1110,"recHits_hRHSumQ-12.gif");
  compare1DPlot("recHits/hRHSumQ-13",f1,f2,"Sum 3x3 RecHit Charge ME-1/3", 1110,"recHits_hRHSumQ-13.gif");
  compare1DPlot("recHits/hRHSumQ-21",f1,f2,"Sum 3x3 RecHit Charge ME-2/1", 1110,"recHits_hRHSumQ-21.gif");
  compare1DPlot("recHits/hRHSumQ-22",f1,f2,"Sum 3x3 RecHit Charge ME-2/2", 1110,"recHits_hRHSumQ-22.gif");
  compare1DPlot("recHits/hRHSumQ-31",f1,f2,"Sum 3x3 RecHit Charge ME-3/1", 1110,"recHits_hRHSumQ-31.gif");
  compare1DPlot("recHits/hRHSumQ-32",f1,f2,"Sum 3x3 RecHit Charge ME-3/2", 1110,"recHits_hRHSumQ-32.gif");
  compare1DPlot("recHits/hRHSumQ-41",f1,f2,"Sum 3x3 RecHit Charge ME-4/1", 1110,"recHits_hRHSumQ-41.gif");

  compare1DPlot("recHits/hRHRatioQ+11",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+1/1", 1110,"recHits_hRHRatioQ+11.gif");
  compare1DPlot("recHits/hRHRatioQ+12",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+1/2", 1110,"recHits_hRHRatioQ+12.gif");
  compare1DPlot("recHits/hRHRatioQ+13",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+1/3", 1110,"recHits_hRHRatioQ+13.gif");
  compare1DPlot("recHits/hRHRatioQ+21",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+2/1", 1110,"recHits_hRHRatioQ+21.gif");
  compare1DPlot("recHits/hRHRatioQ+22",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+2/2", 1110,"recHits_hRHRatioQ+22.gif");
  compare1DPlot("recHits/hRHRatioQ+31",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+3/1", 1110,"recHits_hRHRatioQ+31.gif");
  compare1DPlot("recHits/hRHRatioQ+32",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+3/2", 1110,"recHits_hRHRatioQ+32.gif");
  compare1DPlot("recHits/hRHRatioQ+41",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME+4/1", 1110,"recHits_hRHRatioQ+41.gif");
  compare1DPlot("recHits/hRHRatioQ-11",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-1/1", 1110,"recHits_hRHRatioQ-11.gif");
  compare1DPlot("recHits/hRHRatioQ-12",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-1/2", 1110,"recHits_hRHRatioQ-12.gif");
  compare1DPlot("recHits/hRHRatioQ-13",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-1/3", 1110,"recHits_hRHRatioQ-13.gif");
  compare1DPlot("recHits/hRHRatioQ-21",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-2/1", 1110,"recHits_hRHRatioQ-21.gif");
  compare1DPlot("recHits/hRHRatioQ-22",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-2/2", 1110,"recHits_hRHRatioQ-22.gif");
  compare1DPlot("recHits/hRHRatioQ-31",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-3/1", 1110,"recHits_hRHRatioQ-31.gif");
  compare1DPlot("recHits/hRHRatioQ-32",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-3/2", 1110,"recHits_hRHRatioQ-32.gif");
  compare1DPlot("recHits/hRHRatioQ-41",f1,f2,"Charge Ratio (Ql_Qr)/Qt ME-4/1", 1110,"recHits_hRHRatioQ-41.gif");

  //hits on a segment
  compare1DPlot("Segments/hSnHits+11",f1,f2,"N Hits on Segments ME+1/1", 1110,"Segments_hSnHits+11.gif");
  compare1DPlot("Segments/hSnHits+12",f1,f2,"N Hits on Segments ME+1/2", 1110,"Segments_hSnHits+12.gif");
  compare1DPlot("Segments/hSnHits+13",f1,f2,"N Hits on Segments ME+1/3", 1110,"Segments_hSnHits+13.gif");
  compare1DPlot("Segments/hSnHits+21",f1,f2,"N Hits on Segments ME+2/1", 1110,"Segments_hSnHits+21.gif");
  compare1DPlot("Segments/hSnHits+22",f1,f2,"N Hits on Segments ME+2/2", 1110,"Segments_hSnHits+22.gif");
  compare1DPlot("Segments/hSnHits+31",f1,f2,"N Hits on Segments ME+3/1", 1110,"Segments_hSnHits+31.gif");
  compare1DPlot("Segments/hSnHits+32",f1,f2,"N Hits on Segments ME+3/2", 1110,"Segments_hSnHits+32.gif");
  compare1DPlot("Segments/hSnHits+41",f1,f2,"N Hits on Segments ME+4/1", 1110,"Segments_hSnHits+41.gif");
  compare1DPlot("Segments/hSnHits-11",f1,f2,"N Hits on Segments ME-1/1", 1110,"Segments_hSnHits-11.gif");
  compare1DPlot("Segments/hSnHits-12",f1,f2,"N Hits on Segments ME-1/2", 1110,"Segments_hSnHits-12.gif");
  compare1DPlot("Segments/hSnHits-13",f1,f2,"N Hits on Segments ME-1/3", 1110,"Segments_hSnHits-13.gif");
  compare1DPlot("Segments/hSnHits-21",f1,f2,"N Hits on Segments ME-2/1", 1110,"Segments_hSnHits-21.gif");
  compare1DPlot("Segments/hSnHits-22",f1,f2,"N Hits on Segments ME-2/2", 1110,"Segments_hSnHits-22.gif");
  compare1DPlot("Segments/hSnHits-31",f1,f2,"N Hits on Segments ME-3/1", 1110,"Segments_hSnHits-31.gif");
  compare1DPlot("Segments/hSnHits-32",f1,f2,"N Hits on Segments ME-3/2", 1110,"Segments_hSnHits-32.gif");
  compare1DPlot("Segments/hSnHits-41",f1,f2,"N Hits on Segments ME-4/1", 1110,"Segments_hSnHits-41.gif");

  //miscellaneous
  compare1DPlot("Segments/hSGlobalPhi",f1,f2,"Segment Global Phi", 1110,"Segments_hSGlobalPhi.gif");
  compare1DPlot("Segments/hSGlobalTheta",f1,f2,"Segment Global Theta", 1110,"Segments_hSGlobalTheta.gif");
  


}

EOF

root -l -q -b ${MACRO}

rm makePlots.C

