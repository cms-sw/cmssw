#!/bin/sh
#Example tag : CentralityTable_HFplus100_PAHijing_v1_mc -> CentralityTable_HFplus100_PAHijing_v1_mc_MC_53Y_V0
version=0
macro=makeTable2
for variable in HFtowers HFtowersPlus HFtowersMinus HFtowersPlusTrunc HFtowersMinusTrunc ZDC ZDCplus ZDCminus PixelHits PixelTracks Tracks
  do
  root -b -q -l rootlogon.C "$macro.C++(100,\"$variable\",\"CentralityTable_${variable}100_v${version}_offline\",false,1)"
  cat out/output.txt >> out/Data_nocorr_v${version}.txt
done
