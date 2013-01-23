#!/bin/sh
#Example tag : CentralityTable_HFplus100_PAHijing_v1_mc -> CentralityTable_HFplus100_PAHijing_v1_mc_MC_53Y_V0
version=4
macro=makeTable2
generator=Hijing
#for variable in HFtowers HFtowersPlus HFtowersMinus HFtowersPlusTrunc HFtowersMinusTrunc PixelHits PixelTracks Tracks b Npart Ncoll Nhard
for variable in HFtowersPlusTrunc Tracks
  do
  root -b -q -l rootlogon.C "$macro.C++(100,\"$variable\",\"CentralityTable_${variable}_${generator}_v${version}_mc\",true,1)"
  cat out/output.txt >> out/${generator}_v${version}.txt
done
