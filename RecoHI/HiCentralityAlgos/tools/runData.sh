#!/bin/sh
#Example tag : CentralityTable_HFhits40_Hydjet2760GeV_v1_mc -> CentralityTable_HFhits40_Hydjet2760GeV_v1_mc_MC_38Y_V12
version=0

for run in r150305  r150308 r150431v2  r150436v2  r150442v2  r150471  r150476v2
  do
  for variable in HFtowers HFhits PixelHits Ntracks ETmidRapidity
    do
    root -b -q "makeTable.C+(40,\"$variable\",\"CentralityTable_${variable}40_AMPTOrgan_v${version}_run${run}_mc\",\"$run\")"
  done 
done


