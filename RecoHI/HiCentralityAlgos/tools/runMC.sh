#!/bin/sh
#Example tag : CentralityTable_HFhits40_Hydjet2760GeV_v1_mc -> CentralityTable_HFhits40_Hydjet2760GeV_v1_mc_MC_38Y_V12
conditions=MC_38Y_V12
version=1

for mc in Hydjet AMPT
  do
  for variable in HFtowers HFhits PixelHits Ntracks Npart
    do
    root -b -q "makeMCtableFromOpenHLT.C+(40,\"$variable\",\"CentralityTable_${variable}40_${mc}2760GeV_v${version}_mc_${conditions}\",\"$mc\")"
  done 
done


