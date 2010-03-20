
datadir=$CMSSW_BASE/src/RecoHI/HiCentralityAlgos/data

rm tables.root
#rm $datadir/Cen*.*

root -b -q makeMCCentralityTable.C+'(40,"hf", "HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(40,"ee", "EEbcs40_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(20,"hf", "HFhits20_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(20,"ee", "EEbcs20_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(10,"hf", "HFhits10_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(10,"ee", "EEbcs10_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(5,"hf", "HFhits5_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeMCCentralityTable.C+'(5,"ee", "EEbcs5_MC_Hydjet4TeV_MC_3XY_V24_v0")'

#root -b -q makeMCCentralityTable.C+'(40,"hf", "HFhits40_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(40,"ee", "EEbcs40_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(20,"hf", "HFhits20_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(20,"ee", "EEbcs20_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(10,"hf", "HFhits10_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(10,"ee", "EEbcs10_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(5,"hf", "HFhits5_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q makeMCCentralityTable.C+'(5,"ee", "EEbcs5_MC_Hydjet2760GeV_MC_3XY_V24_v0")'

mv tables.root $datadir/CentralityTables.root

root -b -q pushHistogramIntoFile.C'("HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("EEbcs40_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("HFhits20_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("EEbcs20_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("HFhits10_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("EEbcs10_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("HFhits5_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q pushHistogramIntoFile.C'("EEbcs5_MC_Hydjet4TeV_MC_3XY_V24_v0")'

#root -b -q pushHistogramIntoFile.C'("HFhits40_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("EEbcs40_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("HFhits20_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("EEbcs20_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("HFhits10_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("EEbcs10_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("HFhits5_MC_Hydjet2760GeV_MC_3XY_V24_v0")'
#root -b -q pushHistogramIntoFile.C'("EEbcs5_MC_Hydjet2760GeV_MC_3XY_V24_v0")'







