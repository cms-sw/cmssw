
datadir=../data
rm $datadir/Cen*.*

cp ~/cvs/UserCode/CmsHi/JulyExercise/data/CentralityTables.root ./tables.root

root -b -q makeMCCentralityTable.C+'(40,"hf", "HFhits40_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(20,"hf", "HFhits20_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(10,"hf", "HFhits10_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(5,"hf", "HFhits5_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(1,"hf", "HFhits1_MC_AMPT2760GeV_MC_37Y_V5_v0")'

#root -b -q makeMCCentralityTable.C+'(40,"etmr", "ETmidRap40_MC_AMPT2760GeV_MC_37Y_V5_v0")'
#root -b -q makeMCCentralityTable.C+'(20,"etmr", "ETmidRap20_MC_AMPT2760GeV_MC_37Y_V5_v0")'
#root -b -q makeMCCentralityTable.C+'(10,"etmr", "ETmidRap10_MC_AMPT2760GeV_MC_37Y_V5_v0")'
#root -b -q makeMCCentralityTable.C+'(5,"etmr", "ETmidRap5_MC_AMPT2760GeV_MC_37Y_V5_v0")'
#root -b -q makeMCCentralityTable.C+'(1,"etmr", "ETmidRap1_MC_AMPT2760GeV_MC_37Y_V5_v0")'

root -b -q makeMCCentralityTable.C+'(40,"npix", "PixelNhits40_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(20,"npix", "PixelNhits20_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(10,"npix", "PixelNhits10_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(5,"npix", "PixelNhits5_MC_AMPT2760GeV_MC_37Y_V5_v0")'
root -b -q makeMCCentralityTable.C+'(1,"npix", "PixelNhits1_MC_AMPT2760GeV_MC_37Y_V5_v0")'

mv tables.root $datadir/CentralityTables.root

#root -b -q pushHistogramIntoFile.C'("HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0")'







