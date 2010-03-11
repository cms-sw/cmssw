
datadir=$CMSSW_BASE/src/RecoHI/HiCentralityAlgos/data

root -b -q makeDataCentralityTable.C+'(40,"hf", "HFhits40_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(40,"ee", "EEbcs40_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","EEbcs40_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(20,"hf", "HFhits20_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","HFhits20_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(20,"ee", "EEbcs20_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","EEbcs20_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(10,"hf", "HFhits10_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","HFhits10_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(10,"ee", "EEbcs10_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","EEbcs10_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(5,"hf", "HFhits5_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","HFhits5_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'
root -b -q makeDataCentralityTable.C+'(5,"ee", "EEbcs5_DataHydjet2760GeV_Eff90p2_Hydjet4TeV_MC_3XY_V24_v0","EEbcs5_MC_Hydjet4TeV_MC_3XY_V24_v0",90.2)'

root -b -q makeDataCentralityTable.C+'(40,"hf", "HFhits40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","HFhits40_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(40,"ee", "EEbcs40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","EEbcs40_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(20,"hf", "HFhits40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","HFhits20_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(20,"ee", "EEbcs40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","EEbcs20_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(10,"hf", "HFhits40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","HFhits10_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(10,"ee", "EEbcs40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","EEbcs10_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(5,"hf", "HFhits40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","HFhits5_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'
root -b -q makeDataCentralityTable.C+'(5,"ee", "EEbcs40_DataHydjet4TeV_Eff89p5_Hydjet2760GeV_MC_3XY_V24_v0","EEbcs5_MC_Hydjet2760GeV_MC_3XY_V24_v0",89.5)'










