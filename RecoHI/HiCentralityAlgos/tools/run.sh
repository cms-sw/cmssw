
datadir=$CMSSW_BASE/src/RecoHI/HiCentralityAlgos/data

rm tables.root

root -b -q makeCentralityTable.C+'(40,"hf", "HFhits40_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(40,"ee", "EEbcs40_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(20,"hf", "HFhits20_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(20,"ee", "EEbcs20_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(10,"hf", "HFhits10_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(10,"ee", "EEbcs10_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(5,"hf", "HFhits5_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'(5,"ee", "EEbcs5_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'

mv tables.root $datadir/CentralityTables.root

cmsRun makeDBFromTFile.py outputTag="HFhits40_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs40_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits20_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs20_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits10_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs10_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits5_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs5_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db






