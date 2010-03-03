
datadir=$CMSSW_BASE/src/RecoHI/HiCentralityAlgos/data

rm tables.root

root -b -q makeCentralityTable.C+'("hf", "HFhits40_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
#root -b -q makeCentralityTable.C+'("eb", "EBbcs40_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'
root -b -q makeCentralityTable.C+'("ee", "EEbcs40_MXS0_Hydjet4TeV_MC_3XY_V21_v0", 0.)'

mv tables.root $datadir/CentralityTables.root

cmsRun makeDBFromTFile.py outputTag="HFhits40_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs40_MXS0_Hydjet4TeV_MC_3XY_V21_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db






