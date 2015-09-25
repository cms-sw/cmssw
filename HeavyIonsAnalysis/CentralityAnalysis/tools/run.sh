
datadir=$CMSSW_BASE/src/RecoHI/HiCentralityAlgos/data

rm tables.root
rm $datadir/Cen*.*

root -b -q makeCentralityTable.C+'(40,"hf", "HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(40,"ee", "EEbcs40_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(20,"hf", "HFhits20_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(20,"ee", "EEbcs20_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(10,"hf", "HFhits10_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(10,"ee", "EEbcs10_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(5,"hf", "HFhits5_MC_Hydjet4TeV_MC_3XY_V24_v0")'
root -b -q makeCentralityTable.C+'(5,"ee", "EEbcs5_MC_Hydjet4TeV_MC_3XY_V24_v0")'

mv tables.root $datadir/CentralityTables.root

cmsRun makeDBFromTFile.py outputTag="HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs40_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits20_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs20_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits10_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs10_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits5_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs5_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db






