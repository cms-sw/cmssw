
datadir=$CMSSW_BASE/src/RecoHI/HiCentralityAlgos/data
#rm $datadir/Cen*.db

cmsRun makeDBFromTFile.py outputTag="HFhits40_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs40_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits20_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs20_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits10_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs10_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="HFhits5_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
cmsRun makeDBFromTFile.py outputTag="EEbcs5_MC_Hydjet4TeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db

#cmsRun makeDBFromTFile.py outputTag="HFhits40_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="EEbcs40_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="HFhits20_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="EEbcs20_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="HFhits10_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="EEbcs10_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="HFhits5_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db
#cmsRun makeDBFromTFile.py outputTag="EEbcs5_MC_Hydjet2760GeV_MC_3XY_V24_v0" inputFile=$datadir/CentralityTables.root outputFile=$datadir/CentralityTables.db






