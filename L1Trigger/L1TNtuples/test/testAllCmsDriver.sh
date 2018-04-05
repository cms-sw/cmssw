
# setup
cp ~jbrooke/public/Summer15_25nsV6_DATA.db .

# RAW
echo 'RAW'
echo 'cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleRAW --conditions=auto:run2_data --no_output -n 1 --data'
cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleRAW --conditions=auto:run2_data --no_output -n 1 --data --filein=/store/data/Run2015D/DoubleEG/RAW/v1/000/260/627/00000/1A437A62-AD81-E511-AFED-02163E014414.root
mv L1Ntuple.root L1Ntuple.RAW.root

# RAW + EMU
echo 'RAW+EMU'
echo 'cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/Configuration/customise_Stage2Calo.Stage2CaloFromRaw --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleRAWEMU --conditions=auto:run2_data --no_output -n 1 --data'
cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/Configuration/customise_Stage2Calo.Stage2CaloFromRaw --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleRAWEMU --conditions=auto:run2_data --no_output -n 1 --data --filein=/store/data/Run2015D/DoubleEG/RAW/v1/000/260/627/00000/1A437A62-AD81-E511-AFED-02163E014414.root
mv L1Ntuple.root L1Ntuple.RAWEMU.root

# AOD
echo 'AOD'
echo 'cmsDriver.py l1Ntuple -s NONE --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAOD --conditions=auto:run2_data --no_output -n 1 --data'
cmsDriver.py l1Ntuple -s NONE --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAOD --conditions=auto:run2_data --no_output -n 1 --data --filein=/store/express/Run2015D/ExpressPhysics/FEVT/Express-v4/000/258/287/00000/0635BB66-876B-E511-8BB1-02163E013618.root
mv L1Ntuple.root L1Ntuple.AOD.root

# RAW + AOD
echo 'RAW+AOD'
echo 'cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAODRAW --conditions=auto:run2_data --no_output -n 1 --data'
cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAODRAW --conditions=auto:run2_data --no_output -n 1 --data --filein=/store/express/Run2015D/ExpressPhysics/FEVT/Express-v4/000/258/287/00000/0635BB66-876B-E511-8BB1-02163E013618.root
mv L1Ntuple.root L1Ntuple.AODRAW.root

# RAW + AOD + EMU
echo 'RAW+EMU+AOD'
echo 'cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/Configuration/customise_Stage2Calo.Stage2CaloFromRaw --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAODRAWEMU --conditions=auto:run2_data --no_output -n 1 --data'
cmsDriver.py l1Ntuple -s RAW2DIGI,L1Reco --customise=L1Trigger/Configuration/customise_Stage2Calo.Stage2CaloFromRaw --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAODRAWEMU --conditions=auto:run2_data --no_output -n 1 --data --filein=/store/express/Run2015D/ExpressPhysics/FEVT/Express-v4/000/258/287/00000/0635BB66-876B-E511-8BB1-02163E013618.root
mv L1Ntuple.root L1Ntuple.AODRAWEMU.root
