#!/bin/sh

rm dbfile.db
$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:dbfile.db a a

cmsRun DummyCondDBWriter_SiStripApvGain_cfg.py 
cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@SiStripApvGain_Ideal@SiStripApvGain_IdealSim@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@SiStripApvGain_Ideal@SiStripApvGain_StartUp@" -e "s@MeanSigma=0.0@MeanSigma=0.10@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py

cmsRun DummyCondDBWriter_SiStripBadChannel_cfg.py
cmsRun DummyCondDBWriter_SiStripBadFiber_cfg.py
cmsRun DummyCondDBWriter_SiStripBadModule_cfg.py

cmsRun DummyCondDBWriter_SiStripThreshold_cfg.py
cmsRun DummyCondDBWriter_SiStripClusterThreshold_cfg.py

cmsRun DummyCondDBWriter_SiStripFedCabling_cfg.py

cmsRun DummyCondDBWriter_SiStripLorentzAngle_cfg.py
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_IdealSim@" -e s"@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(0.)@" -e s"@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(0.)@"> DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_StartUp@" -e s"@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(20.)@" -e s"@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(20.)@"> DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py

cmsRun DummyCondDBWriter_SiStripDetVOff_cfg.py
cmsRun DummyCondDBWriter_SiStripModuleHV_cfg.py
cmsRun DummyCondDBWriter_SiStripModuleLV_cfg.py

cmsRun DummyCondDBWriter_SiStripNoises_cfg.py
cmsRun DummyCondDBWriter_SiStripNoises_PeakMode_cfg.py

cmsRun DummyCondDBWriter_SiStripPedestals_cfg.py

