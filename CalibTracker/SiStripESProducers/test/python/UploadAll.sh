#!/bin/sh

rm dbfile.db
$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:dbfile.db a a

oldDest="sqlite_file:dbfile.db"
newDest="sqlite_file:dbfile.db"
#newDest="oracle://cms_orcoff_prep/CMS_COND_STRIP"

oldTag="31X"
newTag="31X"
#newTag="31X_v2"

cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripApvGain_Ideal@SiStripApvGain_IdealSim@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripApvGain_Ideal@SiStripApvGain_StartUp@" -e "s@MeanSigma=0.0@MeanSigma=0.10@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# #
cat DummyCondDBWriter_SiStripBadChannel_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripBadFiber_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripBadModule_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# #
# 
cat DummyCondDBWriter_SiStripThreshold_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripClusterThreshold_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# #
cat DummyCondDBWriter_SiStripFedCabling_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# #
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# 
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_IdealSim@" -e "s@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(0.)@" -e "s@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(0.)@"> DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_StartUp@" -e "s@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(20.)@" -e "s@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(20.)@"> DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# #
cat DummyCondDBWriter_SiStripDetVOff_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripModuleHV_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
cat DummyCondDBWriter_SiStripModuleLV_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
# #
cat DummyCondDBWriter_SiStripNoises_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
mv NoisesBuilder.log NoisesBuilder_DecMode.log
cat DummyCondDBWriter_SiStripNoises_PeakMode_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
mv NoisesBuilder.log NoisesBuilder_PeakMode.log
# #
cat DummyCondDBWriter_SiStripPedestals_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py
