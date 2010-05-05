#!/bin/sh

function run(){
    cmsRun DummyCondDBWriter_tmp_cfg.py
    [ $? -ne 0 ] && echo "PROBLEM" && cat DummyCondDBWriter_tmp_cfg.py && exit
}

rm dbfile.db
$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:dbfile.db a a

oldDest="sqlite_file:dbfile.db"
newDest="sqlite_file:dbfile.db"
#newDest="oracle://cms_orcoff_prep/CMS_COND_STRIP"

oldTag="31X"
newTag="31X"
#newTag="31X_v2"

cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripApvGain_Ideal@SiStripApvGain_IdealSim@" > DummyCondDBWriter_tmp_cfg.py
run
# Need to change also default to gaussian. default = value fixed to 1, gaussian = gaussian smearing with mean = MeanGain and sigma = SigmaGain
cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripApvGain_Ideal@SiStripApvGain_StartUp@" -e "s@SigmaGain=0.0@SigmaGain=0.10@" -e "s@default@gaussian@" > DummyCondDBWriter_tmp_cfg.py
run
# #
cat DummyCondDBWriter_SiStripBadChannel_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
cat DummyCondDBWriter_SiStripBadFiber_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
cat DummyCondDBWriter_SiStripBadModule_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
# #
# 
cat DummyCondDBWriter_SiStripThreshold_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
cat DummyCondDBWriter_SiStripClusterThreshold_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
# #
cat DummyCondDBWriter_SiStripFedCabling_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
# #
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
# 
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_IdealSim@" -e "s@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(0.)@" -e "s@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(0.)@"> DummyCondDBWriter_tmp_cfg.py
run
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_StartUp@" -e "s@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(20.)@" -e "s@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(20.)@"> DummyCondDBWriter_tmp_cfg.py
run
# #
cat DummyCondDBWriter_SiStripDetVOff_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
# #
cat DummyCondDBWriter_SiStripNoises_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
mv NoisesBuilder.log NoisesBuilder_DecMode.log
cat DummyCondDBWriter_SiStripNoises_PeakMode_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
mv NoisesBuilder.log NoisesBuilder_PeakMode.log
# #
cat DummyCondDBWriter_SiStripPedestals_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" > DummyCondDBWriter_tmp_cfg.py
run
