#!/bin/sh

cmsRun EcalAlCaRecoProducers/python/alcaSkimming.py \
type="ALCARECO" \
skim="ZSkim" \
doTree=1 \
doTreeOnly=0 \
doHighEta=1 \
isCrab=0 \
jsonFile="/afs/cern.ch/user/h/heli/work/private/calib/2014Jan27/ecalelf5/CMSSW_5_3_14_patch2/src/Calibration/Cert_190456-208686_8TeV_22Jan2013ReReco_Collisions12_JSON.txt" \
files="root://eoscms//eos/cms/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10C8A92C-8867-E211-B5C7-003048678B0E.root"


#options.files="root://eoscms.cern.ch//eos/cms/store/group/alca_ecalcalib/ecalelf/alcareco/8TeV/DoubleElectron-ZSkim-RUN2012A-22Jan-v1/190645-193621/alcareco_1_1_G0r.root"  
