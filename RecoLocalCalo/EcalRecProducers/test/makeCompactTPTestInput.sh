#!/bin/sh

# for normal testing
globalTag='auto:mc'

nevents=10

function die(){ echo "$*"; }

[ -f ZEE_7TeV_FEVTDEBUGHLT.root ] || cmsDriver.py ZEE_7TeV.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco -n $nevents --geometry DB --conditions $globalTag --relval 9000,100 --datatier GEN-SIM-DIGI-RAW-HLTDEBUG --eventcontent FEVTDEBUGHLT --fileout=ZEE_7TeV_FEVTDEBUGHLT.root || die "Failed to run simulation!"
 
cmsDriver.py step2 -s RAW2DIGI,L1Reco,RECO -n $nevents --datatier GEN-SIM-RECO --eventcontent RECOSIM --geometry DB --conditions=$globalTag --filein=file:ZEE_7TeV_FEVTDEBUGHLT.root --fileout=ZEE_7TeV_RECOSIM.root --no_exec || die "Failed to produce step2 configuration file"

cat >> step2_RAW2DIGI_L1Reco_RECO.py <<EOF
 
process.RECOSIMEventContent.outputCommands.extend(cms.untracked.vstring('keep EcalTriggerPrimitiveDigisSorted_simEcalTriggerPrimitiveDigis_*_*'))
process.RECOSIMEventContent.outputCommands.extend(cms.untracked.vstring('keep EcalTriggerPrimitiveDigisSorted_ecalDigis_*_*'))
EOF

cmsRun step2_RAW2DIGI_L1Reco_RECO.py

ln -s ZEE_7TeV_RECOSIM.root in.root
