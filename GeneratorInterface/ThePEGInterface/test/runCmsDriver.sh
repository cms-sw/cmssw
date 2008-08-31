#!/bin/sh

MYPATH="Configuration/GenProduction"
CONDITIONS="FrontierConditions_GlobalTag,IDEAL_V8::All"

CMSOPTS="-n 100 --eventcontent RAWSIM --conditions $CONDITIONS $CMSOPTS --no_exec"
CUSTOMS="--customise=$MYPATH/customSource.py"
CUSTOMP="--customise=$MYPATH/customProducer.py"

cmsDriver.py "$MYPATH/testThePEGSource.py"        -s GEN --datatier GEN $CMSOPTS $CUSTOMS
cmsDriver.py "$MYPATH/testThePEGProducer.py"      -s GEN:ProducerSourceSequence --datatier GEN $CMSOPTS $CUSTOMP
cmsDriver.py "$MYPATH/testThePEGHadronisation.py" -s GEN:ProducerSourceSequence --datatier GEN $CMSOPTS $CUSTOMP
cmsDriver.py "$MYPATH/testThePEGHadronisation.py" -s GEN:ProducerSourceSequence,SIM,DIGI,L1,DIGI2RAW,HLT --datatier GEN $CMSOPTS $CUSTOMP
