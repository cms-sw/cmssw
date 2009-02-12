#!/bin/sh

#MYPATH="GeneratorInterface/ThePEGInterface"
MYPATH="Configuration/GenProduction"
CONDITIONS="FrontierConditions_GlobalTag,IDEAL_30X::All"

CMSOPTS="-n 100 --eventcontent RAWSIM --conditions $CONDITIONS $CMSOPTS --no_exec"
CUSTOMS="--customise=$MYPATH/customSource"
CUSTOMP="--customise=$MYPATH/customProducer"

cmsDriver.py "$MYPATH/testThePEGGeneratorFilter" -s GEN:ProducerSourceSequence --datatier GEN $CMSOPTS $CUSTOMP

cmsDriver.py "$MYPATH/testThePEGSource"          -s GEN --datatier GEN $CMSOPTS $CUSTOMS
cmsDriver.py "$MYPATH/testThePEGProducer"        -s GEN:ProducerSourceSequence --datatier GEN $CMSOPTS $CUSTOMP
cmsDriver.py "$MYPATH/testThePEGHadronisation"   -s GEN:ProducerSourceSequence --datatier GEN $CMSOPTS $CUSTOMP
cmsDriver.py "$MYPATH/testThePEGHadronisation"   -s GEN:ProducerSourceSequence,SIM,DIGI,L1,DIGI2RAW,HLT --datatier GEN $CMSOPTS $CUSTOMP
