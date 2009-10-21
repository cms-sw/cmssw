#!/bin/sh

#MYPATH="GeneratorInterface/ThePEGInterface/test"
MYPATH="Configuration/GenProduction"
CONDITIONS="FrontierConditions_GlobalTag,IDEAL_31X::All"

CMSOPTS="-n 100 --eventcontent RAWSIM --conditions $CONDITIONS $CMSOPTS --no_exec --mc --customise=$MYPATH/custom"

cmsDriver.py "$MYPATH/testThePEGGeneratorFilter" -s GEN:ProductionFilterSequence --datatier GEN $CMSOPTS

cmsDriver.py "$MYPATH/testThePEGHadronisation"   -s GEN:ProductionFilterSequence --datatier GEN $CMSOPTS
cmsDriver.py "$MYPATH/testThePEGHadronisation"   -s GEN:ProductionFilterSequence,SIM,DIGI,L1,DIGI2RAW,HLT --datatier GEN $CMSOPTS
