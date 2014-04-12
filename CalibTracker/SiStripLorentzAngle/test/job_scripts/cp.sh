#!/bin/bash

WORK_DIR=$PWD

cd $WORK_DIR

export STAGE_SVCCLASS=cmscaf

for (( i=601; i<701; i++))
do

FILE1="/castor/cern.ch/user/s/sfrosali/test_new_binning/CSA08/MinBias/ALCARECO/1PB_V2_RECO_SiStripCalMinBias_v1/histos/LA_Histos_1PB_V2_RECO_SiStripCalMinBias_v1_"$i.root

echo $i;

rfcp $FILE1 . ;

done;

