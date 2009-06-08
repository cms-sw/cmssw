#!/bin/bash

cd WORK_DIR;

eval `scramv1 runtime -sh`

python PY;
cmsRun PY;

export STAGE_SVCCLASS=cmscaf
export STAGER_TRACE=3

rfcp ./PY MY_CASTOR_DIR/DIR_PY;
rm ./PY;

rfcp ./LA_debug_NEWAL_NUMBER.log MY_CASTOR_DIR/DIR_DEBUG;
rm ./LA_debug_NEWAL_NUMBER.log;

cd MY_TMP/;

rfcp ./LA_Histos_NEWAL_NUMBER.root MY_CASTOR_DIR/DIR_HISTOS;
rm ./LA_Histos_NEWAL_NUMBER.root;

rfcp ./LA_Histos_Harv_NEWAL_NUMBER.root MY_CASTOR_DIR/DIR_HISTOHARV;
rm ./LA_Histos_Harv_NEWAL_NUMBER.root;

rfcp ./LA_Trees_NEWAL_NUMBER.root MY_CASTOR_DIR/DIR_TREES;
rm ./LA_Trees_NEWAL_NUMBER.root;

cd WORK_DIR;

rm ./JOB;

mv ./JLIST CRAFT_Repro_Lists/
