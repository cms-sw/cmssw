#!/bin/bash

cd WORK_DIR;

eval `scramv1 runtime -sh`

python PY;
cmsRun PY;

#export STAGE_SVCCLASS=cmscaf
#export STAGER_TRACE=3

rfcp ./PY MY_CASTOR_DIR/DIR_PY;
rm ./PY;

rfcp ./MY_DEBUG_NUMBER.log MY_CASTOR_DIR/DIR_DEBUG;
rm ./MY_DEBUG_NUMBER.log;

cd MY_TMP/;

rfcp ./MY_HISTOS_NUMBER.root MY_CASTOR_DIR/DIR_HISTOS;
rm ./MY_HISTOS_NUMBER.root;

rfcp ./MY_HISTOS_HARV_NUMBER.root MY_CASTOR_DIR/DIR_HISTOHARV;
rm ./MY_HISTOS_HARV_NUMBER.root;

rfcp ./MY_TREE_NUMBER.root MY_CASTOR_DIR/DIR_TREES;
rm ./MY_TREE_NUMBER.root;

cd WORK_DIR;

rm ./JOB;

mv ./JLIST Source_Lists/
