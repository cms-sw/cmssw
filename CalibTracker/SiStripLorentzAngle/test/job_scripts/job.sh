#!/bin/bash

cd WORK_DIR;

eval `scramv1 runtime -sh`

cmsRun CFG;

export STAGE_SVCCLASS=cmscaf
export STAGER_TRACE=3

rfcp ./CFG MY_CASTOR_DIR/DIR_CFG;
rm ./CFG;

rfcp ./LA_debug_FILETAG_NUMBER.log MY_CASTOR_DIR/DIR_DEBUG;
rm ./LA_debug_FILETAG_NUMBER.log;

cd MY_TMP/;

rfcp ./LA_Histos_FILETAG_NUMBER.root MY_CASTOR_DIR/DIR_HISTOS;
rm ./LA_Histos_FILETAG_NUMBER.root;

rfcp ./LA_Trees_FILETAG_NUMBER.root MY_CASTOR_DIR/DIR_TREES;
rm ./LA_Trees_FILETAG_NUMBER.root;

rfcp ./Fit_FILETAG_NUMBER.txt MY_CASTOR_DIR/DIR_FITS;
rm ./Fit_FILETAG_NUMBER.txt;

cd WORK_DIR;

rm ./JOB;
