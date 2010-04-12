#!/bin/bash

echo "=================================================="
echo "Installing Laser Monitoring directory from scratch"
export MACHINE=`uname -n`
echo "=================================================="


if [ "$1" = "reproc" ]; then
  source /nfshome0/ecallaser/config/lmf_cfg_reproc
elif [ "$1" = "dev" ]; then
  source /nfshome0/ecallaser/config/lmf_cfg_dev
else
  source /nfshome0/ecallaser/config/lmf_cfg
fi

echo " sorting dir: $SORT_WORKING_DIR"
echo " monitoring dir: $MON_WORKING_DIR"
echo " monitoring release: $MON_CMSSW_REL_DIR"
echo " monitoring code: $MON_CMSSW_CODE_DIR"
echo " calib path: $MON_CALIB_PATH"
echo " abinit path: $MON_AB_PATH"

date

export PROD=$LMF_LASER_PERIOD
export STORE=$LMF_LASER_PERIOD 

export SCRIPTS="${MON_CMSSW_REL_DIR}/CalibCalorimetry/EcalLaserAnalyzer/scripts"

export MUSECALDATADIR="${MON_CMSSW_REL_DIR}/CalibCalorimetry/EcalLaserAnalyzer/data/musecal"
export PYDIR="${MON_CMSSW_REL_DIR}/CalibCalorimetry/EcalLaserAnalyzer/data/pytemplates"
export MUSECAL="${MON_CMSSW_REL_DIR}/CalibCalorimetry/EcalLaserAnalyzer/test/MusEcal"

${SCRIPTS}/mkdir.sh ${MON_OUTPUT_DIR}/${STORE} 'Monitoring directory'
${SCRIPTS}/mkdir.sh ${LMF_LASER_PRIM_DIR}/${STORE} 'Primitives directory'
${SCRIPTS}/mkdir.sh ${MON_OUTPUT_DIR}/${STORE}/log 'log directory'

${SCRIPTS}/lns.sh ${MON_OUTPUT_DIR}/${STORE} ${PROD}

echo " ... done " 
echo "=================================================="


#${SCRIPTS}/lns.sh ${LMF_LASER_PRIM_DIR}/${STORE} ${PROD}/primitives
#${SCRIPTS}/lns.sh ${MUSECAL} ${PROD}/musecal
#${SCRIPTS}/lns.sh ${SCRIPTS} ${PROD}/scripts
#${SCRIPTS}/lns.sh ${SORT_WORKING_DIR} ${PROD}/sorting
#${SCRIPTS}/lns.sh ${ALPHADIR} ${PROD}/alphabeta
#${SCRIPTS}/lns.sh ${CALIBPATH} ${PROD}/calibpath
#${SCRIPTS}/mkdir.sh ${PROD}/shapes 'shapes directory'
#${SCRIPTS}/mkdir.sh ${PROD}/templates 'templates directory'
${SCRIPTS}/mkdir.sh ${PROD}/meconfig 'meconfig directory'

#${SCRIPTS}/lnsf.sh ${MUSECALDATADIR}/LVB.jpg ${PROD}/meconfig/LVB.jpg 
${SCRIPTS}/cp.sh ${MUSECALDATADIR}/MusEcal_EB.config ${PROD}/meconfig/${STORE}_EB.config
${SCRIPTS}/cp.sh ${MUSECALDATADIR}/MusEcal_EE.config ${PROD}/meconfig/${STORE}_EE.config

#cp  $PYDIR/all.py ${PROD}/templates/. 
#cp  $PYDIR/ab.py ${PROD}/templates/. 


#exit 0
