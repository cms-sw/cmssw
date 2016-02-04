#!/bin/csh

echo "=================================================="
echo "Installing Laser Monitoring directory from scratch"
setenv MACHINE `uname -n`
echo "=================================================="

echo "For machine: $MACHINE"

unalias cp
unalias rm

setenv MACHINEEE srv-C2D17-20
setenv MACHINEEBODD srv-C2D17-19
setenv MACHINEEBEVEN srv-C2D17-18


if( $MACHINE == $MACHINEEE ) then
  echo "EE assigned to this machine"
else if($MACHINE == $MACHINEEBODD ) then
  echo "EB Odd assigned to this machine"
else if($MACHINE == $MACHINEEBEVEN ) then
  echo "EB Even assigned to this machine"
else
  echo "Unknown machine"
  goto error
endif

date

# location of data files to be set by user:
#============================================

#setenv SORTING /cmsecallaser/srv-c2d17-18/disk0/sorting-reprocess #CRAFT
setenv SORTING /cmsecallaser/srv-c2d17-19/disk0/ecallaser/data/run_sorted/cosmics2009
setenv OUTPUTPATH /cmsecallaser/srv-c2d17-19/disk0/ecallaser/data/LM
setenv PRIMITIVES /nfshome0/ecallaser/LaserPrim
setenv PROD     Cosmics09_310
setenv STORE    Cosmics09_310 

setenv DATASORT ${SORTING}/out

setenv SCRIPTS  ${PWD}/CalibCalorimetry/EcalLaserAnalyzer/scripts
setenv SHAPEDIR ${PWD}/CalibCalorimetry/EcalLaserAnalyzer/data/sprshapes
setenv ALPHADIR ${PWD}/CalibCalorimetry/EcalLaserAnalyzer/data/alphabeta
setenv MUSECALDATADIR ${PWD}/CalibCalorimetry/EcalLaserAnalyzer/data/musecal
setenv PYDIR ${PWD}/CalibCalorimetry/EcalLaserAnalyzer/data/pytemplates
setenv MUSECAL ${PWD}/CalibCalorimetry/EcalLaserAnalyzer/test/MusEcal


${SCRIPTS}/mkdir.sh ${OUTPUTPATH}/${STORE} 'Monitoring directory'
${SCRIPTS}/mkdir.sh ${PRIMITIVES}/${STORE} 'Primitives directory'

${SCRIPTS}/lns.sh ${OUTPUTPATH}/${STORE} ${PROD}
${SCRIPTS}/lns.sh ${OUTPUTPATH}/${STORE} ${PROD}
${SCRIPTS}/lns.sh ${PRIMITIVES}/${STORE} ${PROD}/primitives
${SCRIPTS}/lns.sh ${MUSECAL} ${PROD}/musecal
${SCRIPTS}/lns.sh ${SCRIPTS} ${PROD}/scripts
${SCRIPTS}/lns.sh ${DATASORT} ${PROD}/sorting
${SCRIPTS}/lns.sh ${ALPHADIR} ${PROD}/alphabeta

${SCRIPTS}/mkdir.sh ${PROD}/shapes 'shapes directory'
${SCRIPTS}/mkdir.sh ${PROD}/templates 'templates directory'
${SCRIPTS}/mkdir.sh ${PROD}/meconfig 'meconfig directory'
${SCRIPTS}/mkdir.sh ${PROD}/log 'log directory'
${SCRIPTS}/lnsf.sh ${MUSECALDATADIR}/LVB.jpg ${PROD}/meconfig/LVB.jpg 

${SCRIPTS}/cp.sh ${MUSECALDATADIR}/MusEcal_EB.config ${PROD}/meconfig/${STORE}_EB.config
${SCRIPTS}/cp.sh ${MUSECALDATADIR}/MusEcal_EE.config ${PROD}/meconfig/${STORE}_EE.config

cp  $PYDIR/*py ${PROD}/templates/. 
cp  $SHAPEDIR/ElecMeanShape.root ${PROD}/shapes/. 

goto done

done:
     exit 0
error:
     exit 1
