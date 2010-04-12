#!/bin/csh
#cd $2
#cmsRun $1.cfg >& $1.log
setenv PROD     $1
setenv SM       $2
#echo "Coucou" $PROD $SM
rm -r -f ${PROD}/${SM}
echo "Create directory structure for ECAL module " ${SM}
mkdir ${PROD}/${SM}
mkdir ${PROD}/${SM}/Laser
mkdir ${PROD}/${SM}/Laser/Analyzed
mkdir ${PROD}/${SM}/Laser/Analyzed/Failed
mkdir ${PROD}/${SM}/TestPulse
mkdir ${PROD}/${SM}/TestPulse/Analyzed
mkdir ${PROD}/${SM}/TestPulse/Analyzed/Failed
mkdir ${PROD}/${SM}/LED
mkdir ${PROD}/${SM}/LED/Analyzed
mkdir ${PROD}/${SM}/LED/Analyzed/Failed
mkdir ${PROD}/${SM}/Runs
mkdir ${PROD}/${SM}/Runs/Detected
mkdir ${PROD}/${SM}/Runs/Analyzed
mkdir ${PROD}/${SM}/Runs/Analyzed/Failed

#mkdir ${PROD}/${SM}/Pedestal
#mkdir ${PROD}/${SM}/Pedestal/Detected
#mkdir ${PROD}/${SM}/Pedestal/Analyzed
#mkdir ${PROD}/${SM}/Pedestal/Analyzed/Failed
