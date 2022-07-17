#!/bin/tcsh

set DIR=${1}
set LIST=${2}
set DIROUT=${3}
echo ${DIR}
echo ${LIST}
echo ${DIROUT}

foreach i (`cat ${LIST}`)
set NRUN=${i}
echo ${NRUN}
echo " start copying" ${NRUN}
eoscp ${DIR}/${NRUN}/HcalNZS/crab_*_*/*_*/0000/Global_${NRUN}_*.root ${DIROUT}/.
@ i = ${i} + "1"
end
