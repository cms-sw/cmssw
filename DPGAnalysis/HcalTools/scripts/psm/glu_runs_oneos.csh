#!/bin/tcsh

set DIR=${1}
set LIST=${2}
echo ${DIR}
echo ${LIST}

foreach i (`cat ${LIST}`)
set NRUN=${i}
echo ${NRUN}
set list=`ls ${DIR}/${NRUN}/HcalNZS/crab_*_*/*_*/0000/Global_${NRUN}_*.root`
echo ${list}
echo " start hadd   "
hadd Global_${NRUN}.root ${list}
@ i = ${i} + "1"
end

