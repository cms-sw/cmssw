#!/bin/tcsh

set DIR=${1}
set LIST=${2}

foreach i (`cat ${LIST}`)
set NRUN=${i}
set list=`ls ${DIR}/Global_${NRUN}_*.root`
echo ${list}
hadd Global_${NRUN}.root ${list}
@ i = ${i} + "1"
end

