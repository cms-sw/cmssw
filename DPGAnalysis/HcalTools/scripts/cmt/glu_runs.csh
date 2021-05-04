#!/bin/tcsh

set DIR=${1}
set NRUNS=$#argv

echo ${NRUNS}
set i=2
while( ${i} <= ${NRUNS} )
set NRUN=$argv[${i}]
set list=`ls ${DIR}/Global_${NRUN}_*.root`
echo ${list}
hadd Global_${NRUN}.root ${list}
@ i = ${i} + "1"
end

