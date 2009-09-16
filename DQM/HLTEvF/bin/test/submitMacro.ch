#!/bin/csh -f

set DBFILE = $1
echo ${DBFILE}
set DETID = $2
echo ${DETID}
set HISTO = $3
echo ${HISTO}
set PAR = $4
echo ${PAR}
#setenv runDir ${HOME}/scratch0/CMSSW_3_2_4/src/DQM/HLTEvF/bin/test
setenv runDir `pwd`
echo RUN DIRECTORY: ${runDir}
cd ${runDir}

eval `scramv1 runtime -csh`
#root -b << EOF
#.L Inspector.C 
#Inspector("${DBFILE}",${DETID},"${HISTO}","${PAR}")
#EOF

cmsenv
myInspector "${DBFILE}" ${DETID} "${HISTO}" "${PAR}" >& ${HISTO}.log
rm -f core.*

#mv historicDQM.root  ${HISTO}
#mv ${DBFILE} ${HISTO}
#mv ${HISTO}@*.gif ${HISTO}
#mv ${HISTO}.log ${HISTO}
