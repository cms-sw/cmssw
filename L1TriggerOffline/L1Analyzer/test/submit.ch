#!/bin/csh -f

set cfgFile = $1
set runNum = $2
echo CFG FILE: ${cfgFile}
setenv outDir /castor/cern.ch/user/j/jbrooke/trigger/CRAFT/Calo
setenv runDir ${LS_SUBCWD}
setenv tmpDir /tmp
echo TEMPORARY DIRECTORY: ${tmpDir}
echo RUN DIRECTORY: ${runDir}


# setup
cd ${runDir}
eval `scramv1 runtime -csh`

# move to tmp dir and copy job config
cd ${tmpDir}
cp ${runDir}/${cfgFile} .

# run job
cmsRun ${cfgFile} >& ${runNum}.log

# copy output to CASTOR
rfcp ${runNum}.root ${outDir}/.
rfcp ${cfgFile} ${outDir}/.
rfcp ${runNum}.log ${outDir}/.

# cleanup
rm -f ${runNum}.root
rm -f ${runNum}.log
rm -f ${cfgFile}
rm -f core.*
