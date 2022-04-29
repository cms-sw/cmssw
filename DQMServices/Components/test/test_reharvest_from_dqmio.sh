# DQMIO dataset to take data from and run to look at
RUN=302663
DATASET=/SingleElectron/Run2017D-09Aug2019_UL2017-v1/DQMIO
# Workflow to take the HARVESTING step from
WORKFLOW=136.834

# run cmsDriver to generate baseline harvesting config
$(runTheMatrix.py -l $WORKFLOW -ne | fgrep 'HARVESTING:' | grep -o 'cmsDriver.*') --no_exec
pythonname=$(echo step*_HARVESTING.py)

# copy data to local folder and add it to config
./dqmiofilecopy.sh $DATASET $RUN >> $pythonname

# no idea where this is injected in the prod setup, but it must be somewhere.
echo "process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('$RUN:1-$(($RUN+1)):0')" >> $pythonname

echo '### Got all data, starting cmsRun!'

cmsRun $pythonname

