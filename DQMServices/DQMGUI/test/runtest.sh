#!/bin/bash

# Print all commands, fail on any error.
set -e
set -x
set -o pipefail

# Some integration tests for DQMGUI.

if [[ -z ${LOCAL_TEST_DIR} ]]; then
    export LOCAL_TEST_DIR=.
fi

# 0. Create some test data. This should make a DQMIO and a classic DQM file with
# contents representative of current DQM output.
# TODO: hack in customize here to also make a PB file.
if [ ! -f */DQM_*.root ]; then # avoid this in local tests.
  runTheMatrix.py -l 136.891 --command '-n 1' --ibeos
fi
# These values come from the workflow above
RUN=320822
LUMI=75

# 1. Start up a GUI: in the background, so we can interact with it
# set -e will kill the script if it fails, even in background.

python3 $LOCAL_TEST_DIR/../python/app.py -p 7000 -r 1 --in-memory -f &

# Finally: clean shutdown.
stopgui() {
  kill -INT %1
  wait 
}

trap stopgui EXIT

# Give it some time to start up
sleep 5
# Check if the server is alive at all
curl http://localhost:7000/ > /dev/null

# 2. Register some files.
curl -s 'http://localhost:7000/api/v1/register' --data '[{"dataset": "/Unit/Test/DQMCLASSIC", "run": "'$RUN'", "lumi": "0", "file": "'$(readlink -f */DQM_V0001_R000320822__Global__CMSSW_X_Y_Z__RECO.root)'", "fileformat": 1}]'
curl -s 'http://localhost:7000/api/v1/register' --data '[{"dataset": "/Unit/Test/DQMIO", "run": "'$RUN'", "lumi": "'$LUMI'", "file": "'$(readlink -f */step3_inDQM.root)'", "fileformat": 2}]'

# 3. Test if we have the samples.
curl -s 'http://localhost:7000/api/v1/samples' | python -m json.tool | grep /Unit/Test/DQMCLASSIC
# not a complete test of filtering but better than nothing
curl -s 'http://localhost:7000/api/v1/samples?run=7' | python -m json.tool | fgrep '"data": []' # no results
# TODO: fails atm
#curl -s 'http://localhost:7000/api/v1/samples?dataset=IO' | python -m json.tool | fgrep '"data": []' # no results
  
# 4. Trigger an import.
time curl -s 'http://localhost:7000/api/v1/archive/'$RUN'/Unit/Test/DQMCLASSIC/'
time curl -s 'http://localhost:7000/api/v1/archive/'$RUN:$LUMI'/Unit/Test/DQMIO/'

# 5. Load the APIs a bit. (Uses the legacy APIs.)
python3 aioloadgen.py





