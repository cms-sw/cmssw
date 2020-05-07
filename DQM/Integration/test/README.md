# Unit tests of Online DQM clients

There is a dedicated input source for the unit tests: `DQM.Integration.config.unittestinputsource_cfi`.

The input source selects only last 100 events of 1st and 2nd lumisections to make sure that the tests can run fast yet encountering a lumi transition.

The most recent full event data was selected to be used to run the tests. Bellow are the instructions on how to update it:

``` bash
# Get the workflow number:
runTheMatrix.py -n | grep 2020
# And get the info about the workflow:
runTheMatrix.py -l 138.1 -ne
# Dataset and run number will appear in the output
# /ExpressCosmics/Commissioning2019-Express-v1/FEVT
# 334393
```

Dataset and run number has to be changed in the default values of the parameters in this file: `DQM/Integration/python/config/unittestinputsource_cfi.py`

## Running locally:

Running all tests:
``` bash
voms-proxy-init -voms cms -rfc
scram b runtests

# to run tests in parallel:
scram b -k -j 16 runtests
```

Running a single client test:
``` bash
cd DQM/Integration/python/clients
mkdir upload
cmsRun sistrip_dqm_sourceclient-live_cfg.py unitTest=True
```
