# Running tests on lxplus


## Using scram

To be able to run tests locally enable re-director used internally by `cmsbuild`:

```
CMS_PATH="/cvmfs/cms-ib.cern.ch" SITECONFIG_PATH="/cvmfs/cms-ib.cern.ch/SITECONF/local" scram b runtests
```

## Manually (to inspect test output files)

To run the tests in this directory (`Calibration/PPSAlCaRecoProducer/test/`) type:

```
CMS_PATH="/cvmfs/cms-ib.cern.ch" SITECONFIG_PATH="/cvmfs/cms-ib.cern.ch/SITECONF/local" ./test_express_AlCaRecoProducer.sh
```

and

```
SCRAM_TEST_PATH=. ./test_express_PPSAlCaReco_output.sh 
```

The same can be done for prompt tests
