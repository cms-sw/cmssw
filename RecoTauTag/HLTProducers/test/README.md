## Tests for tau HLT producers

### Setup to tes L2TauNNTag
You need to have this repository in CMSSW_12_1_0_pre4/src (or newer). If you don't have it, the command to run is:
```
export SCRAM_ARCH=slc7_amd64_gcc900
cmsrel CMSSW_12_1_0_pre4
cd CMSSW_12_1_0_pre4/src
cmsenv
git cms-merge-topic cms-tau-pog:CMSSW_12_1_X_tau-pog_L2wCNN
scram b -j 8
```

You also need the RecoTauTag-Training files. If you don't have it, run the command:
```
git clone -b L2_CNN_v1 git@github.com:cms-tau-pog/RecoTauTag-TrainingFiles.git RecoTauTag/TrainingFiles/data
```

### L2TauNNTag test
You need to authenticate to run the test command since it exploits files that are saved on store. So you need to run the command:
```
voms-proxy-init --rfc --voms cms
```

You are now ready to run the test for L2TauTagNN
```
cd CMSSW_12_1_0_pre4/src
cmsenv
scram b -j 8
cmsRun RecoTauTag/HLTProducers/test/testL2TauTagNN.py
```

if you want to run on a limited number of events:
```
cmsRun RecoTauTag/HLTProducers/test/testL2TauTagNN.py maxEvents=20
```
