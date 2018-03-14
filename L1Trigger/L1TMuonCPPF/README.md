# CPPFDigis Emulator

This is the first version of the CPPF emulator. we use the RPC Digitization and 
reconstruction as intermedate steps. 
Under test you can find two examples, one unpacking 2017F RAW Data (cppf_emulator_RAW.py)
and another one using generated MC events (cppf_emulator_MC.py). 
The output of the unpacker is an edm branch named emulatorCppfDigis, following
the CPPFDigi dataformat already committed in CMSSW_10_1_X.

# Out of the box instructions

```
ssh -XY username@lxplus.cern.ch
#setenv SCRAM_ARCH slc6_amd64_gcc630 
#(or export SCRAM_ARCH=slc6_amd64_gcc630)
cmsrel CMSSW_10_1_X_2018-03-11-2300
cd CMSSW_10_1_X_2018-03-11-2300/src
cmsenv
```

```
git cms-init
git fetch maseguracern
git cms-merge-topic -u maseguracern:CPPF_Emulator
#scram b clean 
scram b -j6
```

## Run the code (check the input)
```
cd L1Trigger/L1TMuonCPPF
cmsRun test/cppf_emulator_RAW.py
```

## Setup your Github space (In case you haven't)
```
git remote add YourGitHubName git@github.com:YourGitHubName/cmssw.git
git fetch YourGitHubName
git checkout -b YourBranchName
```

## Modifying files
```
git add <Modified files>
git commit -m "Commit message"
git push my-cmssw YourBranchName
```
