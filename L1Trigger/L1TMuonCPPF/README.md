# CPPF Digis Emulator

```
ssh -XY username@lxplus.cern.ch
setenv SCRAM_ARCH slc6_amd64_gcc530 (or export SCRAM_ARCH=slc6_amd64_gcc530)
cmsrel CMSSW_9_2_5_patch2
cd CMSSW_9_2_5_patch2/src
cmsenv
```

```
git cms-init
git cms-addpkg DataFormats/L1TMuon
git cms-addpkg L1Trigger/L1TCommon
git cms-addpkg L1Trigger/L1TMuon
git cms-addpkg L1Trigger/L1TMuonEndCap
git remote add cms-l1t-offline git@github.com:cms-l1t-offline/cmssw.git
git fetch cms-l1t-offline
git cms-merge-topic -u cms-l1t-offline:l1t-integration-CMSSW_9_2_5_patch2
git checkout cms-l1t-offline/l1t-integration-CMSSW_9_2_5_patch2
git clone https://github.com/cms-l1t-offline/L1Trigger-L1TMuon.git L1Trigger/L1TMuon/data
git fetch abrinke1
git cms-merge-topic -u abrinke1:CPPF_emul_9_2_5_patch2
scram b clean
scram b -j6
git fetch maseguracern
git cms-merge-topic -u maseguracern:CPPF_Emulator
scram b clean 
scram b -j6
```
## Run the code (check the input)
ls /eos/cms/store/data/Run2017F/SingleMuon/RAW/v1/000/306/125/00000/
cd L1Trigger/L1TMuonCPPF
cmsRun test/RunCPPF_GEN_UNPACKER_RAW.py

## Notes for Alejandro. Do not run these lines.

```
git remote add YourGitHubName git@github.com:YourGitHubName/cmssw.git
git fetch YourGitHubName
git checkout -b YourBranchName
```


```
cp -r /afs/cern.ch/user/m/masegura/public/CPPF_Emulator/DataFormats/L1TMuon ./DataFormats
cp -r /afs/cern.ch/user/m/masegura/public/CPPF_Emulator/L1TMuonCPPF ./L1Trigger
cp -r /afs/cern.ch/user/m/masegura/public/CPPF_Emulator/DataFormats/RPCDigi ./DataFormats
cp -r /afs/cern.ch/user/m/masegura/public/CPPF_Emulator/EventFilter/RPCRawToDigi ./EventFilter
cp -r /afs/cern.ch/user/m/masegura/public/CPPF_Emulator/DPGAnalysis .
scram b -j6
```

## Check the input file before to run.
```
cd L1Trigger/L1TMuonCPPF
cmsRun test/RunCPPF_GEN_UNPACKER_RAW.py
```

## Modify files
```
git add <Modified files>
git commit -m "Commit message"
git push my-cmssw YourBranchName
```
