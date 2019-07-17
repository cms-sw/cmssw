1) Basic Instructions

```
cmsrel CMSSW_10_1_0_pre3
cd CMSSW_10_1_0_pre3/src
cmsenv
git cms-init
git remote add cms-l1t-offline git@github.com:cms-l1t-offline/cmssw.git
git fetch cms-l1t-offline phase2-l1t-integration-CMSSW_10_1_0_pre3
git cms-merge-topic -u cms-l1t-offline:l1t-phase2-v2.7

#
# Tracklet Tracks
#
git remote add rekovic git@github.com:rekovic/cmssw.git
git fetch rekovic Tracklet-10_1_0_pre3
git cms-merge-topic -u rekovic:Tracklet-10_1_0_pre3-from-skinnari

# Remove tracklets from history
git reset --soft cms-l1t-offline/l1t-phase2-v2.7
git reset HEAD L1Trigger

# Get L1PF_CMSSW
git remote add p2l1pfp git@github.com:p2l1pfp/cmssw.git
git fetch p2l1pfp L1PF_CMSSW
echo -e '/DataFormats/Phase2L1ParticleFlow/\n/L1Trigger/Phase2L1ParticleFlow/' >> .git/info/sparse-checkout
git read-tree -mu HEAD
git checkout -b L1PF_CMSSW p2l1pfp/L1PF_CMSSW

scram b -j8
```

2) Ntuple for jets, jet HT and MET studies

```
cmsRun test/l1pfJetMetTreeProducer.py
```
