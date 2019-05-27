# RPCTriggerPrimitives

This is the first version of the RPCTriggerPrimitives for the RPC detector. Which uses as a input the RPCDigis as an input and the RPCRecHits (New collection) as a output.
The output of this module is an edm branch named RPCPrimitivesDigis, following the RPCRecHit format already committed in CMSSW_10_6_0_pre1.
We apply the cluster size cut and emulate max two clusters per link board. The module can be tuned by the parameters LinkBoardCut and ClusterSizeCut. 

#  Out of the box instructions

```
ssh -XY username@lxplus7.cern.ch
SCRAM_ARCH=slc7_amd64_gcc700; export SCRAM_ARCH   (in .bashrc file)
scram list CMSSW_10_6_0
cmsrel CMSSW_10_6_0
cd CMSSW_10_6_0/src
cmsenv
```

```
You need to do a fork from your githut repository to cmssw-offline repository. The url is:
https://github.com/cms-sw/cmssw
```


```
git cms-init
git cms-addpkg L1Trigger/L1TMuonEndCap
git cms-addpkg CondTools/RPC

cd CondTools/RPC
cp /eos/cms/store/group/dpg_rpc/comm_rpc/Run-II/cppf_payloads/RPCLinkMap.db data
for analyser in test/RPC*LinkMapPopConAnalyzer_cfg.py; do 
  cmsRun $analyser
done; # <- this produces RPCLinkMap.db sqlite file yourself
cd -

git remote add YourGitHubName git@github.com:YourGitHubName/cmssw.git
git fetch YourGitHubName
git checkout -b PrimitiveTrigger
scram b -j6

```

## Set your environment with my branch

```
git remote add maseguracern git@github.com:maseguracern/cmssw.git
git cms-merge-topic -u maseguracern:PrimitiveTrigger 
scram b -j6
```

# Run test producer
```
cd test
cmsRun rpcprimitive_MC.py
```

## Modifying files
```
git add <Modified files>
git commit -m "Commit message"
git push my-cmssw YourBranchName
```

