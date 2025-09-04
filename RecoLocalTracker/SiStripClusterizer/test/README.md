## Installation for proposed method
```sh
  cmsrel CMSSW_15_0_10
  cd CMSSW_15_0_10/src/
  cmsenv
  git cms-init
  git pull git@github.com:saswatinandan/cmssw.git HI_test
  git cms-addpkg Configuration/Eras DataFormats/SiStripCluster
  git cms-addpkg RecoLocalTracker/SiStripClusterizer
  scram b -j 8
  
  copy input from /afs/cern.ch/work/s/snandan/public/hackathon/CMSSW_14_1_5/src/RecoLocalTracker/SiStripClusterizer/test/e1f7f325-4ca8-4fea-8d27-06b33862cf11.root
  cd RecoLocalTracker/SiStripClusterizer/test
  cmsRun step2_L1REPACK_HLT_rawp.py 
```
## To run default method
```
   cmsrel CMSSW_15_0_10
   cd CMSSW_15_0_10/src/
   cmsenv
   copy input from /afs/cern.ch/work/s/snandan/public/hackathon/CMSSW_14_1_5/src/RecoLocalTracker/SiStripClusterizer/test/e1f7f325-4ca8-4fea-8d27-06b33862cf11.root
   copy python file step2_L1REPACK_HLT_rawp.py from previous step and run
   cmsRun step2_L1REPACK_HLT_rawp.py

```
## Compare strip cluster size
```
   edmEvenSize -v outputfile | grep Approx
```
