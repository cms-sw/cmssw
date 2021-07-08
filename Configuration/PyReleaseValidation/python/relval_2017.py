# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them

#WFs to run in IB:
#   2017 (ele guns 10, 35, 1000; pho guns 10, 35; mu guns 1, 10, 100, 1000, QCD 3TeV, QCD Flat)
#        (ZMM, TTbar, ZEE, MinBias, TTbar PU, ZEE PU, TTbar design)
#        (TTbar trackingOnly, trackingRun2, trackingOnlyRun2, trackingLowPU, pixelTrackingOnly)
#        (TTbar PU with JME NanoAOD)
#   2018 (ele guns 10, 35, 1000; pho guns 10, 35; mu guns 1, 10, 100, 1000, QCD 3TeV, QCD Flat)
#   2018 (ZMM, TTbar, ZEE, MinBias, TTbar PU, ZEE PU, TTbar design)
#        (TTbar trackingOnly, pixelTrackingOnly)
#        (HE collapse: TTbar, TTbar PU, TTbar design)
#        (ParkingBPH: TTbar)
#        (TTbar PU with JME NanoAOD)
#        (Patatrack pixel-only: ZMM - on CPU: quadruplets, triplets)
#        (Patatrack pixel-only: TTbar - on CPU: quadruplets, triplets)
#        (Patatrack ECAL-only: TTbar - on CPU)
#        (Patatrack HCAL-only: TTbar - on CPU)
#   2021 (DD4HEP: TTbar, ZMM)
#        (ele guns 10, 35, 1000; pho guns 10, 35; mu guns 1, 10, 100, 1000, QCD 3TeV, QCD Flat)
#        (ZMM, TTbar, ZEE, MinBias, TTbar PU, TTbar PU premix, ZEE PU, TTbar design, GluGluTo2Jets, GluGluTo2Jets PU)
#        (TTbar trackingOnly, pixelTrackingOnly, trackingMkFit, trackdnn)
#        (Patatrack pixel-only: ZMM - on CPU: quadruplets, triplets)
#        (Patatrack pixel-only: TTbar - on CPU: quadruplets, triplets)
#        (Patatrack ECAL-only: TTbar - on CPU)
#        (Patatrack HCAL-only: TTbar - on CPU)
#        (TTbar 0T, TTbar PU 0T)
#        (TTbar PU MLPF)
#        (QCD 1.8TeV DeepCore)
#   2023 (TTbar, TTbar PU, TTbar PU premix)
#   2024 (TTbar, TTbar PU, TTbar PU premix)
numWFIB = [10001.0,10002.0,10003.0,10004.0,10005.0,10006.0,10007.0,10008.0,10009.0,10059.0,10071.0,
           10042.0,10024.0,10025.0,10026.0,10023.0,10224.0,10225.0,10424.0,
           10024.1,10024.2,10024.3,10024.4,10024.5,
           10224.15,
           10801.0,10802.0,10803.0,10804.0,10805.0,10806.0,10807.0,10808.0,10809.0,10859.0,10871.0,
           10842.0,10824.0,10825.0,10826.0,10823.0,11024.0,11025.0,11224.0,
           10824.1,10824.5,
           10824.6,11024.6,11224.6,
           10824.8,
           11024.15,
           10842.501,10842.505,
           10824.501,10824.505,
           10824.511,
           10824.521,
           11634.911, 11650.911,
           # 11634.912, # Wait until CondDB issues resolved
           11601.0,11602.0,11603.0,11604.0,11605.0,11606.0,11607.0,11608.0,11609.0,11630.0,11643.0,
           11650.0,11634.0,11646.0,11640.0,11834.0,11834.99,11846.0,12034.0,11725.0,11925.0,
           11634.1,11634.5,11634.7,11634.91,
           11650.501,11650.505,
           11634.501,11634.505,
           11634.511,
           11634.521,
           11634.24,11834.24,
           11834.13,
           11723.17,
           12434.0,12634.0,12634.99,
           12834.0,13034.0,13034.99,]
for numWF in numWFIB:
    if not numWF in _upgrade_workflows: continue
    workflows[numWF] = _upgrade_workflows[numWF]
