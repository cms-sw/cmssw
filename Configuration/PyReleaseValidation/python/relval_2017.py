
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
#   2018 (ele guns 10, 35, 1000; pho guns 10, 35; mu guns 1, 10, 100, 1000, QCD 3TeV, QCD Flat)
#   2018 (ZMM, TTbar, ZEE, MinBias, TTbar PU, ZEE PU, TTbar design)
#        (TTbar trackingOnly, pixelTrackingOnly)
#         he collapse: TTbar, TTbar PU, TTbar design
#   2019 (ZMM, TTbar, ZEE, MinBias, TTbar PU, ZEE PU, TTbar design)
numWFIB = [10001.0,10002.0,10003.0,10004.0,10005.0,10006.0,10007.0,10008.0,10009.0,10059.0,10071.0,
           10042.0,10024.0,10025.0,10026.0,10023.0,10224.0,10225.0,10424.0,
           10024.1,10024.2,10024.3,10024.4,10024.5,
           10801.0,10802.0,10803.0,10804.0,10805.0,10806.0,10807.0,10808.0,10809.0,10859.0,10871.0,
           10842.0,10824.0,10825.0,10826.0,10823.0,11024.0,11025.0,11224.0,
           10824.1,10824.5,10824.7,10824.8,10824.9,
           10824.6,11024.6,11224.6,
           11642.0,11624.0,11625.0,11626.0,11623.0,11824.0,11825.0,12024.0]
for numWF in numWFIB:
    if not numWF in _upgrade_workflows: continue
    workflows[numWF] = _upgrade_workflows[numWF]
