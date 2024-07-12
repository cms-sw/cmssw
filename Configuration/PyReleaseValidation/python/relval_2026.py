# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them
prefixDet = 29600 #update this line when change the default version

#2026 WFs to run in IB (TTbar)
numWFIB = []
numWFIB.extend([20034.0]) #2026D86
numWFIB.extend([20834.0]) #2026D88
numWFIB.extend([22034.0]) #2026D91
numWFIB.extend([22434.0]) #2026D92
numWFIB.extend([22834.0]) #2026D93
numWFIB.extend([23234.0]) #2026D94
numWFIB.extend([23634.0]) #2026D95
numWFIB.extend([24034.0]) #2026D96
numWFIB.extend([24434.0]) #2026D97
numWFIB.extend([24834.0]) #2026D98
numWFIB.extend([25234.0]) #2026D99
numWFIB.extend([25634.0]) #2026D100
numWFIB.extend([26034.0]) #2026D101
numWFIB.extend([26434.0]) #2026D102
numWFIB.extend([26834.0]) #2026D103
numWFIB.extend([27234.0]) #2026D104
numWFIB.extend([27634.0]) #2026D105
numWFIB.extend([28034.0]) #2026D106
numWFIB.extend([28434.0]) #2026D107
numWFIB.extend([28834.0]) #2026D108
numWFIB.extend([29234.0]) #2026D109
numWFIB.extend([29634.0]) #2026D110
numWFIB.extend([30034.0]) #2026D111
numWFIB.extend([30434.0]) #2026D112
numWFIB.extend([30834.0]) #2026D113
numWFIB.extend([31234.0]) #2026D114

#Additional sample for short matrix and IB
#Default Phase-2 Det NoPU
numWFIB.extend([prefixDet+34.911]) #DD4hep XML
numWFIB.extend([prefixDet+34.702]) #mkFit tracking (initialStep)
numWFIB.extend([prefixDet+34.5])   #pixelTrackingOnly
numWFIB.extend([prefixDet+34.9])   #vector hits
numWFIB.extend([prefixDet+34.501]) #Patatrack local reconstruction on CPU
numWFIB.extend([prefixDet+34.502]) #Patatrack local reconstruction on GPU
numWFIB.extend([prefixDet+34.21])  #prodlike
numWFIB.extend([prefixDet+96.0])   #CloseByPGun CE_E_Front_120um
numWFIB.extend([prefixDet+100.0])  #CloseByPGun CE_H_Coarse_Scint
numWFIB.extend([prefixDet+61.0])   #Nu Gun
#Default Phase-2 Det PU
numWFIB.extend([prefixDet+261.97])   #premixing stage1 (NuGun+PU)
numWFIB.extend([prefixDet+234.99])   #premixing combined stage1+stage2 ttbar+PU200
numWFIB.extend([prefixDet+234.999])  #premixing combined stage1+stage2 ttbar+PU50 for PR test
numWFIB.extend([prefixDet+234.21])   #prodlike PU
numWFIB.extend([prefixDet+234.9921]) #prodlike premix stage1+stage2
numWFIB.extend([prefixDet+234.114])  #PU, with 10% OT inefficiency

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
