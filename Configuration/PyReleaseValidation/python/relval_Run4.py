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

#Run4 WFs to run in IB (TTbar)
numWFIB = []
numWFIB.extend([23634.0]) #Run4D95
numWFIB.extend([24034.0]) #Run4D96
numWFIB.extend([24834.0]) #Run4D98
numWFIB.extend([25234.0]) #Run4D99
numWFIB.extend([25634.0]) #Run4D100
numWFIB.extend([26034.0]) #Run4D101
numWFIB.extend([26434.0]) #Run4D102
numWFIB.extend([26834.0]) #Run4D103
numWFIB.extend([27234.0]) #Run4D104
numWFIB.extend([27634.0]) #Run4D105
numWFIB.extend([28034.0]) #Run4D106
numWFIB.extend([28434.0]) #Run4D107
numWFIB.extend([28834.0]) #Run4D108
numWFIB.extend([29234.0]) #Run4D109
numWFIB.extend([29634.0]) #Run4D110
numWFIB.extend([30034.0]) #Run4D111
numWFIB.extend([30434.0]) #Run4D112
numWFIB.extend([30834.0]) #Run4D113
numWFIB.extend([31234.0]) #Run4D114
numWFIB.extend([32034.0]) #Run4D115
numWFIB.extend([32434.0]) #Run4D116
numWFIB.extend([32834.0]) #Run4D117
numWFIB.extend([33234.0]) #Run4D118
numWFIB.extend([33634.0]) #Run4D119
numWFIB.extend([34034.0]) #Run4D120
numWFIB.extend([34434.0]) #Run4D121

#Additional sample for short matrix and IB
#Default Phase-2 Det NoPU
numWFIB.extend([prefixDet+34.911]) #DD4hep XML
numWFIB.extend([prefixDet+34.702]) #mkFit tracking (initialStep)
numWFIB.extend([prefixDet+34.5])   #pixelTrackingOnly
numWFIB.extend([prefixDet+34.9])   #vector hits
numWFIB.extend([prefixDet+34.402]) #Alpaka local reconstruction offloaded on device (GPU if available)
numWFIB.extend([prefixDet+34.703]) #LST tracking on CPU (initialStep+HighPtTripletStep only)
numWFIB.extend([prefixDet+34.21])  #prodlike
numWFIB.extend([prefixDet+96.0])   #CloseByPGun CE_E_Front_120um
numWFIB.extend([prefixDet+100.0])  #CloseByPGun CE_H_Coarse_Scint
numWFIB.extend([prefixDet+61.0])   #Nu Gun
numWFIB.extend([prefixDet+34.75])  #Timing menu
numWFIB.extend([prefixDet+151.85]) #Heavy ion reconstruction
#Default Phase-2 Det PU
numWFIB.extend([prefixDet+261.97])   #premixing stage1 (NuGun+PU)
numWFIB.extend([prefixDet+234.99])   #premixing combined stage1+stage2 ttbar+PU200
numWFIB.extend([prefixDet+234.999])  #premixing combined stage1+stage2 ttbar+PU50 for PR test
numWFIB.extend([prefixDet+234.21])   #prodlike PU
numWFIB.extend([prefixDet+234.9921]) #prodlike premix stage1+stage2
numWFIB.extend([prefixDet+234.114])  #PU, with 10% OT inefficiency
numWFIB.extend([prefixDet+234.703])  #LST tracking on CPU (initialStep+HighPtTripletStep only)
#
numWFIB.extend([24834.911]) #D98 XML, to monitor instability of DD4hep

# Phase-2 HLT tests
numWFIB.extend([prefixDet+34.751]) # HLTTiming75e33, alpaka
numWFIB.extend([prefixDet+34.752]) # HLTTiming75e33, ticl_v5
numWFIB.extend([prefixDet+34.753]) # HLTTiming75e33, alpaka,singleIterPatatrack
numWFIB.extend([prefixDet+34.754]) # HLTTiming75e33, alpaka,singleIterPatatrack,trackingLST
numWFIB.extend([prefixDet+34.755]) # HLTTiming75e33, alpaka,trackingLST
numWFIB.extend([prefixDet+34.756]) # HLTTiming75e33, phase2_hlt_vertexTrimming
numWFIB.extend([prefixDet+34.7561])# HLTTiming75e33, alpaka,phase2_hlt_vertexTrimming
numWFIB.extend([prefixDet+34.7562])# HLTTiming75e33, alpaka,phase2_hlt_vertexTrimming,singleIterPatatrack
numWFIB.extend([prefixDet+34.757]) # HLTTiming75e33, alpaka,singleIterPatatrack,trackingLST,seedingLST
numWFIB.extend([prefixDet+34.759]) # HLTTiming75e33 + NANO
numWFIB.extend([prefixDet+34.77])  # NGTScouting
numWFIB.extend([prefixDet+34.771]) # NGTScouting + NANO

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
