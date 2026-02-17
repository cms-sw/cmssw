# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them
prefixDet = 34400 #update this line when change the default version

#Run4 WFs to run in IB (TTbar)
numWFIB = []
numWFIB.extend([27234.0]) #Run4D104
numWFIB.extend([29634.0]) #Run4D110
numWFIB.extend([30034.0]) #Run4D111
numWFIB.extend([30434.0]) #Run4D112
numWFIB.extend([30834.0]) #Run4D113
numWFIB.extend([31234.0]) #Run4D114
numWFIB.extend([32034.0]) #Run4D115
numWFIB.extend([34034.0]) #Run4D120
numWFIB.extend([34434.0]) #Run4D121
numWFIB.extend([34834.0]) #Run4D122
numWFIB.extend([35234.0]) #Run4D123
numWFIB.extend([35634.0]) #Run4D124
numWFIB.extend([36034.0]) #Run4D125

#Additional sample for short matrix and IB
#Default Phase-2 Det NoPU
numWFIB.extend([prefixDet+34.911]) #DD4hep XML
numWFIB.extend([prefixDet+34.702]) #mkFit tracking (initialStep)
numWFIB.extend([prefixDet+34.5])   #pixelTrackingOnly
numWFIB.extend([prefixDet+34.9])   #vector hits
numWFIB.extend([prefixDet+34.402]) #Alpaka local reconstruction offloaded on device (GPU if available)
numWFIB.extend([prefixDet+34.711]) #LST tracking on CPU (initialStep+HighPtTripletStep only)
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
numWFIB.extend([prefixDet+234.711])  #LST tracking on CPU (initialStep+HighPtTripletStep only)

# Phase-2 HLT tests
numWFIB.extend([prefixDet+34.7501])# HLTTrackingOnly75e33
numWFIB.extend([prefixDet+34.751]) # HLTTiming75e33, alpaka
numWFIB.extend([prefixDet+34.7511])# HLTTiming75e33, phase2CAExtension
numWFIB.extend([prefixDet+34.752]) # HLTTiming75e33, ticl_v5
numWFIB.extend([prefixDet+34.7521])# HLTTiming75e33, ticl_v5, ticlv5TrackLinkingGNN
numWFIB.extend([prefixDet+34.753]) # HLTTiming75e33, alpaka,singleIterPatatrack
numWFIB.extend([prefixDet+34.754]) # HLTTiming75e33, alpaka,singleIterPatatrack,trackingLST
numWFIB.extend([prefixDet+34.755]) # HLTTiming75e33, alpaka,trackingLST
numWFIB.extend([prefixDet+34.756]) # HLTTiming75e33, phase2_hlt_vertexTrimming
numWFIB.extend([prefixDet+34.7561])# HLTTiming75e33, alpaka,phase2_hlt_vertexTrimming
numWFIB.extend([prefixDet+34.7562])# HLTTiming75e33, alpaka,phase2_hlt_vertexTrimming,singleIterPatatrack
numWFIB.extend([prefixDet+34.757]) # HLTTiming75e33, alpaka,singleIterPatatrack,trackingLST,seedingLST
numWFIB.extend([prefixDet+34.7571]) # HLTTiming75e33, alpaka,singleIterPatatrack,Phase2CAExtension,trackingLST,seedingLST,buildingMkFit
numWFIB.extend([prefixDet+34.7572]) # HLTTiming75e33, alpaka,singleIterPatatrack,Phase2CAExtension,trackingLST,seedingLST,buildingMkFit,fittingMkFit
numWFIB.extend([prefixDet+34.758]) # HLTTiming75e33, ticl_barrel
numWFIB.extend([prefixDet+34.759]) # HLTTiming75e33 + NANO
numWFIB.extend([prefixDet+34.77])  # NGTScouting
numWFIB.extend([prefixDet+34.771]) # NGTScouting + alpaka + TICL-v5 + TICL-Barrel
numWFIB.extend([prefixDet+34.772]) # NGTScouting + NANO
numWFIB.extend([prefixDet+34.773]) # NGTScouting + NANO (including validation)
numWFIB.extend([prefixDet+34.774])  # NGTScouting + Phase2CAExtension as GeneneralTracks
numWFIB.extend([prefixDet+34.775])  # NGTScouting + Phase2CAExtension&LSTT5 as GeneralTracks

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
