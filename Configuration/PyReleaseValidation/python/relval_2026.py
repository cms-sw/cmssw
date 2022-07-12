# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them

#2026 WFs to run in IB (TTbar)
numWFIB = []
numWFIB.extend([38634.0]) #2026D86
numWFIB.extend([39434.0,39434.911,39434.103]) #2026D88 DDD XML, DD4hep XML, aging
numWFIB.extend([39434.75]) #2026D88 with HLT75e33
numWFIB.extend([39661.97]) #2026D88 premixing stage1 (NuGun+PU)
numWFIB.extend([39434.5,39434.9,39434.501,39434.502]) #2026D88 pixelTrackingOnly, vector hits, Patatrack local reconstruction on CPU, Patatrack local reconstruction on GPU
numWFIB.extend([39634.99,39634.999]) #2026D88 premixing combined stage1+stage2 (ttbar+PU200, ttbar+PU50 for PR test)
numWFIB.extend([39434.21,39634.21,39634.9921]) #2026D88 prodlike, prodlike PU, prodlike premix stage1+stage2
numWFIB.extend([39634.114]) #2026D88 PU, with 10% OT ineffiency
numWFIB.extend([40634.0]) #2026D91
numWFIB.extend([41034.0]) #2026D92
numWFIB.extend([41434.0]) #2026D93
numWFIB.extend([41834.0]) #2026D94

#Additional sample for short matrix and IB
#CloseByPGun for HGCAL
numWFIB.extend([39496.0]) #CE_E_Front_120um D88
numWFIB.extend([39500.0]) #CE_H_Coarse_Scint D88

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
