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
numWFIB.extend([23234.0]) #2026D49
numWFIB.extend([23434.99,23434.999]) #2026D49 premixing combined stage1+stage2 (ttbar+PU200, ttbar+PU50 for PR test)
numWFIB.extend([23234.21,23434.21,23434.9921]) #2026D49 prodlike, prodlike PU, prodlike premix stage1+stage2
numWFIB.extend([28234.0]) #2026D60
numWFIB.extend([31434.0]) #2026D68
numWFIB.extend([32234.0]) #2026D70
numWFIB.extend([34634.0]) #2026D76
numWFIB.extend([35034.0]) #2026D77
numWFIB.extend([36234.0]) #2026D80
numWFIB.extend([36634.0]) #2026D81
numWFIB.extend([37034.0]) #2026D82
numWFIB.extend([37434.0]) #2026D83
numWFIB.extend([37834.0]) #2026D84
numWFIB.extend([38234.0]) #2026D85
numWFIB.extend([38634.0]) #2026D86
numWFIB.extend([39434.0,39434.911,39434.103]) #2026D88 DDD XML, DD4hep XML, aging
numWFIB.extend([39434.75]) #2026D88 with HLT75e33
numWFIB.extend([39661.97]) #2026D88 premixing stage1 (NuGun+PU)
numWFIB.extend([39434.5,39434.9,39434.501,39434.502]) #2026D88 pixelTrackingOnly, vector hits, Patatrack local reconstruction on CPU, Patatrack local reconstruction on GPU
numWFIB.extend([39634.99,39634.999]) #2026D88 premixing combined stage1+stage2 (ttbar+PU200, ttbar+PU50 for PR test)
numWFIB.extend([39434.21,39634.21,39634.9921]) #2026D88 prodlike, prodlike PU, prodlike premix stage1+stage2
numWFIB.extend([39634.114]) #2026D88 PU, with 10% OT ineffiency
numWFIB.extend([40634.0]) #2026D91

#Additional sample for short matrix and IB
#CloseByPGun for HGCAL
numWFIB.extend([39496.0]) #CE_E_Front_120um D88
numWFIB.extend([39500.0]) #CE_H_Coarse_Scint D88

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
