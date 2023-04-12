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
numWFIB.extend([20034.0]) #2026D86
numWFIB.extend([20834.0]) #2026D88
numWFIB.extend([21061.97]) #2026D88 premixing stage1 (NuGun+PU)
numWFIB.extend([20834.5,20834.9,20834.501,20834.502]) #2026D88 pixelTrackingOnly, vector hits, Patatrack local reconstruction on CPU, Patatrack local reconstruction on GPU
numWFIB.extend([21034.99,21034.999]) #2026D88 premixing combined stage1+stage2 (ttbar+PU200, ttbar+PU50 for PR test)
numWFIB.extend([20834.21,21034.21,21034.9921]) #2026D88 prodlike, prodlike PU, prodlike premix stage1+stage2
numWFIB.extend([21034.114]) #2026D88 PU, with 10% OT ineffiency
numWFIB.extend([22034.0]) #2026D91
numWFIB.extend([22434.0]) #2026D92
numWFIB.extend([22834.0]) #2026D93
numWFIB.extend([23234.0]) #2026D94
numWFIB.extend([23634.0,23634.911,23634.103]) #2026D95 DDD XML, DD4hep XML, aging
numWFIB.extend([23861.97]) #2026D95 premixing stage1 (NuGun+PU)
numWFIB.extend([23634.5,23634.9]) #2026D95 pixelTrackingOnly, vector hits
numWFIB.extend([23834.99,23834.999]) #2026D95 premixing combined stage1+stage2 (ttbar+PU200, ttbar+PU50 for PR test) 
numWFIB.extend([23634.21,23834.21,23834.9921]) #2026D95 prodlike, prodlike PU, prodlike premix stage1+stage2
numWFIB.extend([24034.0]) #2026D96
numWFIB.extend([24434.0]) #2026D97
numWFIB.extend([24834.0]) #2026D98
numWFIB.extend([25234.0,25234.911]) #2026D99 DDD XML, DD4hep XML

#Additional sample for short matrix and IB
#CloseByPGun for HGCAL
numWFIB.extend([20896.0]) #CE_E_Front_120um D88
numWFIB.extend([20900.0]) #CE_H_Coarse_Scint D88
numWFIB.extend([23696.0]) #CE_E_Front_120um D95
numWFIB.extend([23700.0]) #CE_H_Coarse_Scint D95

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
