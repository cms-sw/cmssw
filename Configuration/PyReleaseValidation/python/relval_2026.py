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
numWFIB.extend([23461.97]) #2026D49 premixing stage1 (NuGun+PU)
numWFIB.extend([23434.99,23434.999]) #2026D49 premixing combined stage1+stage2 (ttbar+PU200, ttbar+PU50 for PR test)
numWFIB.extend([23234.21,23434.21]) #2026D49 prodlike, prodlike PU
numWFIB.extend([23234.103]) #2026D49 aging
numWFIB.extend([23234.9]) #2026D49 vector hits
numWFIB.extend([28234.0]) #2026D60
numWFIB.extend([29834.0]) #2026D64
numWFIB.extend([30234.0]) #2026D65
numWFIB.extend([30634.0]) #2026D66
numWFIB.extend([31034.0]) #2026D67
numWFIB.extend([31434.0]) #2026D68
numWFIB.extend([31834.0]) #2026D69
numWFIB.extend([32234.0]) #2026D70
numWFIB.extend([32634.0]) #2026D71

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
