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
numWFIB.extend([23434.99]) #2026D49 premixing combined stage1+stage2 (ttbar+PU)
numWFIB.extend([23234.21,23434.21]) #2026D49 prodlike, prodlike PU
numWFIB.extend([23234.103]) #2026D49 aging
numWFIB.extend([23634.0]) #2026D51
numWFIB.extend([24834.0]) #2026D54
numWFIB.extend([26634.0]) #2026D56
numWFIB.extend([27034.0]) #2026D57
numWFIB.extend([27434.0]) #2026D58
numWFIB.extend([27834.0]) #2026D59
numWFIB.extend([28234.0]) #2026D60
numWFIB.extend([28634.0]) #2026D61
numWFIB.extend([29034.0]) #2026D62

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
