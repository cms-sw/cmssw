# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them

#2026 WFs to run in IB (TTbar, TTbar+Timing)
numWFIB = [20034.0] #2026D35
numWFIB.extend([20434.0,20434.1]) #2026D41 w/ special tracking and timing workflows
numWFIB.extend([20661.97]) # 2026D41 premixing stage1 (NuGun+PU)
numWFIB.extend([20634.99]) # 2026D41 premixing combined stage1+stage2 (ttbar+PU)
numWFIB.extend([20434.21,20634.21]) #2026D41 prodlike, prodlike PU
numWFIB.extend([20434.103]) #2026D41 aging
numWFIB.extend([20493.52]) #2026D41+TICL
numWFIB.extend([20834.0]) #2026D43
numWFIB.extend([21234.0]) #2026D44
numWFIB.extend([21634.0]) #2026D45
numWFIB.extend([22034.0]) #2026D46
numWFIB.extend([22434.0]) #2026D47
numWFIB.extend([22834.0]) #2026D48
numWFIB.extend([23234.0,23234.1001,23434.1001]) #2026D49, TestOldDigi, TestOldDigi w/ PU
numWFIB.extend([23634.0]) #2026D51
numWFIB.extend([24034.0]) #2026D52

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
