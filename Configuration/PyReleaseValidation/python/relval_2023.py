# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them

#2023 WFs to run in IB (TTbar, TTbar+Timing)
numWFIB = [20034.0,20034.1,20034.11] #2023D17 w/ special tracking and timing workflows
numWFIB.extend([20261.97]) # 2023D17 premixing stage1 (NuGun+PU)
numWFIB.extend([20234.99]) # 2023D17 premixing combined stage1+stage2 (ttbar+PU)
numWFIB.extend([20434.0]) #2023D19 (already has timing)
numWFIB.extend([21234.0,21234.11]) #2023D21
numWFIB.extend([21634.0]) #2023D22
numWFIB.extend([22034.0]) #2023D23
numWFIB.extend([22434.0]) #2023D24
numWFIB.extend([24034.0]) #2023D28
numWFIB.extend([27434.0]) #2023D35
numWFIB.extend([23234.0]) #2023D36
numWFIB.extend([23634.0]) #2023D37
for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
