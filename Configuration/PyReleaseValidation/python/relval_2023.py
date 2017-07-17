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
numWFIB = [20034.0,20034.1,20034.2] #2023D17 w/ special tracking and timing workflows
numWFIB.extend([20434.0]) #2023D19 (already has timing)
numWFIB.extend([20834.0,20834.2]) #2023D20
for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
