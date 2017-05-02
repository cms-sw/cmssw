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
numWFIB = [20034.0,20034.2] #2023D7
numWFIB.extend([20434.0,20434.2]) #2023D10
numWFIB.extend([21234.0,21234.2]) #2023D4
numWFIB.extend([21234.1]) #2023D4 special tracking workflow
numWFIB.extend([23234.0]) #2023D8 - already has timing
numWFIB.extend([23634.0,23634.2]) #2023D9
numWFIB.extend([24034.0,24034.2]) #2023D11
numWFIB.extend([24434.0,24434.2]) #2023D12
numWFIB.extend([24834.0,24834.2]) #2023D13
numWFIB.extend([26234.0,26234.2]) #2023D14
numWFIB.extend([26634.0,26634.2]) #2023D15
numWFIB.extend([27034.0,27034.2]) #2023D16
numWFIB.extend([27434.0,27434.2]) #2023D17
for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
