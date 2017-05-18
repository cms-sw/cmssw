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
numWFIB = [20034.0,20034.2] #2023D10
numWFIB.extend([20434.0,20434.2]) #2023D11
numWFIB.extend([20434.1]) #2023D11 special tracking workflow
numWFIB.extend([21234.0,21234.2]) #2023D14
numWFIB.extend([23234.0,23234.2]) #2023D16
numWFIB.extend([23634.0,23634.2]) #2023D17
for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
