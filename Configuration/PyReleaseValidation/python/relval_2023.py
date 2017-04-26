# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them

#2023 WFs to run in IB (TenMuE_0_200, TTbar, TTbar+Timing, ZEE, MinBias)
numWFIB = [20021.0,20034.0,20034.2,20046.0,20053.0] #2023D7 scenario
numWFIB.extend([20421.0,20434.0,20434.2,20446.0,20453.0]) #2023D10
numWFIB.extend([21221.0,21234.0,21234.2,21246.0,21253.0]) #2023D4
numWFIB.extend([21234.1]) #2023D4 special tracking workflow
numWFIB.extend([23221.0,23234.0,23234.2,23246.0,23253.0]) #2023D8
numWFIB.extend([23621.0,23634.0,23634.2,23646.0,23653.0]) #2023D9
numWFIB.extend([24034.0,24034.2])#2023D11 TTbar only
numWFIB.extend([24434.0,24434.2])#2023D12 TTbar only
numWFIB.extend([24834.0,24834.2])#2023D13 TTbar only
numWFIB.extend([26234.0,26234.2])#2023D14 TTbar only
for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
