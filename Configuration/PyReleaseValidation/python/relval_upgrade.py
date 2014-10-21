
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),  the name of step1 will be used

#Example ExtendedPhase1 part gun
#workflows[3100] = ['', ['FourMuPt1_200_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]

# place holder of the upgrade workflow to validate the upgrade software during the 73X migration and beyond

