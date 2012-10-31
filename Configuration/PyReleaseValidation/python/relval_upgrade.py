
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used


workflows[3107] = ['', ['FourMuPt1_200_UPGphase1']]
workflows[3123] = ['', ['MinBias_UPGphase1_14']]
workflows[3135] = ['', ['TTbar_Tauola_UPGphase1_14']]
