
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[201]=['',['ZmumuJets_Pt_20_300PU1','DIGIPU1','RECOPU1']]
workflows[202]=['TTbar',['TTbarPU2','DIGIPU1','RECOPU1']]

