
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[201]=['',['ZmumuJets_Pt_20_300','DIGI','RECO'],{'--pileup':'E7TeV_AVE_2_BX2808'}]
workflows[202]=['TTbar',['TTbar2','DIGI','RECO'],[stCond,{'--pileup':'E7TeV_AVE_2_BX156'}]]

