
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[50000] = ['',['RunMinBias2012C','HLTD','RECODreHLT']]

workflows[50001] = ['SingleMuPt10_UP15ID', ['SingleMuPt10_UP15IDINPUT','SingleMuPt10_UP15_ID','DIGIUP15_ID','RECOUP15_ID','HARVESTUP15']]
workflows[50002] = ['TTbar_13_ID', ['TTbar_13IDINPUT','TTbar_13_ID','DIGIUP15_ID','RECOUP15_ID','HARVESTUP15']]

workflows[50101] = ['SingleMuPt10_UP15ID', ['SingleMuPt10_UP15FSIDINPUT','SingleMuPt10FS_UP15_ID','HARVESTUP15FS']]
workflows[50102] = ['TTbar_13_ID', ['TTbar_13FSIDINPUT','TTbarFS_13_ID','HARVESTUP15FS']]
