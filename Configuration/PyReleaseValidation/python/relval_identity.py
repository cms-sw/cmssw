
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *
from .MatrixUtil import Matrix

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[50000] = ['',['RunMinBias2012C','HLTD','RECODreHLT']]
