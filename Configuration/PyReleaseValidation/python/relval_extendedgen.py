# 13TeV workflow added my Ian M. Nugent (nugent@physik.rwth-aachen.de)
#
# import the definition of the steps and input files:
from Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

# 'generator' the base set of relval for generators
# 'extendedgen' extends the base set to a more thorough assesment of GEN
# the two sets are exclusive

# LO Generators
workflows[505]=['',['DYToLL_M-50_13TeV_pythia8','HARVGEN']]
workflows[506]=['',['WToLNu_13TeV_pythia8','HARVGEN']]
workflows[507]=['',['SoftQCDDiffractive_13TeV_pythia8','HARVGEN']]
workflows[508]=['',['SoftQCDnonDiffractive_13TeV_pythia8','HARVGEN']]
workflows[509]=['',['SoftQCDelastic_13TeV_pythia8','HARVGEN']]
workflows[510]=['',['SoftQCDinelastic_13TeV_pythia8','HARVGEN']]

# Matrix Element Generations (LHE Generation)

# Hadronization (Hadronization of LHE)
workflows[514]=['',['GGToH_13TeV_pythia8','HARVGEN']]
workflows[515]=['',['ZJetsLLtaupinu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[516]=['',['WJetsLNutaupinu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[517]=['',['GGToHtaupinu_13TeV_pythia8','HARVGEN']]
workflows[518]=['',['ZJetsLLtaurhonu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[519]=['',['WJetsLNutaurhonu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[520]=['',['GGToHtaurhonu_13TeV_pythia8','HARVGEN']]

# External Decays
workflows[524]=['',['GGToH_13TeV_pythia8-tauola','HARVGEN']]
workflows[525]=['',['WToLNutaupinu_13TeV_pythia8-tauola','HARVGEN']]
workflows[526]=['',['DYToLLtaupinu_M-50_13TeV_pythia8-tauola','HARVGEN']]
workflows[527]=['',['GGToHtaupinu_13TeV_pythia8-tauola','HARVGEN']]
workflows[528]=['',['WToLNutaurhonu_13TeV_pythia8-tauola','HARVGEN']]
workflows[529]=['',['DYToLLtaurhonu_M-50_13TeV_pythia8-tauola','HARVGEN']]
workflows[530]=['',['GGToHtaurhonu_13TeV_pythia8-tauola','HARVGEN']]

# Heavy Ion
#workflows[532]=['',['Hijing_PPb_MinimumBias','HARVGEN']]

# Miscellaneous
