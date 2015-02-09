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
workflows[507]=['',['SoftQCDDiffractive_13TeV_pythia8','HARVESTGEN']]
workflows[508]=['',['SoftQCDnonDiffractive_13TeV_pythia8','HARVESTGEN']]
workflows[509]=['',['SoftQCDelastic_13TeV_pythia8','HARVESTGEN']]
workflows[510]=['',['SoftQCDinelastic_13TeV_pythia8','HARVESTGEN']]

# Matrix Element Generations (LHE Generation)

# Hadronization (LHE Generation + Hadronization)
workflows[514]=['',['GGToH_13TeV_pythia8','HARVESTGEN']]
workflows[515]=['DYTollJets_LO_Mad_13TeV_py8_taupinu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu','HARVESTGEN2']]
workflows[516]=['',['WJetsLNutaupinu_13TeV_madgraph-pythia8','HARVESTGEN']]

workflows[517]=['',['GGToHtaupinu_13TeV_pythia8','HARVESTGEN']]
workflows[518]=['DYTollJets_LO_Mad_13TeV_py8_taurhonu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu','HARVESTGEN2']]
workflows[519]=['',['WJetsLNutaurhonu_13TeV_madgraph-pythia8','HARVESTGEN']]
workflows[520]=['',['GGToHtaurhonu_13TeV_pythia8','HARVESTGEN']]

# External Decays
workflows[524]=['',['GGToH_13TeV_pythia8-tauola','HARVESTGEN']]
workflows[525]=['',['WToLNutaupinu_13TeV_pythia8-tauola','HARVESTGEN']]
workflows[526]=['DYTollJets_LO_Mad_13TeV_py8_Ta_taupinu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu','HARVESTGEN2']]
workflows[527]=['',['GGToHtaupinu_13TeV_pythia8-tauola','HARVESTGEN']]
workflows[528]=['',['WToLNutaurhonu_13TeV_pythia8-tauola','HARVESTGEN']]
workflows[529]=['DYTollJets_LO_Mad_13TeV_py8_Ta_taurhonu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu','HARVESTGEN2']]
workflows[530]=['',['GGToHtaurhonu_13TeV_pythia8-tauola','HARVESTGEN']]

# Heavy Ion
#workflows[532]=['',['Hijing_PPb_MinimumBias','HARVESTGEN']]

# Miscellaneous
