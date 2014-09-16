# 13TeV workflow added my Ian M. Nugent (nugent@physik.rwth-aachen.de)
#
# import the definition of the steps and input files:
from Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

# LO Generators
workflows[501]=['',['MinBias_TuneZ2star_13TeV_pythia6','HARVGEN']]
workflows[502]=['',['QCD_Pt-30_TuneZ2star_13TeV_pythia6','HARVGEN']]
workflows[503]=['',['MinBias_13TeV_pythia8','HARVGEN']]
workflows[504]=['',['QCD_Pt-30_13TeV_pythia8','HARVGEN']]

workflows[505]=['',['DYToLL_M-50_13TeV_pythia8','HARVGEN']]
workflows[506]=['',['WToLNu_13TeV_pythia8','HARVGEN']]

workflows[507]=['',['SoftQCDDiffractive_13TeV_pythia8','HARVGEN']]
workflows[508]=['',['SoftQCDnonDiffractive_13TeV_pythia8','HARVGEN']]
workflows[509]=['',['SoftQCDelastic_13TeV_pythia8','HARVGEN']]
workflows[510]=['',['SoftQCDinelastic_13TeV_pythia8','HARVGEN']]

# Matrix Element Generations (LHE Generation)

# Matrix Element
workflows[511]=['',['QCD_Pt-30_8TeV_herwigpp','HARVGEN']]

# Hadronization (Hadronization of LHE)
workflows[512]=['',['ZJetsLL_13TeV_madgraph-pythia8','HARVGEN']]
workflows[513]=['',['WJetsLNu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[514]=['',['GGToH_13TeV_pythia8','HARVGEN']]

workflows[515]=['',['ZJetsLLtaupinu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[516]=['',['WJetsLNutaupinu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[517]=['',['GGToHtaupinu_13TeV_pythia8','HARVGEN']]

workflows[518]=['',['ZJetsLLtaurhonu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[519]=['',['WJetsLNutaurhonu_13TeV_madgraph-pythia8','HARVGEN']]
workflows[520]=['',['GGToHtaurhonu_13TeV_pythia8','HARVGEN']]

# External Decays
workflows[521]=['',['TT_13TeV_pythia8-evtgen','HARVGEN']]

workflows[522]=['',['DYToLL_M-50_13TeV_pythia8-tauola','HARVGEN']]
workflows[523]=['',['WToLNu_13TeV_pythia8-tauola','HARVGEN']]
workflows[524]=['',['GGToH_13TeV_pythia8-tauola','HARVGEN']]

workflows[525]=['',['WToLNutaupinu_13TeV_pythia8-tauola','HARVGEN']]
workflows[526]=['',['DYToLLtaupinu_M-50_13TeV_pythia8-tauola','HARVGEN']]
workflows[527]=['',['GGToHtaupinu_13TeV_pythia8-tauola','HARVGEN']]

workflows[528]=['',['WToLNutaurhonu_13TeV_pythia8-tauola','HARVGEN']]
workflows[529]=['',['DYToLLtaurhonu_M-50_13TeV_pythia8-tauola','HARVGEN']]
workflows[530]=['',['GGToHtaurhonu_13TeV_pythia8-tauola','HARVGEN']]

# Heavy Ion
workflows[531]=['',['ReggeGribovPartonMC_EposLHC_5TeV_pPb','HARVGEN']]
#workflows[532]=['',['Hijing_PPb_MinimumBias','HARVGEN']]

# Miscellaneous
