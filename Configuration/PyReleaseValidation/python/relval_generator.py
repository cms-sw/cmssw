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
#workflows[501]=['',['MinBias_TuneZ2star_13TeV_pythia6','HARVESTGEN']]
#workflows[502]=['',['QCD_Pt-30_TuneZ2star_13TeV_pythia6','HARVESTGEN']]
workflows[503]=['',['MinBias_13TeV_pythia8','HARVESTGEN']]
workflows[504]=['',['QCD_Pt-30_13TeV_pythia8','HARVESTGEN']]
workflows[505]=['',['DYToLL_M-50_13TeV_pythia8','HARVESTGEN']]
workflows[506]=['',['WToLNu_13TeV_pythia8','HARVESTGEN']]
workflows[511]=['',['QCD_Pt-30_13TeV_aMCatNLO_herwig7','HARVESTGEN']]
workflows[511.1]=['',['PPtoee-M60-120_13TeV_madgraph_openloops_herwig7','HARVESTGEN']]

# Matrix Element Generations & Hadronization (LHE Generation + Hadronization)
workflows[512]=['DYTollJets_LO_Mad_13TeV_py8',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8','HARVESTGEN2']]
# Matrix Element Generations & Hadronization (Sherpa)

# External Decays

# Heavy Ion
workflows[531]=['',['ReggeGribovPartonMC_EposLHC_5TeV_pPb','HARVESTGEN']]

# B-physics
workflows[541]=['',['BuToKstarJPsiToMuMu_forSTEAM_13TeV_TuneCUETP8M1','HARVESTGEN']]
#workflows[542]=['',['Upsilon4swithBuToKstarJPsiToMuMu_forSTEAM_13TeV_TuneCUETP8M1','HARVESTGEN']]
#workflows[543]=['',['Upsilon4sBaBarExample_BpBm_Dstarpipi_D0Kpi_nonres_forSTEAM_13TeV_TuneCUETP8M1','HARVESTGEN']]
#workflows[544]=['',['LambdaBToLambdaMuMuToPPiMuMu_forSTEAM_13TeV_TuneCUETP8M1','HARVESTGEN']]
workflows[545]=['',['BsToMuMu_forSTEAM_13TeV_TuneCUETP8M1','HARVESTGEN']]

# Miscellaneous
