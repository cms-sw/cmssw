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

# Matrix Element Generations (scerpa)
#workflows[533]=['',['sherpa_ZtoEE_0j_BlackHat_13TeV_MASTER','HARVESTGEN']]
workflows[534]=['',['sherpa_ZtoEE_0j_OpenLoops_13TeV_MASTER','HARVESTGEN']]

# Hadronization (LHE Generation + Hadronization)
workflows[555]=['DYTollJets_NLO_Mad_13TeV_py8',['DYToll012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max0p_LHE_pythia8','HARVESTGEN2']]     # ALWAYS RUN
workflows[513]=['WTolNuJets_LO_Mad_13TeV_py8',['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8','HARVESTGEN2']]     # ALWAYS RUN
workflows[551]=['TTbar012Jets_NLO_Mad_13TeV_py8',['TTbar012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8','HARVESTGEN2']]     # ALWAYS RUN
workflows[556]=['TTbar_NLO_Pow_13TeV_py8',['TTbar_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto2p_pythia8','HARVESTGEN2']]     # ALWAYS RUN
workflows[514]=['GGToHgg_NLO_Pow_13TeV_py8',['GGToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Hgg_powhegEmissionVeto_pythia8','HARVESTGEN2']]     # ALWAYS RUN
workflows[552]=['VHToHtt_NLO_Pow_13TeV_py8',['VHToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Htt_powhegEmissionVeto_pythia8','HARVESTGEN2']]     # ALWAYS RUN
workflows[554]=['VBFToH4l_NLO_Pow_JHU_13TeV_py8',['VBFToH_Pow_JHU4l_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_pythia8','HARVESTGEN2']]     # ALWAYS RUN


workflows[515]=['DYTollJets_LO_Mad_13TeV_py8_taupinu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu','HARVESTGEN2']]
workflows[516]=['WTolNuJets_LO_Mad_13TeV_py8_taupinu',['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu','HARVESTGEN2']]
workflows[517]=['VHToHtt_NLO_Pow_13TeV_py8_taupinu',['VHToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Httpinu_powhegEmissionVeto_pythia8','HARVESTGEN2']]
workflows[518]=['DYTollJets_LO_Mad_13TeV_py8_taurhonu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu','HARVESTGEN2']]
workflows[519]=['WTolNuJets_LO_Mad_13TeV_py8_taurhonu',['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu','HARVESTGEN2']]
workflows[520]=['VHToHtt_NLO_Pow_13TeV_py8_taurhonu',['VHToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Httrhonu_powhegEmissionVeto_pythia8','HARVESTGEN2']]

# External Decays

workflows[521]=['WTolNuJets_LO_Mad_13TeV_py8_Ta',['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola','HARVESTGEN2']]     # ALWAYS RUN
workflows[522]=['DYTollJets_LO_Mad_13TeV_py8_Ta',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola','HARVESTGEN2']]     # ALWAYS RUN
workflows[523]=['TTbar012Jets_NLO_Mad_13TeV_py8_Evt',['TTbar012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_evtgen','HARVESTGEN2']]     # ALWAYS RUN
workflows[524]=['VHToHtt_NLO_Pow_13TeV_py8_Ta',['VHToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Htt_powhegEmissionVeto_pythia8_tauola','HARVESTGEN2']]     # ALWAYS RUN

workflows[527]=['VHToHtt_NLO_Pow_13TeV_py8_Ta_taupinu',['VHToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Httpinu_powhegEmissionVeto_pythia8_tauola','HARVESTGEN2']]
workflows[529]=['DYTollJets_LO_Mad_13TeV_py8_Ta_taurhonu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu','HARVESTGEN2']]
workflows[530]=['VHToHtt_NLO_Pow_13TeV_py8_Ta_taurhonu',['VHToH_Pow_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_Httrhonu_powhegEmissionVeto_pythia8_tauola','HARVESTGEN2']]
workflows[526]=['DYTollJets_LO_Mad_13TeV_py8_Ta_taupinu',['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu','HARVESTGEN2']]
workflows[525]=['WTolNuJets_LO_Mad_13TeV_py8_Ta_taupinu',['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu','HARVESTGEN2']]
workflows[528]=['WTolNuJets_LO_Mad_13TeV_py8_Ta_taurhonu',['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV','Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu','HARVESTGEN2']]

# Heavy Ion
#workflows[532]=['',['Hijing_PPb_MinimumBias','HARVESTGEN']]

# Miscellaneous
workflows[560]=['',['ZprimeToll_M3000_13TeV_pythia8','HARVESTGEN']]
workflows[561]=['',['WprimeTolNu_M3000_13TeV_pythia8','HARVESTGEN']]
workflows[562]=['BulkG_ZZ_2L2Q_M1200_narrow_13TeV_pythia8',['BulkG_M1200_narrow_2L2Q_LHE_13TeV','Hadronizer_TuneCUETP8M1_Mad_pythia8','HARVESTGEN2']]


