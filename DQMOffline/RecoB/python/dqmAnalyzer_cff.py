import FWCore.ParameterSet.Config as cms

from RecoBTag.Combined.pfDeepCSVDiscriminatorsJetTags_cfi import pfDeepCSVDiscriminatorsJetTags

######### DATA ############
from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagAnalysis.ptRanges = cms.vdouble(0.0)
bTagAnalysis.doJetID = True
bTagAnalysis.doJEC = True
#Residual correction will be added inside the c++ code only for data (checking the presence of genParticles collection), not explicit here as this sequence also ran on MC FullSim
bTagPlotsDATA = cms.Sequence(pfDeepCSVDiscriminatorsJetTags * bTagAnalysis)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(bTagAnalysis,
                                      doJEC=False
)


########## MC ############
#Matching
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
myak4JetFlavourInfos = ak4JetFlavourInfos.clone(
    jets = cms.InputTag("ak4PFJetsCHS"),
    partons = cms.InputTag("selectedHadronsAndPartons","algorithmicPartons"),
    hadronFlavourHasPriority = cms.bool(True)
    )

#Get gen jet collection for real jets
ak4GenJetsForPUid = cms.EDFilter("GenJetSelector",
                                 src = cms.InputTag("ak4GenJets"),
                                 cut = cms.string('pt > 8.'),
                                 filter = cms.bool(False)
                                 )
#do reco gen - reco matching
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetGenJetMatch
newpatJetGenJetMatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4PFJetsCHS"),
    matched = cms.InputTag("ak4GenJetsForPUid"),
    maxDeltaR = cms.double(0.25),
    resolveAmbiguities = cms.bool(True)
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(newpatJetGenJetMatch, src = "akCs4PFJets")

# Module execution for MC
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.jetMCSrc = 'myak4JetFlavourInfos'
bTagValidation.ptRanges = cms.vdouble(0.0)
bTagValidation.etaRanges = cms.vdouble(0.0)
bTagValidation.doJetID = True
bTagValidation.doJEC = True
bTagValidation.genJetsMatched = cms.InputTag("newpatJetGenJetMatch")
#to run on fastsim
prebTagSequenceMC = cms.Sequence(ak4GenJetsForPUid*newpatJetGenJetMatch*selectedHadronsAndPartons*myak4JetFlavourInfos*pfDeepCSVDiscriminatorsJetTags)
bTagPlotsMC = cms.Sequence(bTagValidation)

## customizations for the pp_on_AA eras
(pp_on_XeXe_2017 | pp_on_AA).toModify(bTagValidation,
                                      doJEC=False
)

#to run on fullsim in the validation sequence, all histograms produced in the dqmoffline sequence
bTagValidationNoall = bTagValidation.clone(flavPlots="bcl")
bTagPlotsMCbcl = cms.Sequence(bTagValidationNoall)
