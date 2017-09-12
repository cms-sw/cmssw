import FWCore.ParameterSet.Config as cms

######### DATA ############
from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagAnalysis.ptRanges = cms.vdouble(0.0)
bTagAnalysis.doJetID = True
bTagAnalysis.doJEC = True
#Residual correction will be added inside the c++ code only for data (checking the presence of genParticles collection), not explicit here as this sequence also ran on MC FullSim
bTagPlotsDATA = cms.Sequence(bTagAnalysis)

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

# Module execution for MC
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.jetMCSrc = 'myak4JetFlavourInfos'
bTagValidation.ptRanges = cms.vdouble(0.0)
bTagValidation.etaRanges = cms.vdouble(0.0)
bTagValidation.doJetID = True
bTagValidation.doJEC = True
bTagValidation.genJetsMatched = cms.InputTag("newpatJetGenJetMatch")
#to run on fastsim
prebTagSequenceMC = cms.Sequence(ak4GenJetsForPUid*newpatJetGenJetMatch*selectedHadronsAndPartons*myak4JetFlavourInfos)
bTagPlotsMC = cms.Sequence(bTagValidation)

#to run on fullsim in the validation sequence, all histograms produced in the dqmoffline sequence
bTagValidationNoall = bTagValidation.clone(flavPlots="bcl")
bTagPlotsMCbcl = cms.Sequence(bTagValidationNoall)
