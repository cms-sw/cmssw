import FWCore.ParameterSet.Config as cms

#not useful anymore for b-tagging but used in some other sequences
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL2L3,ak4PFL2Relative,ak4PFL3Absolute

#JEC for CHS
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFCHSL1Fastjet, ak4PFCHSL2Relative, ak4PFCHSL3Absolute, ak4PFCHSResidual, ak4PFCHSL1FastL2L3, ak4PFCHSL1FastL2L3Residual

######### DATA ############
from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagAnalysis.ptRanges = cms.vdouble(0.0)
bTagAnalysis.doJetID = True
bTagAnalysis.doJEC = True
#Residual correction will be added inside the c++ code only for data (checking the presence of genParticles collection), not explicit here as this sequence also ran on MC FullSim
bTagAnalysis.JECsource = cms.string("ak4PFCHSL1FastL2L3") 
bTagPlotsDATA = cms.Sequence(bTagAnalysis)

########## MC ############
#Matching
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK4byRef.jets = cms.InputTag("ak4PFJetsCHS")
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
bTagValidation.jetMCSrc = 'AK4byValAlgo'
bTagValidation.ptRanges = cms.vdouble(0.0)
bTagValidation.etaRanges = cms.vdouble(0.0)
bTagValidation.doJetID = True
bTagValidation.doJEC = True
bTagValidation.JECsource = cms.string("ak4PFCHSL1FastL2L3")
bTagValidation.genJetsMatched = cms.InputTag("newpatJetGenJetMatch")
#to run on fastsim
prebTagSequenceMC = cms.Sequence(ak4GenJetsForPUid*newpatJetGenJetMatch*myPartons*AK4Flavour)
#waiting for fastsim to support cand-based taggers
tagConfigFS = cms.VPSet(
    cms.PSet(
        bTagTrackIPAnalysisBlock,
        type = cms.string('TrackIP'),
        label = cms.InputTag("impactParameterTagInfos"),
        folder = cms.string("IPTag")
        ),
    cms.PSet(
        bTagCombinedSVAnalysisBlock,
        ipTagInfos = cms.InputTag("impactParameterTagInfos"),
        type = cms.string('GenericMVA'),
        svTagInfos = cms.InputTag("secondaryVertexTagInfos"),
        label = cms.InputTag("combinedSecondaryVertexComputer"),
        folder = cms.string("CSVTag")
        ),
    cms.PSet(
        bTagTrackCountingAnalysisBlock,
        label = cms.InputTag("trackCountingHighEffBJetTags"),
        folder = cms.string("TCHE")
        ),
    cms.PSet(
        bTagTrackCountingAnalysisBlock,
        label = cms.InputTag("trackCountingHighPurBJetTags"),
        folder = cms.string("TCHP")
        ),
    cms.PSet(
        bTagProbabilityAnalysisBlock,
        label = cms.InputTag("jetProbabilityBJetTags"),
        folder = cms.string("JP")
        ),
    cms.PSet(
        bTagBProbabilityAnalysisBlock,
        label = cms.InputTag("jetBProbabilityBJetTags"),
        folder = cms.string("JBP")
        ),
    cms.PSet(
        bTagSimpleSVAnalysisBlock,
        label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
        folder = cms.string("SSVHE")
        ),
    cms.PSet(
        bTagSimpleSVAnalysisBlock,
        label = cms.InputTag("simpleSecondaryVertexHighPurBJetTags"),
        folder = cms.string("SSVHP")
        ),
    cms.PSet(
        bTagGenericAnalysisBlock,
        label = cms.InputTag("combinedSecondaryVertexBJetTags"),
        folder = cms.string("CSV")
        ),
    cms.PSet(
        bTagGenericAnalysisBlock,
        label = cms.InputTag("combinedInclusiveSecondaryVertexV2BJetTags"),
        folder = cms.string("CSVv2")
        ),
    cms.PSet(
        bTagGenericAnalysisBlock,
        label = cms.InputTag("ghostTrackBJetTags"),
        folder = cms.string("GhTrk")
        ),
    cms.PSet(
        bTagSoftLeptonAnalysisBlock,
        label = cms.InputTag("softPFMuonBJetTags"),
        folder = cms.string("SMT")
        ),
    cms.PSet(
        bTagSoftLeptonAnalysisBlock,
        label = cms.InputTag("softPFElectronBJetTags"),
        folder = cms.string("SET")
        ),
    ) 
bTagValidationFS = bTagValidation.clone(tagConfig=tagConfigFS)
bTagPlotsMC = cms.Sequence(bTagValidationFS)

#to run on fullsim in the validation sequence, all histograms produced in the dqmoffline sequence
bTagValidationNoall = bTagValidation.clone(flavPlots="bcl")
bTagPlotsMCbcl = cms.Sequence(bTagValidationNoall)
