import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration                                                                                                                                              
from DQMOffline.RecoB.bTagCombinedSVVariables_cff import *
from DQMOffline.RecoB.bTagTrackIPAnalysis_cff import *
from DQMOffline.RecoB.bTagCombinedSVAnalysis_cff import *
from DQMOffline.RecoB.bTagTrackCountingAnalysis_cff import *
from DQMOffline.RecoB.bTagTrackProbabilityAnalysis_cff import *
from DQMOffline.RecoB.bTagTrackBProbabilityAnalysis_cff import *
from DQMOffline.RecoB.bTagGenericAnalysis_cff import *
from DQMOffline.RecoB.bTagSimpleSVAnalysis_cff import *
from DQMOffline.RecoB.bTagSoftLeptonAnalysis_cff import *
from DQMOffline.RecoB.bTagSoftLeptonByPtAnalysis_cff import *
from DQMOffline.RecoB.bTagSoftLeptonByIPAnalysis_cff import *

bTagCommonBlock = cms.PSet(
    # use pre-computed jet flavour identification
    # new default is to use genparticles - and it is the only option
    # Parameters which are common to all tagger algorithms
    # rec. jet
    ptRecJetMin = cms.double(30.0),
    ptRecJetMax = cms.double(40000.0),
    # eta
    etaMin = cms.double(0.0),
    etaMax = cms.double(2.4),
    # lepton momentum to jet energy ratio, if you use caloJets put ratioMin to -1.0 and ratioMax to 0.8
    ratioMin = cms.double(-9999.0),
    ratioMax = cms.double(9999.0),
    softLeptonInfo = cms.InputTag("softPFElectronsTagInfos"),
    # Section for the jet flavour identification
    jetMCSrc = cms.InputTag("mcJetFlavour"),
    caloJetMCSrc = cms.InputTag(""), #To define only if you use the old flavour tool
    useOldFlavourTool = cms.bool(False), #Recommended only for CaloJets, if True then define caloJetMCSrc and ignore jetMCSrc
    # eta and pt ranges
    ptRanges = cms.vdouble(50.0, 80.0, 120.0),
    etaRanges = cms.vdouble(0.0, 1.4, 2.4),
    #Jet ID and EnergyCorr.
    doJetID = cms.bool(False),
    doJEC = cms.bool(False),
    JECsource = cms.string("ak5PFCHSL1FastL2L3"),
    #tagger configuration
    tagConfig = cms.VPSet(
        cms.PSet(
            bTagTrackIPAnalysisBlock,
            type = cms.string('CandIP'),
            label = cms.InputTag("pfImpactParameterTagInfos"),
            folder = cms.string("IPTag")
        ),
        cms.PSet(
            bTagCombinedSVAnalysisBlock,
            ipTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
            type = cms.string('GenericMVA'),
            svTagInfos = cms.InputTag("pfSecondaryVertexTagInfos"),
            label = cms.InputTag("candidateCombinedSecondaryVertexComputer"),
            folder = cms.string("CSVTag")

        ),
        cms.PSet(
            bTagTrackCountingAnalysisBlock,
            label = cms.InputTag("pfTrackCountingHighEffBJetTags"),
            folder = cms.string("TCHE")
        ),
        cms.PSet(
            bTagTrackCountingAnalysisBlock,
            label = cms.InputTag("pfTrackCountingHighPurBJetTags"),
            folder = cms.string("TCHP")
        ),
        cms.PSet(
            bTagProbabilityAnalysisBlock,
            label = cms.InputTag("pfJetProbabilityBJetTags"),
            folder = cms.string("JP")
        ),
        cms.PSet(
            bTagBProbabilityAnalysisBlock,
            label = cms.InputTag("pfJetBProbabilityBJetTags"),
            folder = cms.string("JBP")
        ),
        cms.PSet(
            bTagSimpleSVAnalysisBlock,
            label = cms.InputTag("pfSimpleSecondaryVertexHighEffBJetTags"),
            folder = cms.string("SSVHE")
        ),
        cms.PSet(
            bTagSimpleSVAnalysisBlock,
            label = cms.InputTag("pfSimpleSecondaryVertexHighPurBJetTags"),
            folder = cms.string("SSVHP")
        ),
        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("combinedSecondaryVertexBJetTags"),
            folder = cms.string("CSV_tkOnly")
        ),
        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("pfCombinedSecondaryVertexBJetTags"),
            folder = cms.string("CSV")
        ),
        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
            folder = cms.string("CSVv2")
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
)


