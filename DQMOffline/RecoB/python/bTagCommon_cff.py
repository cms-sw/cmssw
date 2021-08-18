import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration                                                                                                                                              
from DQMOffline.RecoB.bTagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.cTagGenericAnalysis_cff import cTagGenericAnalysisBlock
from DQMOffline.RecoB.bTagTrackIPAnalysis_cff import bTagTrackIPAnalysisBlock
from DQMOffline.RecoB.bTagTrackProbabilityAnalysis_cff import bTagProbabilityAnalysisBlock



bTagCommonBlock = cms.PSet(
    # use pre-computed jet flavour identification
    # new default is to use genparticles - and it is the only option
    # Parameters which are common to all tagger algorithms
    # rec. jet
    ptRecJetMin = cms.double(30.0),
    ptRecJetMax = cms.double(40000.0),
    # eta
    etaMin = cms.double(0.0),
    etaMax = cms.double(2.5),
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
    etaRanges = cms.vdouble(0.0, 1.4, 2.5),
    #Jet ID and EnergyCorr.
    doJetID = cms.bool(False),
    doJEC = cms.bool(False),
    JECsourceMC = cms.InputTag("ak4PFCHSL1FastL2L3Corrector"),
    JECsourceData = cms.InputTag("ak4PFCHSL1FastL2L3ResidualCorrector"),
    #tagger configuration
    tagConfig = cms.VPSet(

        cms.PSet(
            bTagTrackIPAnalysisBlock,
            type = cms.string('CandIP'),
            label = cms.InputTag("pfImpactParameterTagInfos"),
            folder = cms.string("IPTag")
        ),

        cms.PSet(
            bTagProbabilityAnalysisBlock,
            label = cms.InputTag("pfJetProbabilityBJetTags"),
            folder = cms.string("JP")
        ),

        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:BvsAll"),
            folder = cms.string("deepCSV_BvsAll"),
            differentialPlots = cms.bool(True),
            discrCut = cms.double(0.1522)
        ),
        cms.PSet(
            cTagGenericAnalysisBlock,
            label = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:CvsL"),
            folder = cms.string("deepCSV_CvsL"),
            doCTagPlots = cms.bool(True),
            differentialPlots = cms.bool(True),
            discrCut = cms.double(0.15)
        ),
        cms.PSet(
            cTagGenericAnalysisBlock,
            label = cms.InputTag("pfDeepCSVDiscriminatorsJetTags:CvsB"),
            folder = cms.string("deepCSV_CvsB"),
            doCTagPlots = cms.bool(True),
            differentialPlots = cms.bool(True),
            discrCut = cms.double(0.28)
        ),

    )
)
