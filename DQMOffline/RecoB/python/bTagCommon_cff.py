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
from DQMOffline.RecoB.cTagGenericAnalysis_cff import *
from DQMOffline.RecoB.cTagCombinedSVAnalysis_cff import *

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
    JECsourceMC = cms.InputTag("ak4PFCHSL1FastL2L3Corrector"),
    JECsourceData = cms.InputTag("ak4PFCHSL1FastL2L3ResidualCorrector"),
    #tagger configuration
    tagConfig = cms.VPSet(
	      cms.PSet(
           cTagCombinedSVAnalysisBlock,
           listTagInfos = cms.VInputTag(
           	cms.InputTag("pfImpactParameterTagInfos"),
           	cms.InputTag("pfInclusiveSecondaryVertexFinderCvsLTagInfos"),                
           	cms.InputTag("softPFMuonsTagInfos"),
           	cms.InputTag("softPFElectronsTagInfos")
           ),
           type = cms.string('GenericMVA'),
           label = cms.InputTag("candidateCombinedSecondaryVertexSoftLeptonComputer"),
           folder = cms.string("charmtaggerTag")
        ),
	      cms.PSet(
            cTagGenericAnalysisBlock,
            label = cms.InputTag("pfCombinedCvsLJetTags"),
            folder = cms.string("charmtagger_CvsL"),
            doCTagPlots = cms.bool(True)
        ),
        cms.PSet(
            cTagGenericAnalysisBlock,
            label = cms.InputTag("pfCombinedCvsBJetTags"),
            folder = cms.string("charmtagger_CvsB"),
            doCTagPlots = cms.bool(True)
        )
    )    
)
