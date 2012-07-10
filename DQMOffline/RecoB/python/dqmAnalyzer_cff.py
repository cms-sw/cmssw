import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.bTagAnalysisData_cfi import *
calobTagAnalysis = bTagAnalysis.clone()
bTagPlots = cms.Sequence(calobTagAnalysis)
calobTagAnalysis.finalizePlots = False
calobTagAnalysis.finalizeOnly = False

#collection of good vertices
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
    )

#Jet collection
JetCut=cms.string("neutralHadronEnergyFraction < 0.99 && neutralEmEnergyFraction < 0.99 && nConstituents > 1 && chargedHadronEnergyFraction > 0.0 && chargedMultiplicity > 0.0 && chargedEmEnergyFraction < 0.99")

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFL2L3,ak5PFL2Relative,ak5PFL3Absolute
newAk5PFL2L3 = ak5PFL2L3.clone()

from JetMETCorrections.Configuration.DefaultJEC_cff import ak5PFJetsL2L3
ak5PFJetsJEC = ak5PFJetsL2L3.clone(
    correctors = ['newAk5PFL2L3']
    )

PFJetsFilter = cms.EDFilter("PFJetSelector",
                            src = cms.InputTag("ak5PFJetsJEC"),
                            cut = JetCut,
                            filter = cms.bool(True)
                            )

jetID = cms.InputTag("PFJetsFilter")

#JTA
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
pfAk5JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone(jets = jetID)

#btag sequence
from RecoBTag.Configuration.RecoBTag_cff import *

pfImpactParameterTagInfos = impactParameterTagInfos.clone(jetTracks = cms.InputTag("pfAk5JetTracksAssociatorAtVertex"))
pfSecondaryVertexTagInfos = secondaryVertexTagInfos.clone(trackIPTagInfos = "pfImpactParameterTagInfos")

pfTrackCountingHighEffBJetTags = trackCountingHighEffBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))
pfTrackCountingHighPurBJetTags = trackCountingHighPurBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))

pfJetProbabilityBJetTags = jetProbabilityBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))
pfJetBProbabilityBJetTags = jetBProbabilityBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))

pfSimpleSecondaryVertexHighEffBJetTags = simpleSecondaryVertexHighEffBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos")))
pfSimpleSecondaryVertexHighPurBJetTags = simpleSecondaryVertexHighPurBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos")))

pfGhostTrackVertexTagInfos = pfSecondaryVertexTagInfos.clone()
pfGhostTrackVertexTagInfos.vertexReco = ghostTrackVertexRecoBlock.vertexReco
pfGhostTrackVertexTagInfos.vertexCuts.multiplicityMin = 1
pfGhostTrackBJetTags = ghostTrackBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfGhostTrackVertexTagInfos"))
    )

pfCombinedSecondaryVertexBJetTags = combinedSecondaryVertexBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfSecondaryVertexTagInfos"))
    )
pfCombinedSecondaryVertexMVABJetTags = combinedSecondaryVertexMVABJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfSecondaryVertexTagInfos"))
    )

pfSoftMuonTagInfos = softMuonTagInfos.clone(jets = jetID)
pfSoftElectronTagInfos = softElectronTagInfos.clone(jets = jetID)
pfSoftMuonBJetTags = softMuonBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftMuonTagInfos")))
pfSoftMuonByPtBJetTags = softMuonByPtBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftMuonTagInfos")))
pfSoftMuonByIP3dBJetTags = softMuonByIP3dBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftMuonTagInfos")))
pfSoftElectronByPtBJetTags = softElectronByPtBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftElectronTagInfos")))
pfSoftElectronByIP3dBJetTags = softElectronByIP3dBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftElectronTagInfos")))

pfbtagging = cms.Sequence(
    (
    pfImpactParameterTagInfos *
    ( pfTrackCountingHighEffBJetTags +
      pfTrackCountingHighPurBJetTags +
      pfJetProbabilityBJetTags +
      pfJetBProbabilityBJetTags +
      
      pfSecondaryVertexTagInfos *
      ( pfSimpleSecondaryVertexHighEffBJetTags +
        pfSimpleSecondaryVertexHighPurBJetTags +
        pfCombinedSecondaryVertexBJetTags +
        pfCombinedSecondaryVertexMVABJetTags
        ) +
      pfGhostTrackVertexTagInfos *
      pfGhostTrackBJetTags
      ) +
    
    softElectronCands *
    pfSoftElectronTagInfos * (
    pfSoftElectronByIP3dBJetTags +
    pfSoftElectronByPtBJetTags
    ) +
    
    pfSoftMuonTagInfos *
    ( pfSoftMuonBJetTags +
      pfSoftMuonByIP3dBJetTags +
      pfSoftMuonByPtBJetTags
      )
    )
)

#preSeq
prebTagSequence = cms.Sequence(goodOfflinePrimaryVertices*ak5PFJetsJEC*PFJetsFilter*pfAk5JetTracksAssociatorAtVertex*pfbtagging)

# Module execution for data
#from DQMOffline.RecoB.bTagAnalysisData_cfi import *
pfbTagAnalysis = bTagAnalysis.clone(
    tagConfig = cms.VPSet(
       cms.PSet(
           bTagTrackIPAnalysisBlock,
           type = cms.string('TrackIP'),
           label = cms.InputTag("pfImpactParameterTagInfos"),
           folder = cms.string("IPTag")
           ),
       cms.PSet(
           bTagCombinedSVAnalysisBlock,
           ipTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
           type = cms.string('GenericMVA'),
           svTagInfos = cms.InputTag("pfSecondaryVertexTagInfos"),
           label = cms.InputTag("combinedSecondaryVertex"),
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
           label = cms.InputTag("pfCombinedSecondaryVertexBJetTags"),
           folder = cms.string("CSV")
           ),
       cms.PSet(
           bTagGenericAnalysisBlock,
           label = cms.InputTag("pfCombinedSecondaryVertexMVABJetTags"),
           folder = cms.string("CSVMVA")
           ),
       cms.PSet(
           bTagGenericAnalysisBlock,
           label = cms.InputTag("pfGhostTrackBJetTags"),
           folder = cms.string("GhTrk")
           ),
       cms.PSet(
           bTagSoftLeptonAnalysisBlock,
           label = cms.InputTag("pfSoftMuonBJetTags"),
           folder = cms.string("SMT")
           ),
       cms.PSet(
           bTagSoftLeptonByIPAnalysisBlock,
           label = cms.InputTag("pfSoftMuonByIP3dBJetTags"),
           folder = cms.string("SMTIP3d")
           ),
       cms.PSet(
           bTagSoftLeptonByPtAnalysisBlock,
           label = cms.InputTag("pfSoftMuonByPtBJetTags"),
           folder = cms.string("SMTPt")
           )
       ),
    )
pfbTagAnalysis.finalizePlots = False
pfbTagAnalysis.finalizeOnly = False

bTagPlotsDATA = cms.Sequence(pfbTagAnalysis)


########## MC ############
#Matching
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK5byRef.jets = jetID

# Module execution for MC
from Validation.RecoB.bTagAnalysis_cfi import *
pfbTagValidation = bTagValidation.clone(
    tagConfig = cms.VPSet(
       cms.PSet(
           bTagTrackIPAnalysisBlock,
           type = cms.string('TrackIP'),
           label = cms.InputTag("pfImpactParameterTagInfos"),
           folder = cms.string("IPTag")
           ),
       cms.PSet(
           bTagCombinedSVAnalysisBlock,
           ipTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
           type = cms.string('GenericMVA'),
           svTagInfos = cms.InputTag("pfSecondaryVertexTagInfos"),
           label = cms.InputTag("combinedSecondaryVertex"),
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
           label = cms.InputTag("pfCombinedSecondaryVertexBJetTags"),
           folder = cms.string("CSV")
           ),
       cms.PSet(
           bTagGenericAnalysisBlock,
           label = cms.InputTag("pfCombinedSecondaryVertexMVABJetTags"),
           folder = cms.string("CSVMVA")
           ),
       cms.PSet(
           bTagGenericAnalysisBlock,
           label = cms.InputTag("pfGhostTrackBJetTags"),
           folder = cms.string("GhTrk")
           ),
       cms.PSet(
           bTagSoftLeptonAnalysisBlock,
           label = cms.InputTag("pfSoftMuonBJetTags"),
           folder = cms.string("SMT")
           ),
       cms.PSet(
           bTagSoftLeptonByIPAnalysisBlock,
           label = cms.InputTag("pfSoftMuonByIP3dBJetTags"),
           folder = cms.string("SMTIP3d")
           ),
       cms.PSet(
           bTagSoftLeptonByPtAnalysisBlock,
           label = cms.InputTag("pfSoftMuonByPtBJetTags"),
           folder = cms.string("SMTPt")
           )
       ),
    )
pfbTagValidation.finalizePlots = False
pfbTagValidation.finalizeOnly = False
pfbTagValidation.jetMCSrc = 'AK5byValAlgo'
pfbTagValidation.ptRanges = cms.vdouble(0.0)
pfbTagValidation.etaRanges = cms.vdouble(0.0)

bTagPlotsMC = cms.Sequence(myPartons*AK5Flavour*pfbTagValidation)
