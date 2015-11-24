

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.JPTJetAnalyzer_cff import *

#Parameters for JPT RECO jets in PbPb

from RecoJets.JetPlusTracks.JetPlusTrackCorrectionsAA_cff import *
tracks = cms.InputTag("hiGeneralTracks")

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackExtrapolator.trackSrc = cms.InputTag("hiGeneralTracks")

from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*
JPTAntiKtPu5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiGeneralTracks")

from RecoJets.JetPlusTracks.JetPlusTrackCorrectionsAA_cff import *
JetPlusTrackZSPCorJetAntiKtPu5.tracks = cms.InputTag("hiGeneralTracks")
JetPlusTrackZSPCorJetAntiKtPu5.UseElectrons = cms.bool(False)
JetPlusTrackZSPCorJetAntiKtPu5.EfficiencyMap = cms.string("HeavyIonsAnalysis/Configuration/data/CMSSW_538HI_TrackNonEff.txt")
JetPlusTrackZSPCorJetAntiKtPu5.ResponseMap = cms.string("HeavyIonsAnalysis/Configuration/data/CMSSW_538HI_response.txt")
JetPlusTrackZSPCorJetAntiKtPu5.LeakageMap = cms.string("HeavyIonsAnalysis/Configuration/data/CMSSW_538HI_TrackLeakage.txt")


from RecoJets.JetAssociationProducers.ak5JTA_cff import*
JPTAntiKtPu5JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone()
JPTAntiKtPu5JetTracksAssociatorAtVertex.jets = cms.InputTag("akPu5CaloJets")
JPTAntiKtPu5JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiGeneralTracks")

JPTAntiKtPu5JetTracksAssociatorAtCaloFace = ak5JetTracksAssociatorAtCaloFace.clone()
JPTAntiKtPu5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("akPu5CaloJets")
JPTAntiKtPu5JetTracksAssociatorAtCaloFace.tracks = cms.InputTag("hiGeneralTracks")

JPTAntiKtPu5JetExtender.jets = cms.InputTag("akPu5CaloJets")
JPTAntiKtPu5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTAntiKtPu5JetTracksAssociatorAtCaloFace")
JPTAntiKtPu5JetExtender.jet2TracksAtVX = cms.InputTag("JPTAntiKtPu5JetTracksAssociatorAtVertex")

from RecoJets.JetPlusTracks.JetPlusTrackCorrectionsAA_cff import *
#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetAntiKtPu5  = cms.Sequence(JetPlusTrackCorrectionsAntiKtPu5)

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *
ak5JPTJetsL1   = cms.EDProducer('JPTJetCorrectionProducer',
    src         = cms.InputTag('JetPlusTrackZSPCorJetAntiKtPu5'),
    correctors  = cms.vstring('ak5L1JPTOffset'),
    vertexCollection = cms.string('hiSelectedVertex')
    )

ak5L1JPTOffset.offsetService = cms.string('')

recoJPTJetsHIC=cms.Sequence(jetPlusTrackZSPCorJetAntiKtPu5*ak5JPTJetsL1)

akPu5JPTmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak5JPTJetsL1"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

akPu5JPTparton = patJetPartonMatch.clone(
                                                        src = cms.InputTag("ak5JPTJetsL1"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu5JPTcorr = patJetCorrFactors.clone(
    useNPV = True,
    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak5JPTJetsL1"),
    payload = "AK5JPT",
    )



akPu5JPTpatJets = patJets.clone(
                                               jetSource = cms.InputTag("ak5JPTJetsL1"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu5JPTcorr")),
                                               genJetMatch = cms.InputTag("akPu5JPTmatch"),
                                               genPartonMatch = cms.InputTag("akPu5JPTparton"),
                                               jetIDMap = cms.InputTag("ak5CaloJetID"),
					       addBTagInfo         = False,
                                               addTagInfos         = False,
                                               addDiscriminators   = False,
                                               addAssociatedTracks = False,
                                               addJetCharge        = False,
                                               addJetID            = False,
                                               getJetMCFlavour     = False,
                                               addGenPartonMatch   = False, 
                                               addGenJetMatch      = False,
                                               embedGenJetMatch    = False,
                                               embedGenPartonMatch = False,
                                               embedCaloTowers     = False,
                                               embedPFCandidates = False
				            )

akPu5JPTJetAnalyzer = JPTJetAnalyzer.clone(jetTag = cms.InputTag("akPu5JPTpatJets"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(True),
                                                             matchTag = 'akPu5CalopatJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles")
                                                             )

akPu5JPTJetSequence_mc = cms.Sequence(
						  akPu5JPTmatch
                                                  *
                                                  akPu5JPTparton
                                                  *
                                                  akPu5JPTcorr
                                                  *
                                                  akPu5JPTpatJets
                                                  *
                                                  akPu5JPTJetAnalyzer
                                                  )

akPu5JPTJetSequence_data = cms.Sequence(
						    akPu5JPTcorr
                                                    *
                                                    akPu5JPTpatJets
                                                    *
                                                    akPu5JPTJetAnalyzer
                                                    )

akPu5JPTJetSequence = cms.Sequence(akPu5JPTJetSequence_data)
