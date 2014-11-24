

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak2PFJets"),
    matched = cms.InputTag("ak2HiGenJetsCleaned")
    )

ak2PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak2PFJets")
                                                        )

ak2PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak2PFJets"),
    payload = "AK2PF_hiIterativeTracks"
    )

ak2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak2CaloJets'))

ak2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJetsCleaned'))

ak2PFbTagger = bTaggers("ak2PF")

#create objects locally since they dont load properly otherwise
ak2PFmatch = ak2PFbTagger.match
ak2PFparton = ak2PFbTagger.parton
ak2PFPatJetFlavourAssociation = ak2PFbTagger.PatJetFlavourAssociation
ak2PFJetTracksAssociatorAtVertex = ak2PFbTagger.JetTracksAssociatorAtVertex
ak2PFSimpleSecondaryVertexHighEffBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak2PFSimpleSecondaryVertexHighPurBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak2PFCombinedSecondaryVertexBJetTags = ak2PFbTagger.CombinedSecondaryVertexBJetTags
ak2PFCombinedSecondaryVertexMVABJetTags = ak2PFbTagger.CombinedSecondaryVertexMVABJetTags
ak2PFJetBProbabilityBJetTags = ak2PFbTagger.JetBProbabilityBJetTags
ak2PFSoftMuonByPtBJetTags = ak2PFbTagger.SoftMuonByPtBJetTags
ak2PFSoftMuonByIP3dBJetTags = ak2PFbTagger.SoftMuonByIP3dBJetTags
ak2PFTrackCountingHighEffBJetTags = ak2PFbTagger.TrackCountingHighEffBJetTags
ak2PFTrackCountingHighPurBJetTags = ak2PFbTagger.TrackCountingHighPurBJetTags
ak2PFPatJetPartonAssociation = ak2PFbTagger.PatJetPartonAssociation

ak2PFImpactParameterTagInfos = ak2PFbTagger.ImpactParameterTagInfos
ak2PFJetProbabilityBJetTags = ak2PFbTagger.JetProbabilityBJetTags
ak2PFPositiveOnlyJetProbabilityJetTags = ak2PFbTagger.PositiveOnlyJetProbabilityJetTags
ak2PFNegativeOnlyJetProbabilityJetTags = ak2PFbTagger.NegativeOnlyJetProbabilityJetTags
ak2PFNegativeTrackCountingHighEffJetTags = ak2PFbTagger.NegativeTrackCountingHighEffJetTags
ak2PFNegativeTrackCountingHighPur = ak2PFbTagger.NegativeTrackCountingHighPur
ak2PFNegativeOnlyJetBProbabilityJetTags = ak2PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak2PFPositiveOnlyJetBProbabilityJetTags = ak2PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak2PFSecondaryVertexTagInfos = ak2PFbTagger.SecondaryVertexTagInfos
ak2PFSimpleSecondaryVertexHighEffBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak2PFSimpleSecondaryVertexHighPurBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak2PFCombinedSecondaryVertexBJetTags = ak2PFbTagger.CombinedSecondaryVertexBJetTags
ak2PFCombinedSecondaryVertexMVABJetTags = ak2PFbTagger.CombinedSecondaryVertexMVABJetTags

ak2PFSecondaryVertexNegativeTagInfos = ak2PFbTagger.SecondaryVertexNegativeTagInfos
ak2PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak2PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak2PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak2PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak2PFCombinedSecondaryVertexNegativeBJetTags = ak2PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak2PFCombinedSecondaryVertexPositiveBJetTags = ak2PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak2PFSoftMuonTagInfos = ak2PFbTagger.SoftMuonTagInfos
ak2PFSoftMuonBJetTags = ak2PFbTagger.SoftMuonBJetTags
ak2PFSoftMuonByIP3dBJetTags = ak2PFbTagger.SoftMuonByIP3dBJetTags
ak2PFSoftMuonByPtBJetTags = ak2PFbTagger.SoftMuonByPtBJetTags
ak2PFNegativeSoftMuonByPtBJetTags = ak2PFbTagger.NegativeSoftMuonByPtBJetTags
ak2PFPositiveSoftMuonByPtBJetTags = ak2PFbTagger.PositiveSoftMuonByPtBJetTags

ak2PFPatJetFlavourId = cms.Sequence(ak2PFPatJetPartonAssociation*ak2PFPatJetFlavourAssociation)

ak2PFJetBtaggingIP       = cms.Sequence(ak2PFImpactParameterTagInfos *
            (ak2PFTrackCountingHighEffBJetTags +
             ak2PFTrackCountingHighPurBJetTags +
             ak2PFJetProbabilityBJetTags +
             ak2PFJetBProbabilityBJetTags +
             ak2PFPositiveOnlyJetProbabilityJetTags +
             ak2PFNegativeOnlyJetProbabilityJetTags +
             ak2PFNegativeTrackCountingHighEffJetTags +
             ak2PFNegativeTrackCountingHighPur +
             ak2PFNegativeOnlyJetBProbabilityJetTags +
             ak2PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak2PFJetBtaggingSV = cms.Sequence(ak2PFImpactParameterTagInfos
            *
            ak2PFSecondaryVertexTagInfos
            * (ak2PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak2PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak2PFCombinedSecondaryVertexBJetTags
                +
                ak2PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak2PFJetBtaggingNegSV = cms.Sequence(ak2PFImpactParameterTagInfos
            *
            ak2PFSecondaryVertexNegativeTagInfos
            * (ak2PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak2PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak2PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak2PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak2PFJetBtaggingMu = cms.Sequence(ak2PFSoftMuonTagInfos * (ak2PFSoftMuonBJetTags
                +
                ak2PFSoftMuonByIP3dBJetTags
                +
                ak2PFSoftMuonByPtBJetTags
                +
                ak2PFNegativeSoftMuonByPtBJetTags
                +
                ak2PFPositiveSoftMuonByPtBJetTags
              )
            )

ak2PFJetBtagging = cms.Sequence(ak2PFJetBtaggingIP
            *ak2PFJetBtaggingSV
            *ak2PFJetBtaggingNegSV
            *ak2PFJetBtaggingMu
            )

ak2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak2PFJets"),
        genJetMatch          = cms.InputTag("ak2PFmatch"),
        genPartonMatch       = cms.InputTag("ak2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak2PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak2PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak2PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak2PFJetBProbabilityBJetTags"),
            cms.InputTag("ak2PFJetProbabilityBJetTags"),
            cms.InputTag("ak2PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak2PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak2PFJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        embedCaloTowers = False,
        embedPFCandidates = True
        )

ak2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak2PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJetsCleaned',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak2PFJetSequence_mc = cms.Sequence(
                                                  ak2PFclean
                                                  *
                                                  ak2PFmatch
                                                  *
                                                  ak2PFparton
                                                  *
                                                  ak2PFcorr
                                                  *
                                                  ak2PFJetID
                                                  *
                                                  ak2PFPatJetFlavourId
                                                  *
                                                  ak2PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak2PFJetBtagging
                                                  *
                                                  ak2PFpatJetsWithBtagging
                                                  *
                                                  ak2PFJetAnalyzer
                                                  )

ak2PFJetSequence_data = cms.Sequence(ak2PFcorr
                                                    *
                                                    ak2PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak2PFJetBtagging
                                                    *
                                                    ak2PFpatJetsWithBtagging
                                                    *
                                                    ak2PFJetAnalyzer
                                                    )

ak2PFJetSequence_jec = ak2PFJetSequence_mc
ak2PFJetSequence_mix = ak2PFJetSequence_mc

ak2PFJetSequence = cms.Sequence(ak2PFJetSequence_mc)
