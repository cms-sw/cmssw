

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu3PFJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

akPu3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu3PFJets")
                                                        )

akPu3PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu3PFJets"),
    payload = "AKPu3PF_hiIterativeTracks"
    )

akPu3PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu3CaloJets'))

akPu3PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJetsCleaned'))

akPu3PFbTagger = bTaggers("akPu3PF")

#create objects locally since they dont load properly otherwise
akPu3PFmatch = akPu3PFbTagger.match
akPu3PFparton = akPu3PFbTagger.parton
akPu3PFPatJetFlavourAssociation = akPu3PFbTagger.PatJetFlavourAssociation
akPu3PFJetTracksAssociatorAtVertex = akPu3PFbTagger.JetTracksAssociatorAtVertex
akPu3PFSimpleSecondaryVertexHighEffBJetTags = akPu3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu3PFSimpleSecondaryVertexHighPurBJetTags = akPu3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu3PFCombinedSecondaryVertexBJetTags = akPu3PFbTagger.CombinedSecondaryVertexBJetTags
akPu3PFCombinedSecondaryVertexMVABJetTags = akPu3PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu3PFJetBProbabilityBJetTags = akPu3PFbTagger.JetBProbabilityBJetTags
akPu3PFSoftMuonByPtBJetTags = akPu3PFbTagger.SoftMuonByPtBJetTags
akPu3PFSoftMuonByIP3dBJetTags = akPu3PFbTagger.SoftMuonByIP3dBJetTags
akPu3PFTrackCountingHighEffBJetTags = akPu3PFbTagger.TrackCountingHighEffBJetTags
akPu3PFTrackCountingHighPurBJetTags = akPu3PFbTagger.TrackCountingHighPurBJetTags
akPu3PFPatJetPartonAssociation = akPu3PFbTagger.PatJetPartonAssociation

akPu3PFImpactParameterTagInfos = akPu3PFbTagger.ImpactParameterTagInfos
akPu3PFJetProbabilityBJetTags = akPu3PFbTagger.JetProbabilityBJetTags
akPu3PFPositiveOnlyJetProbabilityJetTags = akPu3PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu3PFNegativeOnlyJetProbabilityJetTags = akPu3PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu3PFNegativeTrackCountingHighEffJetTags = akPu3PFbTagger.NegativeTrackCountingHighEffJetTags
akPu3PFNegativeTrackCountingHighPur = akPu3PFbTagger.NegativeTrackCountingHighPur
akPu3PFNegativeOnlyJetBProbabilityJetTags = akPu3PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu3PFPositiveOnlyJetBProbabilityJetTags = akPu3PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu3PFSecondaryVertexTagInfos = akPu3PFbTagger.SecondaryVertexTagInfos
akPu3PFSimpleSecondaryVertexHighEffBJetTags = akPu3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu3PFSimpleSecondaryVertexHighPurBJetTags = akPu3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu3PFCombinedSecondaryVertexBJetTags = akPu3PFbTagger.CombinedSecondaryVertexBJetTags
akPu3PFCombinedSecondaryVertexMVABJetTags = akPu3PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu3PFSecondaryVertexNegativeTagInfos = akPu3PFbTagger.SecondaryVertexNegativeTagInfos
akPu3PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu3PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu3PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu3PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu3PFCombinedSecondaryVertexNegativeBJetTags = akPu3PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu3PFCombinedSecondaryVertexPositiveBJetTags = akPu3PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu3PFSoftMuonTagInfos = akPu3PFbTagger.SoftMuonTagInfos
akPu3PFSoftMuonBJetTags = akPu3PFbTagger.SoftMuonBJetTags
akPu3PFSoftMuonByIP3dBJetTags = akPu3PFbTagger.SoftMuonByIP3dBJetTags
akPu3PFSoftMuonByPtBJetTags = akPu3PFbTagger.SoftMuonByPtBJetTags
akPu3PFNegativeSoftMuonByPtBJetTags = akPu3PFbTagger.NegativeSoftMuonByPtBJetTags
akPu3PFPositiveSoftMuonByPtBJetTags = akPu3PFbTagger.PositiveSoftMuonByPtBJetTags

akPu3PFPatJetFlavourId = cms.Sequence(akPu3PFPatJetPartonAssociation*akPu3PFPatJetFlavourAssociation)

akPu3PFJetBtaggingIP       = cms.Sequence(akPu3PFImpactParameterTagInfos *
            (akPu3PFTrackCountingHighEffBJetTags +
             akPu3PFTrackCountingHighPurBJetTags +
             akPu3PFJetProbabilityBJetTags +
             akPu3PFJetBProbabilityBJetTags +
             akPu3PFPositiveOnlyJetProbabilityJetTags +
             akPu3PFNegativeOnlyJetProbabilityJetTags +
             akPu3PFNegativeTrackCountingHighEffJetTags +
             akPu3PFNegativeTrackCountingHighPur +
             akPu3PFNegativeOnlyJetBProbabilityJetTags +
             akPu3PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu3PFJetBtaggingSV = cms.Sequence(akPu3PFImpactParameterTagInfos
            *
            akPu3PFSecondaryVertexTagInfos
            * (akPu3PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu3PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu3PFCombinedSecondaryVertexBJetTags
                +
                akPu3PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu3PFJetBtaggingNegSV = cms.Sequence(akPu3PFImpactParameterTagInfos
            *
            akPu3PFSecondaryVertexNegativeTagInfos
            * (akPu3PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu3PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu3PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu3PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu3PFJetBtaggingMu = cms.Sequence(akPu3PFSoftMuonTagInfos * (akPu3PFSoftMuonBJetTags
                +
                akPu3PFSoftMuonByIP3dBJetTags
                +
                akPu3PFSoftMuonByPtBJetTags
                +
                akPu3PFNegativeSoftMuonByPtBJetTags
                +
                akPu3PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu3PFJetBtagging = cms.Sequence(akPu3PFJetBtaggingIP
            *akPu3PFJetBtaggingSV
            *akPu3PFJetBtaggingNegSV
            *akPu3PFJetBtaggingMu
            )

akPu3PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu3PFJets"),
        genJetMatch          = cms.InputTag("akPu3PFmatch"),
        genPartonMatch       = cms.InputTag("akPu3PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu3PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu3PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu3PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu3PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu3PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu3PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu3PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu3PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu3PFJetProbabilityBJetTags"),
            cms.InputTag("akPu3PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu3PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu3PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu3PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu3PFJetID"),
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

akPu3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu3PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJetsCleaned',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu3PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu3PFJetSequence_mc = cms.Sequence(
                                                  akPu3PFclean
                                                  *
                                                  akPu3PFmatch
                                                  *
                                                  akPu3PFparton
                                                  *
                                                  akPu3PFcorr
                                                  *
                                                  akPu3PFJetID
                                                  *
                                                  akPu3PFPatJetFlavourId
                                                  *
                                                  akPu3PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu3PFJetBtagging
                                                  *
                                                  akPu3PFpatJetsWithBtagging
                                                  *
                                                  akPu3PFJetAnalyzer
                                                  )

akPu3PFJetSequence_data = cms.Sequence(akPu3PFcorr
                                                    *
                                                    akPu3PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu3PFJetBtagging
                                                    *
                                                    akPu3PFpatJetsWithBtagging
                                                    *
                                                    akPu3PFJetAnalyzer
                                                    )

akPu3PFJetSequence_jec = akPu3PFJetSequence_mc
akPu3PFJetSequence_mix = akPu3PFJetSequence_mc

akPu3PFJetSequence = cms.Sequence(akPu3PFJetSequence_mix)
