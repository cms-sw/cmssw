

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs3PFJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

akVs3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3PFJets")
                                                        )

akVs3PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs3PFJets"),
    payload = "AKVs3PF_hiIterativeTracks"
    )

akVs3PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs3CaloJets'))

akVs3PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJetsCleaned'))

akVs3PFbTagger = bTaggers("akVs3PF")

#create objects locally since they dont load properly otherwise
akVs3PFmatch = akVs3PFbTagger.match
akVs3PFparton = akVs3PFbTagger.parton
akVs3PFPatJetFlavourAssociation = akVs3PFbTagger.PatJetFlavourAssociation
akVs3PFJetTracksAssociatorAtVertex = akVs3PFbTagger.JetTracksAssociatorAtVertex
akVs3PFSimpleSecondaryVertexHighEffBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3PFSimpleSecondaryVertexHighPurBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3PFCombinedSecondaryVertexBJetTags = akVs3PFbTagger.CombinedSecondaryVertexBJetTags
akVs3PFCombinedSecondaryVertexMVABJetTags = akVs3PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs3PFJetBProbabilityBJetTags = akVs3PFbTagger.JetBProbabilityBJetTags
akVs3PFSoftMuonByPtBJetTags = akVs3PFbTagger.SoftMuonByPtBJetTags
akVs3PFSoftMuonByIP3dBJetTags = akVs3PFbTagger.SoftMuonByIP3dBJetTags
akVs3PFTrackCountingHighEffBJetTags = akVs3PFbTagger.TrackCountingHighEffBJetTags
akVs3PFTrackCountingHighPurBJetTags = akVs3PFbTagger.TrackCountingHighPurBJetTags
akVs3PFPatJetPartonAssociation = akVs3PFbTagger.PatJetPartonAssociation

akVs3PFImpactParameterTagInfos = akVs3PFbTagger.ImpactParameterTagInfos
akVs3PFJetProbabilityBJetTags = akVs3PFbTagger.JetProbabilityBJetTags
akVs3PFPositiveOnlyJetProbabilityJetTags = akVs3PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs3PFNegativeOnlyJetProbabilityJetTags = akVs3PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs3PFNegativeTrackCountingHighEffJetTags = akVs3PFbTagger.NegativeTrackCountingHighEffJetTags
akVs3PFNegativeTrackCountingHighPur = akVs3PFbTagger.NegativeTrackCountingHighPur
akVs3PFNegativeOnlyJetBProbabilityJetTags = akVs3PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs3PFPositiveOnlyJetBProbabilityJetTags = akVs3PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs3PFSecondaryVertexTagInfos = akVs3PFbTagger.SecondaryVertexTagInfos
akVs3PFSimpleSecondaryVertexHighEffBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3PFSimpleSecondaryVertexHighPurBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3PFCombinedSecondaryVertexBJetTags = akVs3PFbTagger.CombinedSecondaryVertexBJetTags
akVs3PFCombinedSecondaryVertexMVABJetTags = akVs3PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs3PFSecondaryVertexNegativeTagInfos = akVs3PFbTagger.SecondaryVertexNegativeTagInfos
akVs3PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs3PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs3PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs3PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs3PFCombinedSecondaryVertexNegativeBJetTags = akVs3PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs3PFCombinedSecondaryVertexPositiveBJetTags = akVs3PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs3PFSoftMuonTagInfos = akVs3PFbTagger.SoftMuonTagInfos
akVs3PFSoftMuonBJetTags = akVs3PFbTagger.SoftMuonBJetTags
akVs3PFSoftMuonByIP3dBJetTags = akVs3PFbTagger.SoftMuonByIP3dBJetTags
akVs3PFSoftMuonByPtBJetTags = akVs3PFbTagger.SoftMuonByPtBJetTags
akVs3PFNegativeSoftMuonByPtBJetTags = akVs3PFbTagger.NegativeSoftMuonByPtBJetTags
akVs3PFPositiveSoftMuonByPtBJetTags = akVs3PFbTagger.PositiveSoftMuonByPtBJetTags

akVs3PFPatJetFlavourId = cms.Sequence(akVs3PFPatJetPartonAssociation*akVs3PFPatJetFlavourAssociation)

akVs3PFJetBtaggingIP       = cms.Sequence(akVs3PFImpactParameterTagInfos *
            (akVs3PFTrackCountingHighEffBJetTags +
             akVs3PFTrackCountingHighPurBJetTags +
             akVs3PFJetProbabilityBJetTags +
             akVs3PFJetBProbabilityBJetTags +
             akVs3PFPositiveOnlyJetProbabilityJetTags +
             akVs3PFNegativeOnlyJetProbabilityJetTags +
             akVs3PFNegativeTrackCountingHighEffJetTags +
             akVs3PFNegativeTrackCountingHighPur +
             akVs3PFNegativeOnlyJetBProbabilityJetTags +
             akVs3PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs3PFJetBtaggingSV = cms.Sequence(akVs3PFImpactParameterTagInfos
            *
            akVs3PFSecondaryVertexTagInfos
            * (akVs3PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs3PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs3PFCombinedSecondaryVertexBJetTags
                +
                akVs3PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs3PFJetBtaggingNegSV = cms.Sequence(akVs3PFImpactParameterTagInfos
            *
            akVs3PFSecondaryVertexNegativeTagInfos
            * (akVs3PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs3PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs3PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs3PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs3PFJetBtaggingMu = cms.Sequence(akVs3PFSoftMuonTagInfos * (akVs3PFSoftMuonBJetTags
                +
                akVs3PFSoftMuonByIP3dBJetTags
                +
                akVs3PFSoftMuonByPtBJetTags
                +
                akVs3PFNegativeSoftMuonByPtBJetTags
                +
                akVs3PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs3PFJetBtagging = cms.Sequence(akVs3PFJetBtaggingIP
            *akVs3PFJetBtaggingSV
            *akVs3PFJetBtaggingNegSV
            *akVs3PFJetBtaggingMu
            )

akVs3PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs3PFJets"),
        genJetMatch          = cms.InputTag("akVs3PFmatch"),
        genPartonMatch       = cms.InputTag("akVs3PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs3PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs3PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs3PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs3PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs3PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs3PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs3PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs3PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs3PFJetProbabilityBJetTags"),
            cms.InputTag("akVs3PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs3PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs3PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs3PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs3PFJetID"),
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

akVs3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs3PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJetsCleaned',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akVs3PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs3PFJetSequence_mc = cms.Sequence(
                                                  akVs3PFclean
                                                  *
                                                  akVs3PFmatch
                                                  *
                                                  akVs3PFparton
                                                  *
                                                  akVs3PFcorr
                                                  *
                                                  akVs3PFJetID
                                                  *
                                                  akVs3PFPatJetFlavourId
                                                  *
                                                  akVs3PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs3PFJetBtagging
                                                  *
                                                  akVs3PFpatJetsWithBtagging
                                                  *
                                                  akVs3PFJetAnalyzer
                                                  )

akVs3PFJetSequence_data = cms.Sequence(akVs3PFcorr
                                                    *
                                                    akVs3PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs3PFJetBtagging
                                                    *
                                                    akVs3PFpatJetsWithBtagging
                                                    *
                                                    akVs3PFJetAnalyzer
                                                    )

akVs3PFJetSequence_jec = akVs3PFJetSequence_mc
akVs3PFJetSequence_mix = akVs3PFJetSequence_mc

akVs3PFJetSequence = cms.Sequence(akVs3PFJetSequence_mc)
