

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs5PFJets"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

akVs5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5PFJets")
                                                        )

akVs5PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs5PFJets"),
    payload = "AKVs5PF_hiIterativeTracks"
    )

akVs5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs5CaloJets'))

akVs5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJetsCleaned'))

akVs5PFbTagger = bTaggers("akVs5PF")

#create objects locally since they dont load properly otherwise
akVs5PFmatch = akVs5PFbTagger.match
akVs5PFparton = akVs5PFbTagger.parton
akVs5PFPatJetFlavourAssociation = akVs5PFbTagger.PatJetFlavourAssociation
akVs5PFJetTracksAssociatorAtVertex = akVs5PFbTagger.JetTracksAssociatorAtVertex
akVs5PFSimpleSecondaryVertexHighEffBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5PFSimpleSecondaryVertexHighPurBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5PFCombinedSecondaryVertexBJetTags = akVs5PFbTagger.CombinedSecondaryVertexBJetTags
akVs5PFCombinedSecondaryVertexMVABJetTags = akVs5PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs5PFJetBProbabilityBJetTags = akVs5PFbTagger.JetBProbabilityBJetTags
akVs5PFSoftMuonByPtBJetTags = akVs5PFbTagger.SoftMuonByPtBJetTags
akVs5PFSoftMuonByIP3dBJetTags = akVs5PFbTagger.SoftMuonByIP3dBJetTags
akVs5PFTrackCountingHighEffBJetTags = akVs5PFbTagger.TrackCountingHighEffBJetTags
akVs5PFTrackCountingHighPurBJetTags = akVs5PFbTagger.TrackCountingHighPurBJetTags
akVs5PFPatJetPartonAssociation = akVs5PFbTagger.PatJetPartonAssociation

akVs5PFImpactParameterTagInfos = akVs5PFbTagger.ImpactParameterTagInfos
akVs5PFJetProbabilityBJetTags = akVs5PFbTagger.JetProbabilityBJetTags
akVs5PFPositiveOnlyJetProbabilityJetTags = akVs5PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs5PFNegativeOnlyJetProbabilityJetTags = akVs5PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs5PFNegativeTrackCountingHighEffJetTags = akVs5PFbTagger.NegativeTrackCountingHighEffJetTags
akVs5PFNegativeTrackCountingHighPur = akVs5PFbTagger.NegativeTrackCountingHighPur
akVs5PFNegativeOnlyJetBProbabilityJetTags = akVs5PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs5PFPositiveOnlyJetBProbabilityJetTags = akVs5PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs5PFSecondaryVertexTagInfos = akVs5PFbTagger.SecondaryVertexTagInfos
akVs5PFSimpleSecondaryVertexHighEffBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5PFSimpleSecondaryVertexHighPurBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5PFCombinedSecondaryVertexBJetTags = akVs5PFbTagger.CombinedSecondaryVertexBJetTags
akVs5PFCombinedSecondaryVertexMVABJetTags = akVs5PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs5PFSecondaryVertexNegativeTagInfos = akVs5PFbTagger.SecondaryVertexNegativeTagInfos
akVs5PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs5PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs5PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs5PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs5PFCombinedSecondaryVertexNegativeBJetTags = akVs5PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs5PFCombinedSecondaryVertexPositiveBJetTags = akVs5PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs5PFSoftMuonTagInfos = akVs5PFbTagger.SoftMuonTagInfos
akVs5PFSoftMuonBJetTags = akVs5PFbTagger.SoftMuonBJetTags
akVs5PFSoftMuonByIP3dBJetTags = akVs5PFbTagger.SoftMuonByIP3dBJetTags
akVs5PFSoftMuonByPtBJetTags = akVs5PFbTagger.SoftMuonByPtBJetTags
akVs5PFNegativeSoftMuonByPtBJetTags = akVs5PFbTagger.NegativeSoftMuonByPtBJetTags
akVs5PFPositiveSoftMuonByPtBJetTags = akVs5PFbTagger.PositiveSoftMuonByPtBJetTags

akVs5PFPatJetFlavourId = cms.Sequence(akVs5PFPatJetPartonAssociation*akVs5PFPatJetFlavourAssociation)

akVs5PFJetBtaggingIP       = cms.Sequence(akVs5PFImpactParameterTagInfos *
            (akVs5PFTrackCountingHighEffBJetTags +
             akVs5PFTrackCountingHighPurBJetTags +
             akVs5PFJetProbabilityBJetTags +
             akVs5PFJetBProbabilityBJetTags +
             akVs5PFPositiveOnlyJetProbabilityJetTags +
             akVs5PFNegativeOnlyJetProbabilityJetTags +
             akVs5PFNegativeTrackCountingHighEffJetTags +
             akVs5PFNegativeTrackCountingHighPur +
             akVs5PFNegativeOnlyJetBProbabilityJetTags +
             akVs5PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs5PFJetBtaggingSV = cms.Sequence(akVs5PFImpactParameterTagInfos
            *
            akVs5PFSecondaryVertexTagInfos
            * (akVs5PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs5PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs5PFCombinedSecondaryVertexBJetTags
                +
                akVs5PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs5PFJetBtaggingNegSV = cms.Sequence(akVs5PFImpactParameterTagInfos
            *
            akVs5PFSecondaryVertexNegativeTagInfos
            * (akVs5PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs5PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs5PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs5PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs5PFJetBtaggingMu = cms.Sequence(akVs5PFSoftMuonTagInfos * (akVs5PFSoftMuonBJetTags
                +
                akVs5PFSoftMuonByIP3dBJetTags
                +
                akVs5PFSoftMuonByPtBJetTags
                +
                akVs5PFNegativeSoftMuonByPtBJetTags
                +
                akVs5PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs5PFJetBtagging = cms.Sequence(akVs5PFJetBtaggingIP
            *akVs5PFJetBtaggingSV
            *akVs5PFJetBtaggingNegSV
            *akVs5PFJetBtaggingMu
            )

akVs5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs5PFJets"),
        genJetMatch          = cms.InputTag("akVs5PFmatch"),
        genPartonMatch       = cms.InputTag("akVs5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs5PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs5PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs5PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs5PFJetProbabilityBJetTags"),
            cms.InputTag("akVs5PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs5PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs5PFJetID"),
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

akVs5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akVs5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs5PFJetSequence_mc = cms.Sequence(
                                                  akVs5PFclean
                                                  *
                                                  akVs5PFmatch
                                                  *
                                                  akVs5PFparton
                                                  *
                                                  akVs5PFcorr
                                                  *
                                                  akVs5PFJetID
                                                  *
                                                  akVs5PFPatJetFlavourId
                                                  *
                                                  akVs5PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs5PFJetBtagging
                                                  *
                                                  akVs5PFpatJetsWithBtagging
                                                  *
                                                  akVs5PFJetAnalyzer
                                                  )

akVs5PFJetSequence_data = cms.Sequence(akVs5PFcorr
                                                    *
                                                    akVs5PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs5PFJetBtagging
                                                    *
                                                    akVs5PFpatJetsWithBtagging
                                                    *
                                                    akVs5PFJetAnalyzer
                                                    )

akVs5PFJetSequence_jec = akVs5PFJetSequence_mc
akVs5PFJetSequence_mix = akVs5PFJetSequence_mc

akVs5PFJetSequence = cms.Sequence(akVs5PFJetSequence_mix)
