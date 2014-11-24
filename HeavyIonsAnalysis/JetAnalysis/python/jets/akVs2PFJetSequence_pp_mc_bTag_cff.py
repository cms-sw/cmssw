

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs2PFJets"),
    matched = cms.InputTag("ak2HiGenJets")
    )

akVs2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2PFJets")
                                                        )

akVs2PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs2PFJets"),
    payload = "AKVs2PF_generalTracks"
    )

akVs2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs2CaloJets'))

akVs2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

akVs2PFbTagger = bTaggers("akVs2PF")

#create objects locally since they dont load properly otherwise
akVs2PFmatch = akVs2PFbTagger.match
akVs2PFparton = akVs2PFbTagger.parton
akVs2PFPatJetFlavourAssociation = akVs2PFbTagger.PatJetFlavourAssociation
akVs2PFJetTracksAssociatorAtVertex = akVs2PFbTagger.JetTracksAssociatorAtVertex
akVs2PFSimpleSecondaryVertexHighEffBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2PFSimpleSecondaryVertexHighPurBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2PFCombinedSecondaryVertexBJetTags = akVs2PFbTagger.CombinedSecondaryVertexBJetTags
akVs2PFCombinedSecondaryVertexMVABJetTags = akVs2PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs2PFJetBProbabilityBJetTags = akVs2PFbTagger.JetBProbabilityBJetTags
akVs2PFSoftMuonByPtBJetTags = akVs2PFbTagger.SoftMuonByPtBJetTags
akVs2PFSoftMuonByIP3dBJetTags = akVs2PFbTagger.SoftMuonByIP3dBJetTags
akVs2PFTrackCountingHighEffBJetTags = akVs2PFbTagger.TrackCountingHighEffBJetTags
akVs2PFTrackCountingHighPurBJetTags = akVs2PFbTagger.TrackCountingHighPurBJetTags
akVs2PFPatJetPartonAssociation = akVs2PFbTagger.PatJetPartonAssociation

akVs2PFImpactParameterTagInfos = akVs2PFbTagger.ImpactParameterTagInfos
akVs2PFJetProbabilityBJetTags = akVs2PFbTagger.JetProbabilityBJetTags
akVs2PFPositiveOnlyJetProbabilityJetTags = akVs2PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs2PFNegativeOnlyJetProbabilityJetTags = akVs2PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs2PFNegativeTrackCountingHighEffJetTags = akVs2PFbTagger.NegativeTrackCountingHighEffJetTags
akVs2PFNegativeTrackCountingHighPur = akVs2PFbTagger.NegativeTrackCountingHighPur
akVs2PFNegativeOnlyJetBProbabilityJetTags = akVs2PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs2PFPositiveOnlyJetBProbabilityJetTags = akVs2PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs2PFSecondaryVertexTagInfos = akVs2PFbTagger.SecondaryVertexTagInfos
akVs2PFSimpleSecondaryVertexHighEffBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2PFSimpleSecondaryVertexHighPurBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2PFCombinedSecondaryVertexBJetTags = akVs2PFbTagger.CombinedSecondaryVertexBJetTags
akVs2PFCombinedSecondaryVertexMVABJetTags = akVs2PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs2PFSecondaryVertexNegativeTagInfos = akVs2PFbTagger.SecondaryVertexNegativeTagInfos
akVs2PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs2PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs2PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs2PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs2PFCombinedSecondaryVertexNegativeBJetTags = akVs2PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs2PFCombinedSecondaryVertexPositiveBJetTags = akVs2PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs2PFSoftMuonTagInfos = akVs2PFbTagger.SoftMuonTagInfos
akVs2PFSoftMuonBJetTags = akVs2PFbTagger.SoftMuonBJetTags
akVs2PFSoftMuonByIP3dBJetTags = akVs2PFbTagger.SoftMuonByIP3dBJetTags
akVs2PFSoftMuonByPtBJetTags = akVs2PFbTagger.SoftMuonByPtBJetTags
akVs2PFNegativeSoftMuonByPtBJetTags = akVs2PFbTagger.NegativeSoftMuonByPtBJetTags
akVs2PFPositiveSoftMuonByPtBJetTags = akVs2PFbTagger.PositiveSoftMuonByPtBJetTags

akVs2PFPatJetFlavourId = cms.Sequence(akVs2PFPatJetPartonAssociation*akVs2PFPatJetFlavourAssociation)

akVs2PFJetBtaggingIP       = cms.Sequence(akVs2PFImpactParameterTagInfos *
            (akVs2PFTrackCountingHighEffBJetTags +
             akVs2PFTrackCountingHighPurBJetTags +
             akVs2PFJetProbabilityBJetTags +
             akVs2PFJetBProbabilityBJetTags +
             akVs2PFPositiveOnlyJetProbabilityJetTags +
             akVs2PFNegativeOnlyJetProbabilityJetTags +
             akVs2PFNegativeTrackCountingHighEffJetTags +
             akVs2PFNegativeTrackCountingHighPur +
             akVs2PFNegativeOnlyJetBProbabilityJetTags +
             akVs2PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs2PFJetBtaggingSV = cms.Sequence(akVs2PFImpactParameterTagInfos
            *
            akVs2PFSecondaryVertexTagInfos
            * (akVs2PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs2PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs2PFCombinedSecondaryVertexBJetTags
                +
                akVs2PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs2PFJetBtaggingNegSV = cms.Sequence(akVs2PFImpactParameterTagInfos
            *
            akVs2PFSecondaryVertexNegativeTagInfos
            * (akVs2PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs2PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs2PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs2PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs2PFJetBtaggingMu = cms.Sequence(akVs2PFSoftMuonTagInfos * (akVs2PFSoftMuonBJetTags
                +
                akVs2PFSoftMuonByIP3dBJetTags
                +
                akVs2PFSoftMuonByPtBJetTags
                +
                akVs2PFNegativeSoftMuonByPtBJetTags
                +
                akVs2PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs2PFJetBtagging = cms.Sequence(akVs2PFJetBtaggingIP
            *akVs2PFJetBtaggingSV
            *akVs2PFJetBtaggingNegSV
            *akVs2PFJetBtaggingMu
            )

akVs2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs2PFJets"),
        genJetMatch          = cms.InputTag("akVs2PFmatch"),
        genPartonMatch       = cms.InputTag("akVs2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs2PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs2PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs2PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs2PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs2PFJetProbabilityBJetTags"),
            cms.InputTag("akVs2PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs2PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs2PFJetID"),
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

akVs2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs2PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs2PFJetSequence_mc = cms.Sequence(
                                                  akVs2PFclean
                                                  *
                                                  akVs2PFmatch
                                                  *
                                                  akVs2PFparton
                                                  *
                                                  akVs2PFcorr
                                                  *
                                                  akVs2PFJetID
                                                  *
                                                  akVs2PFPatJetFlavourId
                                                  *
                                                  akVs2PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs2PFJetBtagging
                                                  *
                                                  akVs2PFpatJetsWithBtagging
                                                  *
                                                  akVs2PFJetAnalyzer
                                                  )

akVs2PFJetSequence_data = cms.Sequence(akVs2PFcorr
                                                    *
                                                    akVs2PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs2PFJetBtagging
                                                    *
                                                    akVs2PFpatJetsWithBtagging
                                                    *
                                                    akVs2PFJetAnalyzer
                                                    )

akVs2PFJetSequence_jec = akVs2PFJetSequence_mc
akVs2PFJetSequence_mix = akVs2PFJetSequence_mc

akVs2PFJetSequence = cms.Sequence(akVs2PFJetSequence_mc)
