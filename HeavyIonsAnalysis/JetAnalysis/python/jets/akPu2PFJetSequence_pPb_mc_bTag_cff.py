

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu2PFJets"),
    matched = cms.InputTag("ak2HiGenJets")
    )

akPu2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu2PFJets")
                                                        )

akPu2PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu2PFJets"),
    payload = "AKPu2PF_generalTracks"
    )

akPu2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu2CaloJets'))

akPu2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

akPu2PFbTagger = bTaggers("akPu2PF")

#create objects locally since they dont load properly otherwise
akPu2PFmatch = akPu2PFbTagger.match
akPu2PFparton = akPu2PFbTagger.parton
akPu2PFPatJetFlavourAssociation = akPu2PFbTagger.PatJetFlavourAssociation
akPu2PFJetTracksAssociatorAtVertex = akPu2PFbTagger.JetTracksAssociatorAtVertex
akPu2PFSimpleSecondaryVertexHighEffBJetTags = akPu2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu2PFSimpleSecondaryVertexHighPurBJetTags = akPu2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu2PFCombinedSecondaryVertexBJetTags = akPu2PFbTagger.CombinedSecondaryVertexBJetTags
akPu2PFCombinedSecondaryVertexMVABJetTags = akPu2PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu2PFJetBProbabilityBJetTags = akPu2PFbTagger.JetBProbabilityBJetTags
akPu2PFSoftMuonByPtBJetTags = akPu2PFbTagger.SoftMuonByPtBJetTags
akPu2PFSoftMuonByIP3dBJetTags = akPu2PFbTagger.SoftMuonByIP3dBJetTags
akPu2PFTrackCountingHighEffBJetTags = akPu2PFbTagger.TrackCountingHighEffBJetTags
akPu2PFTrackCountingHighPurBJetTags = akPu2PFbTagger.TrackCountingHighPurBJetTags
akPu2PFPatJetPartonAssociation = akPu2PFbTagger.PatJetPartonAssociation

akPu2PFImpactParameterTagInfos = akPu2PFbTagger.ImpactParameterTagInfos
akPu2PFJetProbabilityBJetTags = akPu2PFbTagger.JetProbabilityBJetTags
akPu2PFPositiveOnlyJetProbabilityJetTags = akPu2PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu2PFNegativeOnlyJetProbabilityJetTags = akPu2PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu2PFNegativeTrackCountingHighEffJetTags = akPu2PFbTagger.NegativeTrackCountingHighEffJetTags
akPu2PFNegativeTrackCountingHighPur = akPu2PFbTagger.NegativeTrackCountingHighPur
akPu2PFNegativeOnlyJetBProbabilityJetTags = akPu2PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu2PFPositiveOnlyJetBProbabilityJetTags = akPu2PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu2PFSecondaryVertexTagInfos = akPu2PFbTagger.SecondaryVertexTagInfos
akPu2PFSimpleSecondaryVertexHighEffBJetTags = akPu2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu2PFSimpleSecondaryVertexHighPurBJetTags = akPu2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu2PFCombinedSecondaryVertexBJetTags = akPu2PFbTagger.CombinedSecondaryVertexBJetTags
akPu2PFCombinedSecondaryVertexMVABJetTags = akPu2PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu2PFSecondaryVertexNegativeTagInfos = akPu2PFbTagger.SecondaryVertexNegativeTagInfos
akPu2PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu2PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu2PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu2PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu2PFCombinedSecondaryVertexNegativeBJetTags = akPu2PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu2PFCombinedSecondaryVertexPositiveBJetTags = akPu2PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu2PFSoftMuonTagInfos = akPu2PFbTagger.SoftMuonTagInfos
akPu2PFSoftMuonBJetTags = akPu2PFbTagger.SoftMuonBJetTags
akPu2PFSoftMuonByIP3dBJetTags = akPu2PFbTagger.SoftMuonByIP3dBJetTags
akPu2PFSoftMuonByPtBJetTags = akPu2PFbTagger.SoftMuonByPtBJetTags
akPu2PFNegativeSoftMuonByPtBJetTags = akPu2PFbTagger.NegativeSoftMuonByPtBJetTags
akPu2PFPositiveSoftMuonByPtBJetTags = akPu2PFbTagger.PositiveSoftMuonByPtBJetTags

akPu2PFPatJetFlavourId = cms.Sequence(akPu2PFPatJetPartonAssociation*akPu2PFPatJetFlavourAssociation)

akPu2PFJetBtaggingIP       = cms.Sequence(akPu2PFImpactParameterTagInfos *
            (akPu2PFTrackCountingHighEffBJetTags +
             akPu2PFTrackCountingHighPurBJetTags +
             akPu2PFJetProbabilityBJetTags +
             akPu2PFJetBProbabilityBJetTags +
             akPu2PFPositiveOnlyJetProbabilityJetTags +
             akPu2PFNegativeOnlyJetProbabilityJetTags +
             akPu2PFNegativeTrackCountingHighEffJetTags +
             akPu2PFNegativeTrackCountingHighPur +
             akPu2PFNegativeOnlyJetBProbabilityJetTags +
             akPu2PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu2PFJetBtaggingSV = cms.Sequence(akPu2PFImpactParameterTagInfos
            *
            akPu2PFSecondaryVertexTagInfos
            * (akPu2PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu2PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu2PFCombinedSecondaryVertexBJetTags
                +
                akPu2PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu2PFJetBtaggingNegSV = cms.Sequence(akPu2PFImpactParameterTagInfos
            *
            akPu2PFSecondaryVertexNegativeTagInfos
            * (akPu2PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu2PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu2PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu2PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu2PFJetBtaggingMu = cms.Sequence(akPu2PFSoftMuonTagInfos * (akPu2PFSoftMuonBJetTags
                +
                akPu2PFSoftMuonByIP3dBJetTags
                +
                akPu2PFSoftMuonByPtBJetTags
                +
                akPu2PFNegativeSoftMuonByPtBJetTags
                +
                akPu2PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu2PFJetBtagging = cms.Sequence(akPu2PFJetBtaggingIP
            *akPu2PFJetBtaggingSV
            *akPu2PFJetBtaggingNegSV
            *akPu2PFJetBtaggingMu
            )

akPu2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu2PFJets"),
        genJetMatch          = cms.InputTag("akPu2PFmatch"),
        genPartonMatch       = cms.InputTag("akPu2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu2PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu2PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu2PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu2PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu2PFJetProbabilityBJetTags"),
            cms.InputTag("akPu2PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu2PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu2PFJetID"),
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

akPu2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu2PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu2PFJetSequence_mc = cms.Sequence(
                                                  akPu2PFclean
                                                  *
                                                  akPu2PFmatch
                                                  *
                                                  akPu2PFparton
                                                  *
                                                  akPu2PFcorr
                                                  *
                                                  akPu2PFJetID
                                                  *
                                                  akPu2PFPatJetFlavourId
                                                  *
                                                  akPu2PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu2PFJetBtagging
                                                  *
                                                  akPu2PFpatJetsWithBtagging
                                                  *
                                                  akPu2PFJetAnalyzer
                                                  )

akPu2PFJetSequence_data = cms.Sequence(akPu2PFcorr
                                                    *
                                                    akPu2PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu2PFJetBtagging
                                                    *
                                                    akPu2PFpatJetsWithBtagging
                                                    *
                                                    akPu2PFJetAnalyzer
                                                    )

akPu2PFJetSequence_jec = akPu2PFJetSequence_mc
akPu2PFJetSequence_mix = akPu2PFJetSequence_mc

akPu2PFJetSequence = cms.Sequence(akPu2PFJetSequence_mc)
