

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs7PFJets"),
    matched = cms.InputTag("ak7HiGenJets")
    )

akVs7PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs7PFJets")
                                                        )

akVs7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs7PFJets"),
    payload = "AKVs7PF_generalTracks"
    )

akVs7PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs7CaloJets'))

akVs7PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

akVs7PFbTagger = bTaggers("akVs7PF")

#create objects locally since they dont load properly otherwise
akVs7PFmatch = akVs7PFbTagger.match
akVs7PFparton = akVs7PFbTagger.parton
akVs7PFPatJetFlavourAssociation = akVs7PFbTagger.PatJetFlavourAssociation
akVs7PFJetTracksAssociatorAtVertex = akVs7PFbTagger.JetTracksAssociatorAtVertex
akVs7PFSimpleSecondaryVertexHighEffBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7PFSimpleSecondaryVertexHighPurBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7PFCombinedSecondaryVertexBJetTags = akVs7PFbTagger.CombinedSecondaryVertexBJetTags
akVs7PFCombinedSecondaryVertexMVABJetTags = akVs7PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs7PFJetBProbabilityBJetTags = akVs7PFbTagger.JetBProbabilityBJetTags
akVs7PFSoftMuonByPtBJetTags = akVs7PFbTagger.SoftMuonByPtBJetTags
akVs7PFSoftMuonByIP3dBJetTags = akVs7PFbTagger.SoftMuonByIP3dBJetTags
akVs7PFTrackCountingHighEffBJetTags = akVs7PFbTagger.TrackCountingHighEffBJetTags
akVs7PFTrackCountingHighPurBJetTags = akVs7PFbTagger.TrackCountingHighPurBJetTags
akVs7PFPatJetPartonAssociation = akVs7PFbTagger.PatJetPartonAssociation

akVs7PFImpactParameterTagInfos = akVs7PFbTagger.ImpactParameterTagInfos
akVs7PFJetProbabilityBJetTags = akVs7PFbTagger.JetProbabilityBJetTags
akVs7PFPositiveOnlyJetProbabilityJetTags = akVs7PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs7PFNegativeOnlyJetProbabilityJetTags = akVs7PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs7PFNegativeTrackCountingHighEffJetTags = akVs7PFbTagger.NegativeTrackCountingHighEffJetTags
akVs7PFNegativeTrackCountingHighPur = akVs7PFbTagger.NegativeTrackCountingHighPur
akVs7PFNegativeOnlyJetBProbabilityJetTags = akVs7PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs7PFPositiveOnlyJetBProbabilityJetTags = akVs7PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs7PFSecondaryVertexTagInfos = akVs7PFbTagger.SecondaryVertexTagInfos
akVs7PFSimpleSecondaryVertexHighEffBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7PFSimpleSecondaryVertexHighPurBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7PFCombinedSecondaryVertexBJetTags = akVs7PFbTagger.CombinedSecondaryVertexBJetTags
akVs7PFCombinedSecondaryVertexMVABJetTags = akVs7PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs7PFSecondaryVertexNegativeTagInfos = akVs7PFbTagger.SecondaryVertexNegativeTagInfos
akVs7PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs7PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs7PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs7PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs7PFCombinedSecondaryVertexNegativeBJetTags = akVs7PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs7PFCombinedSecondaryVertexPositiveBJetTags = akVs7PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs7PFSoftMuonTagInfos = akVs7PFbTagger.SoftMuonTagInfos
akVs7PFSoftMuonBJetTags = akVs7PFbTagger.SoftMuonBJetTags
akVs7PFSoftMuonByIP3dBJetTags = akVs7PFbTagger.SoftMuonByIP3dBJetTags
akVs7PFSoftMuonByPtBJetTags = akVs7PFbTagger.SoftMuonByPtBJetTags
akVs7PFNegativeSoftMuonByPtBJetTags = akVs7PFbTagger.NegativeSoftMuonByPtBJetTags
akVs7PFPositiveSoftMuonByPtBJetTags = akVs7PFbTagger.PositiveSoftMuonByPtBJetTags

akVs7PFPatJetFlavourId = cms.Sequence(akVs7PFPatJetPartonAssociation*akVs7PFPatJetFlavourAssociation)

akVs7PFJetBtaggingIP       = cms.Sequence(akVs7PFImpactParameterTagInfos *
            (akVs7PFTrackCountingHighEffBJetTags +
             akVs7PFTrackCountingHighPurBJetTags +
             akVs7PFJetProbabilityBJetTags +
             akVs7PFJetBProbabilityBJetTags +
             akVs7PFPositiveOnlyJetProbabilityJetTags +
             akVs7PFNegativeOnlyJetProbabilityJetTags +
             akVs7PFNegativeTrackCountingHighEffJetTags +
             akVs7PFNegativeTrackCountingHighPur +
             akVs7PFNegativeOnlyJetBProbabilityJetTags +
             akVs7PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs7PFJetBtaggingSV = cms.Sequence(akVs7PFImpactParameterTagInfos
            *
            akVs7PFSecondaryVertexTagInfos
            * (akVs7PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs7PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs7PFCombinedSecondaryVertexBJetTags
                +
                akVs7PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs7PFJetBtaggingNegSV = cms.Sequence(akVs7PFImpactParameterTagInfos
            *
            akVs7PFSecondaryVertexNegativeTagInfos
            * (akVs7PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs7PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs7PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs7PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs7PFJetBtaggingMu = cms.Sequence(akVs7PFSoftMuonTagInfos * (akVs7PFSoftMuonBJetTags
                +
                akVs7PFSoftMuonByIP3dBJetTags
                +
                akVs7PFSoftMuonByPtBJetTags
                +
                akVs7PFNegativeSoftMuonByPtBJetTags
                +
                akVs7PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs7PFJetBtagging = cms.Sequence(akVs7PFJetBtaggingIP
            *akVs7PFJetBtaggingSV
            *akVs7PFJetBtaggingNegSV
            *akVs7PFJetBtaggingMu
            )

akVs7PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs7PFJets"),
        genJetMatch          = cms.InputTag("akVs7PFmatch"),
        genPartonMatch       = cms.InputTag("akVs7PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs7PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs7PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs7PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs7PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs7PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs7PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs7PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs7PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs7PFJetProbabilityBJetTags"),
            cms.InputTag("akVs7PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs7PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs7PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs7PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs7PFJetID"),
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

akVs7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs7PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
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
                                                             bTagJetName = cms.untracked.string("akVs7PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs7PFJetSequence_mc = cms.Sequence(
                                                  akVs7PFclean
                                                  *
                                                  akVs7PFmatch
                                                  *
                                                  akVs7PFparton
                                                  *
                                                  akVs7PFcorr
                                                  *
                                                  akVs7PFJetID
                                                  *
                                                  akVs7PFPatJetFlavourId
                                                  *
                                                  akVs7PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs7PFJetBtagging
                                                  *
                                                  akVs7PFpatJetsWithBtagging
                                                  *
                                                  akVs7PFJetAnalyzer
                                                  )

akVs7PFJetSequence_data = cms.Sequence(akVs7PFcorr
                                                    *
                                                    akVs7PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs7PFJetBtagging
                                                    *
                                                    akVs7PFpatJetsWithBtagging
                                                    *
                                                    akVs7PFJetAnalyzer
                                                    )

akVs7PFJetSequence_jec = akVs7PFJetSequence_mc
akVs7PFJetSequence_mix = akVs7PFJetSequence_mc

akVs7PFJetSequence = cms.Sequence(akVs7PFJetSequence_mc)
