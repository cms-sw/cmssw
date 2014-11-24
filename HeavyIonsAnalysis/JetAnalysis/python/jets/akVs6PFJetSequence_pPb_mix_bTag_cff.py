

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs6PFJets"),
    matched = cms.InputTag("ak6HiGenJets")
    )

akVs6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs6PFJets")
                                                        )

akVs6PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs6PFJets"),
    payload = "AKVs6PF_generalTracks"
    )

akVs6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs6CaloJets'))

akVs6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

akVs6PFbTagger = bTaggers("akVs6PF")

#create objects locally since they dont load properly otherwise
akVs6PFmatch = akVs6PFbTagger.match
akVs6PFparton = akVs6PFbTagger.parton
akVs6PFPatJetFlavourAssociation = akVs6PFbTagger.PatJetFlavourAssociation
akVs6PFJetTracksAssociatorAtVertex = akVs6PFbTagger.JetTracksAssociatorAtVertex
akVs6PFSimpleSecondaryVertexHighEffBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs6PFSimpleSecondaryVertexHighPurBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs6PFCombinedSecondaryVertexBJetTags = akVs6PFbTagger.CombinedSecondaryVertexBJetTags
akVs6PFCombinedSecondaryVertexMVABJetTags = akVs6PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs6PFJetBProbabilityBJetTags = akVs6PFbTagger.JetBProbabilityBJetTags
akVs6PFSoftMuonByPtBJetTags = akVs6PFbTagger.SoftMuonByPtBJetTags
akVs6PFSoftMuonByIP3dBJetTags = akVs6PFbTagger.SoftMuonByIP3dBJetTags
akVs6PFTrackCountingHighEffBJetTags = akVs6PFbTagger.TrackCountingHighEffBJetTags
akVs6PFTrackCountingHighPurBJetTags = akVs6PFbTagger.TrackCountingHighPurBJetTags
akVs6PFPatJetPartonAssociation = akVs6PFbTagger.PatJetPartonAssociation

akVs6PFImpactParameterTagInfos = akVs6PFbTagger.ImpactParameterTagInfos
akVs6PFJetProbabilityBJetTags = akVs6PFbTagger.JetProbabilityBJetTags
akVs6PFPositiveOnlyJetProbabilityJetTags = akVs6PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs6PFNegativeOnlyJetProbabilityJetTags = akVs6PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs6PFNegativeTrackCountingHighEffJetTags = akVs6PFbTagger.NegativeTrackCountingHighEffJetTags
akVs6PFNegativeTrackCountingHighPur = akVs6PFbTagger.NegativeTrackCountingHighPur
akVs6PFNegativeOnlyJetBProbabilityJetTags = akVs6PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs6PFPositiveOnlyJetBProbabilityJetTags = akVs6PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs6PFSecondaryVertexTagInfos = akVs6PFbTagger.SecondaryVertexTagInfos
akVs6PFSimpleSecondaryVertexHighEffBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs6PFSimpleSecondaryVertexHighPurBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs6PFCombinedSecondaryVertexBJetTags = akVs6PFbTagger.CombinedSecondaryVertexBJetTags
akVs6PFCombinedSecondaryVertexMVABJetTags = akVs6PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs6PFSecondaryVertexNegativeTagInfos = akVs6PFbTagger.SecondaryVertexNegativeTagInfos
akVs6PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs6PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs6PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs6PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs6PFCombinedSecondaryVertexNegativeBJetTags = akVs6PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs6PFCombinedSecondaryVertexPositiveBJetTags = akVs6PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs6PFSoftMuonTagInfos = akVs6PFbTagger.SoftMuonTagInfos
akVs6PFSoftMuonBJetTags = akVs6PFbTagger.SoftMuonBJetTags
akVs6PFSoftMuonByIP3dBJetTags = akVs6PFbTagger.SoftMuonByIP3dBJetTags
akVs6PFSoftMuonByPtBJetTags = akVs6PFbTagger.SoftMuonByPtBJetTags
akVs6PFNegativeSoftMuonByPtBJetTags = akVs6PFbTagger.NegativeSoftMuonByPtBJetTags
akVs6PFPositiveSoftMuonByPtBJetTags = akVs6PFbTagger.PositiveSoftMuonByPtBJetTags

akVs6PFPatJetFlavourId = cms.Sequence(akVs6PFPatJetPartonAssociation*akVs6PFPatJetFlavourAssociation)

akVs6PFJetBtaggingIP       = cms.Sequence(akVs6PFImpactParameterTagInfos *
            (akVs6PFTrackCountingHighEffBJetTags +
             akVs6PFTrackCountingHighPurBJetTags +
             akVs6PFJetProbabilityBJetTags +
             akVs6PFJetBProbabilityBJetTags +
             akVs6PFPositiveOnlyJetProbabilityJetTags +
             akVs6PFNegativeOnlyJetProbabilityJetTags +
             akVs6PFNegativeTrackCountingHighEffJetTags +
             akVs6PFNegativeTrackCountingHighPur +
             akVs6PFNegativeOnlyJetBProbabilityJetTags +
             akVs6PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs6PFJetBtaggingSV = cms.Sequence(akVs6PFImpactParameterTagInfos
            *
            akVs6PFSecondaryVertexTagInfos
            * (akVs6PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs6PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs6PFCombinedSecondaryVertexBJetTags
                +
                akVs6PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs6PFJetBtaggingNegSV = cms.Sequence(akVs6PFImpactParameterTagInfos
            *
            akVs6PFSecondaryVertexNegativeTagInfos
            * (akVs6PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs6PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs6PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs6PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs6PFJetBtaggingMu = cms.Sequence(akVs6PFSoftMuonTagInfos * (akVs6PFSoftMuonBJetTags
                +
                akVs6PFSoftMuonByIP3dBJetTags
                +
                akVs6PFSoftMuonByPtBJetTags
                +
                akVs6PFNegativeSoftMuonByPtBJetTags
                +
                akVs6PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs6PFJetBtagging = cms.Sequence(akVs6PFJetBtaggingIP
            *akVs6PFJetBtaggingSV
            *akVs6PFJetBtaggingNegSV
            *akVs6PFJetBtaggingMu
            )

akVs6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs6PFJets"),
        genJetMatch          = cms.InputTag("akVs6PFmatch"),
        genPartonMatch       = cms.InputTag("akVs6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs6PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs6PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs6PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs6PFJetProbabilityBJetTags"),
            cms.InputTag("akVs6PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs6PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs6PFJetID"),
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

akVs6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs6PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs6PFJetSequence_mc = cms.Sequence(
                                                  akVs6PFclean
                                                  *
                                                  akVs6PFmatch
                                                  *
                                                  akVs6PFparton
                                                  *
                                                  akVs6PFcorr
                                                  *
                                                  akVs6PFJetID
                                                  *
                                                  akVs6PFPatJetFlavourId
                                                  *
                                                  akVs6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs6PFJetBtagging
                                                  *
                                                  akVs6PFpatJetsWithBtagging
                                                  *
                                                  akVs6PFJetAnalyzer
                                                  )

akVs6PFJetSequence_data = cms.Sequence(akVs6PFcorr
                                                    *
                                                    akVs6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs6PFJetBtagging
                                                    *
                                                    akVs6PFpatJetsWithBtagging
                                                    *
                                                    akVs6PFJetAnalyzer
                                                    )

akVs6PFJetSequence_jec = akVs6PFJetSequence_mc
akVs6PFJetSequence_mix = akVs6PFJetSequence_mc

akVs6PFJetSequence = cms.Sequence(akVs6PFJetSequence_mix)
