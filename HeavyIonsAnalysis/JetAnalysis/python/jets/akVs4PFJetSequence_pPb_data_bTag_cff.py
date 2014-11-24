

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs4PFJets"),
    matched = cms.InputTag("ak4HiGenJets")
    )

akVs4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4PFJets")
                                                        )

akVs4PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs4PFJets"),
    payload = "AKVs4PF_generalTracks"
    )

akVs4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs4CaloJets'))

akVs4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

akVs4PFbTagger = bTaggers("akVs4PF")

#create objects locally since they dont load properly otherwise
akVs4PFmatch = akVs4PFbTagger.match
akVs4PFparton = akVs4PFbTagger.parton
akVs4PFPatJetFlavourAssociation = akVs4PFbTagger.PatJetFlavourAssociation
akVs4PFJetTracksAssociatorAtVertex = akVs4PFbTagger.JetTracksAssociatorAtVertex
akVs4PFSimpleSecondaryVertexHighEffBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4PFSimpleSecondaryVertexHighPurBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4PFCombinedSecondaryVertexBJetTags = akVs4PFbTagger.CombinedSecondaryVertexBJetTags
akVs4PFCombinedSecondaryVertexMVABJetTags = akVs4PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs4PFJetBProbabilityBJetTags = akVs4PFbTagger.JetBProbabilityBJetTags
akVs4PFSoftMuonByPtBJetTags = akVs4PFbTagger.SoftMuonByPtBJetTags
akVs4PFSoftMuonByIP3dBJetTags = akVs4PFbTagger.SoftMuonByIP3dBJetTags
akVs4PFTrackCountingHighEffBJetTags = akVs4PFbTagger.TrackCountingHighEffBJetTags
akVs4PFTrackCountingHighPurBJetTags = akVs4PFbTagger.TrackCountingHighPurBJetTags
akVs4PFPatJetPartonAssociation = akVs4PFbTagger.PatJetPartonAssociation

akVs4PFImpactParameterTagInfos = akVs4PFbTagger.ImpactParameterTagInfos
akVs4PFJetProbabilityBJetTags = akVs4PFbTagger.JetProbabilityBJetTags
akVs4PFPositiveOnlyJetProbabilityJetTags = akVs4PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs4PFNegativeOnlyJetProbabilityJetTags = akVs4PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs4PFNegativeTrackCountingHighEffJetTags = akVs4PFbTagger.NegativeTrackCountingHighEffJetTags
akVs4PFNegativeTrackCountingHighPur = akVs4PFbTagger.NegativeTrackCountingHighPur
akVs4PFNegativeOnlyJetBProbabilityJetTags = akVs4PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs4PFPositiveOnlyJetBProbabilityJetTags = akVs4PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs4PFSecondaryVertexTagInfos = akVs4PFbTagger.SecondaryVertexTagInfos
akVs4PFSimpleSecondaryVertexHighEffBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4PFSimpleSecondaryVertexHighPurBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4PFCombinedSecondaryVertexBJetTags = akVs4PFbTagger.CombinedSecondaryVertexBJetTags
akVs4PFCombinedSecondaryVertexMVABJetTags = akVs4PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs4PFSecondaryVertexNegativeTagInfos = akVs4PFbTagger.SecondaryVertexNegativeTagInfos
akVs4PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs4PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs4PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs4PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs4PFCombinedSecondaryVertexNegativeBJetTags = akVs4PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs4PFCombinedSecondaryVertexPositiveBJetTags = akVs4PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs4PFSoftMuonTagInfos = akVs4PFbTagger.SoftMuonTagInfos
akVs4PFSoftMuonBJetTags = akVs4PFbTagger.SoftMuonBJetTags
akVs4PFSoftMuonByIP3dBJetTags = akVs4PFbTagger.SoftMuonByIP3dBJetTags
akVs4PFSoftMuonByPtBJetTags = akVs4PFbTagger.SoftMuonByPtBJetTags
akVs4PFNegativeSoftMuonByPtBJetTags = akVs4PFbTagger.NegativeSoftMuonByPtBJetTags
akVs4PFPositiveSoftMuonByPtBJetTags = akVs4PFbTagger.PositiveSoftMuonByPtBJetTags

akVs4PFPatJetFlavourId = cms.Sequence(akVs4PFPatJetPartonAssociation*akVs4PFPatJetFlavourAssociation)

akVs4PFJetBtaggingIP       = cms.Sequence(akVs4PFImpactParameterTagInfos *
            (akVs4PFTrackCountingHighEffBJetTags +
             akVs4PFTrackCountingHighPurBJetTags +
             akVs4PFJetProbabilityBJetTags +
             akVs4PFJetBProbabilityBJetTags +
             akVs4PFPositiveOnlyJetProbabilityJetTags +
             akVs4PFNegativeOnlyJetProbabilityJetTags +
             akVs4PFNegativeTrackCountingHighEffJetTags +
             akVs4PFNegativeTrackCountingHighPur +
             akVs4PFNegativeOnlyJetBProbabilityJetTags +
             akVs4PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs4PFJetBtaggingSV = cms.Sequence(akVs4PFImpactParameterTagInfos
            *
            akVs4PFSecondaryVertexTagInfos
            * (akVs4PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs4PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs4PFCombinedSecondaryVertexBJetTags
                +
                akVs4PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs4PFJetBtaggingNegSV = cms.Sequence(akVs4PFImpactParameterTagInfos
            *
            akVs4PFSecondaryVertexNegativeTagInfos
            * (akVs4PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs4PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs4PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs4PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs4PFJetBtaggingMu = cms.Sequence(akVs4PFSoftMuonTagInfos * (akVs4PFSoftMuonBJetTags
                +
                akVs4PFSoftMuonByIP3dBJetTags
                +
                akVs4PFSoftMuonByPtBJetTags
                +
                akVs4PFNegativeSoftMuonByPtBJetTags
                +
                akVs4PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs4PFJetBtagging = cms.Sequence(akVs4PFJetBtaggingIP
            *akVs4PFJetBtaggingSV
            *akVs4PFJetBtaggingNegSV
            *akVs4PFJetBtaggingMu
            )

akVs4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs4PFJets"),
        genJetMatch          = cms.InputTag("akVs4PFmatch"),
        genPartonMatch       = cms.InputTag("akVs4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs4PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs4PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs4PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs4PFJetProbabilityBJetTags"),
            cms.InputTag("akVs4PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs4PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs4PFJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        embedCaloTowers = False,
        embedPFCandidates = True
        )

akVs4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs4PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs4PFJetSequence_mc = cms.Sequence(
                                                  akVs4PFclean
                                                  *
                                                  akVs4PFmatch
                                                  *
                                                  akVs4PFparton
                                                  *
                                                  akVs4PFcorr
                                                  *
                                                  akVs4PFJetID
                                                  *
                                                  akVs4PFPatJetFlavourId
                                                  *
                                                  akVs4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs4PFJetBtagging
                                                  *
                                                  akVs4PFpatJetsWithBtagging
                                                  *
                                                  akVs4PFJetAnalyzer
                                                  )

akVs4PFJetSequence_data = cms.Sequence(akVs4PFcorr
                                                    *
                                                    akVs4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs4PFJetBtagging
                                                    *
                                                    akVs4PFpatJetsWithBtagging
                                                    *
                                                    akVs4PFJetAnalyzer
                                                    )

akVs4PFJetSequence_jec = akVs4PFJetSequence_mc
akVs4PFJetSequence_mix = akVs4PFJetSequence_mc

akVs4PFJetSequence = cms.Sequence(akVs4PFJetSequence_data)
