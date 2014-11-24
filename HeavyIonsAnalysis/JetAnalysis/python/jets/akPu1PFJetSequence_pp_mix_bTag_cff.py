

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu1PFJets"),
    matched = cms.InputTag("ak1HiGenJets")
    )

akPu1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu1PFJets")
                                                        )

akPu1PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu1PFJets"),
    payload = "AKPu1PF_generalTracks"
    )

akPu1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu1CaloJets'))

akPu1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJets'))

akPu1PFbTagger = bTaggers("akPu1PF")

#create objects locally since they dont load properly otherwise
akPu1PFmatch = akPu1PFbTagger.match
akPu1PFparton = akPu1PFbTagger.parton
akPu1PFPatJetFlavourAssociation = akPu1PFbTagger.PatJetFlavourAssociation
akPu1PFJetTracksAssociatorAtVertex = akPu1PFbTagger.JetTracksAssociatorAtVertex
akPu1PFSimpleSecondaryVertexHighEffBJetTags = akPu1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu1PFSimpleSecondaryVertexHighPurBJetTags = akPu1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu1PFCombinedSecondaryVertexBJetTags = akPu1PFbTagger.CombinedSecondaryVertexBJetTags
akPu1PFCombinedSecondaryVertexMVABJetTags = akPu1PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu1PFJetBProbabilityBJetTags = akPu1PFbTagger.JetBProbabilityBJetTags
akPu1PFSoftMuonByPtBJetTags = akPu1PFbTagger.SoftMuonByPtBJetTags
akPu1PFSoftMuonByIP3dBJetTags = akPu1PFbTagger.SoftMuonByIP3dBJetTags
akPu1PFTrackCountingHighEffBJetTags = akPu1PFbTagger.TrackCountingHighEffBJetTags
akPu1PFTrackCountingHighPurBJetTags = akPu1PFbTagger.TrackCountingHighPurBJetTags
akPu1PFPatJetPartonAssociation = akPu1PFbTagger.PatJetPartonAssociation

akPu1PFImpactParameterTagInfos = akPu1PFbTagger.ImpactParameterTagInfos
akPu1PFJetProbabilityBJetTags = akPu1PFbTagger.JetProbabilityBJetTags
akPu1PFPositiveOnlyJetProbabilityJetTags = akPu1PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu1PFNegativeOnlyJetProbabilityJetTags = akPu1PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu1PFNegativeTrackCountingHighEffJetTags = akPu1PFbTagger.NegativeTrackCountingHighEffJetTags
akPu1PFNegativeTrackCountingHighPur = akPu1PFbTagger.NegativeTrackCountingHighPur
akPu1PFNegativeOnlyJetBProbabilityJetTags = akPu1PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu1PFPositiveOnlyJetBProbabilityJetTags = akPu1PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu1PFSecondaryVertexTagInfos = akPu1PFbTagger.SecondaryVertexTagInfos
akPu1PFSimpleSecondaryVertexHighEffBJetTags = akPu1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu1PFSimpleSecondaryVertexHighPurBJetTags = akPu1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu1PFCombinedSecondaryVertexBJetTags = akPu1PFbTagger.CombinedSecondaryVertexBJetTags
akPu1PFCombinedSecondaryVertexMVABJetTags = akPu1PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu1PFSecondaryVertexNegativeTagInfos = akPu1PFbTagger.SecondaryVertexNegativeTagInfos
akPu1PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu1PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu1PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu1PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu1PFCombinedSecondaryVertexNegativeBJetTags = akPu1PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu1PFCombinedSecondaryVertexPositiveBJetTags = akPu1PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu1PFSoftMuonTagInfos = akPu1PFbTagger.SoftMuonTagInfos
akPu1PFSoftMuonBJetTags = akPu1PFbTagger.SoftMuonBJetTags
akPu1PFSoftMuonByIP3dBJetTags = akPu1PFbTagger.SoftMuonByIP3dBJetTags
akPu1PFSoftMuonByPtBJetTags = akPu1PFbTagger.SoftMuonByPtBJetTags
akPu1PFNegativeSoftMuonByPtBJetTags = akPu1PFbTagger.NegativeSoftMuonByPtBJetTags
akPu1PFPositiveSoftMuonByPtBJetTags = akPu1PFbTagger.PositiveSoftMuonByPtBJetTags

akPu1PFPatJetFlavourId = cms.Sequence(akPu1PFPatJetPartonAssociation*akPu1PFPatJetFlavourAssociation)

akPu1PFJetBtaggingIP       = cms.Sequence(akPu1PFImpactParameterTagInfos *
            (akPu1PFTrackCountingHighEffBJetTags +
             akPu1PFTrackCountingHighPurBJetTags +
             akPu1PFJetProbabilityBJetTags +
             akPu1PFJetBProbabilityBJetTags +
             akPu1PFPositiveOnlyJetProbabilityJetTags +
             akPu1PFNegativeOnlyJetProbabilityJetTags +
             akPu1PFNegativeTrackCountingHighEffJetTags +
             akPu1PFNegativeTrackCountingHighPur +
             akPu1PFNegativeOnlyJetBProbabilityJetTags +
             akPu1PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu1PFJetBtaggingSV = cms.Sequence(akPu1PFImpactParameterTagInfos
            *
            akPu1PFSecondaryVertexTagInfos
            * (akPu1PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu1PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu1PFCombinedSecondaryVertexBJetTags
                +
                akPu1PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu1PFJetBtaggingNegSV = cms.Sequence(akPu1PFImpactParameterTagInfos
            *
            akPu1PFSecondaryVertexNegativeTagInfos
            * (akPu1PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu1PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu1PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu1PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu1PFJetBtaggingMu = cms.Sequence(akPu1PFSoftMuonTagInfos * (akPu1PFSoftMuonBJetTags
                +
                akPu1PFSoftMuonByIP3dBJetTags
                +
                akPu1PFSoftMuonByPtBJetTags
                +
                akPu1PFNegativeSoftMuonByPtBJetTags
                +
                akPu1PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu1PFJetBtagging = cms.Sequence(akPu1PFJetBtaggingIP
            *akPu1PFJetBtaggingSV
            *akPu1PFJetBtaggingNegSV
            *akPu1PFJetBtaggingMu
            )

akPu1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu1PFJets"),
        genJetMatch          = cms.InputTag("akPu1PFmatch"),
        genPartonMatch       = cms.InputTag("akPu1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu1PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu1PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu1PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu1PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu1PFJetProbabilityBJetTags"),
            cms.InputTag("akPu1PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu1PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu1PFJetID"),
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

akPu1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu1PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu1PFJetSequence_mc = cms.Sequence(
                                                  akPu1PFclean
                                                  *
                                                  akPu1PFmatch
                                                  *
                                                  akPu1PFparton
                                                  *
                                                  akPu1PFcorr
                                                  *
                                                  akPu1PFJetID
                                                  *
                                                  akPu1PFPatJetFlavourId
                                                  *
                                                  akPu1PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu1PFJetBtagging
                                                  *
                                                  akPu1PFpatJetsWithBtagging
                                                  *
                                                  akPu1PFJetAnalyzer
                                                  )

akPu1PFJetSequence_data = cms.Sequence(akPu1PFcorr
                                                    *
                                                    akPu1PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu1PFJetBtagging
                                                    *
                                                    akPu1PFpatJetsWithBtagging
                                                    *
                                                    akPu1PFJetAnalyzer
                                                    )

akPu1PFJetSequence_jec = akPu1PFJetSequence_mc
akPu1PFJetSequence_mix = akPu1PFJetSequence_mc

akPu1PFJetSequence = cms.Sequence(akPu1PFJetSequence_mix)
