

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu6PFJets"),
    matched = cms.InputTag("ak6HiGenJets")
    )

akPu6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6PFJets")
                                                        )

akPu6PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu6PFJets"),
    payload = "AKPu6PF_generalTracks"
    )

akPu6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu6CaloJets'))

akPu6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

akPu6PFbTagger = bTaggers("akPu6PF")

#create objects locally since they dont load properly otherwise
akPu6PFmatch = akPu6PFbTagger.match
akPu6PFparton = akPu6PFbTagger.parton
akPu6PFPatJetFlavourAssociation = akPu6PFbTagger.PatJetFlavourAssociation
akPu6PFJetTracksAssociatorAtVertex = akPu6PFbTagger.JetTracksAssociatorAtVertex
akPu6PFSimpleSecondaryVertexHighEffBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6PFSimpleSecondaryVertexHighPurBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6PFCombinedSecondaryVertexBJetTags = akPu6PFbTagger.CombinedSecondaryVertexBJetTags
akPu6PFCombinedSecondaryVertexMVABJetTags = akPu6PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu6PFJetBProbabilityBJetTags = akPu6PFbTagger.JetBProbabilityBJetTags
akPu6PFSoftMuonByPtBJetTags = akPu6PFbTagger.SoftMuonByPtBJetTags
akPu6PFSoftMuonByIP3dBJetTags = akPu6PFbTagger.SoftMuonByIP3dBJetTags
akPu6PFTrackCountingHighEffBJetTags = akPu6PFbTagger.TrackCountingHighEffBJetTags
akPu6PFTrackCountingHighPurBJetTags = akPu6PFbTagger.TrackCountingHighPurBJetTags
akPu6PFPatJetPartonAssociation = akPu6PFbTagger.PatJetPartonAssociation

akPu6PFImpactParameterTagInfos = akPu6PFbTagger.ImpactParameterTagInfos
akPu6PFJetProbabilityBJetTags = akPu6PFbTagger.JetProbabilityBJetTags
akPu6PFPositiveOnlyJetProbabilityJetTags = akPu6PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu6PFNegativeOnlyJetProbabilityJetTags = akPu6PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu6PFNegativeTrackCountingHighEffJetTags = akPu6PFbTagger.NegativeTrackCountingHighEffJetTags
akPu6PFNegativeTrackCountingHighPur = akPu6PFbTagger.NegativeTrackCountingHighPur
akPu6PFNegativeOnlyJetBProbabilityJetTags = akPu6PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu6PFPositiveOnlyJetBProbabilityJetTags = akPu6PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu6PFSecondaryVertexTagInfos = akPu6PFbTagger.SecondaryVertexTagInfos
akPu6PFSimpleSecondaryVertexHighEffBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6PFSimpleSecondaryVertexHighPurBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6PFCombinedSecondaryVertexBJetTags = akPu6PFbTagger.CombinedSecondaryVertexBJetTags
akPu6PFCombinedSecondaryVertexMVABJetTags = akPu6PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu6PFSecondaryVertexNegativeTagInfos = akPu6PFbTagger.SecondaryVertexNegativeTagInfos
akPu6PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu6PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu6PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu6PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu6PFCombinedSecondaryVertexNegativeBJetTags = akPu6PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu6PFCombinedSecondaryVertexPositiveBJetTags = akPu6PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu6PFSoftMuonTagInfos = akPu6PFbTagger.SoftMuonTagInfos
akPu6PFSoftMuonBJetTags = akPu6PFbTagger.SoftMuonBJetTags
akPu6PFSoftMuonByIP3dBJetTags = akPu6PFbTagger.SoftMuonByIP3dBJetTags
akPu6PFSoftMuonByPtBJetTags = akPu6PFbTagger.SoftMuonByPtBJetTags
akPu6PFNegativeSoftMuonByPtBJetTags = akPu6PFbTagger.NegativeSoftMuonByPtBJetTags
akPu6PFPositiveSoftMuonByPtBJetTags = akPu6PFbTagger.PositiveSoftMuonByPtBJetTags

akPu6PFPatJetFlavourId = cms.Sequence(akPu6PFPatJetPartonAssociation*akPu6PFPatJetFlavourAssociation)

akPu6PFJetBtaggingIP       = cms.Sequence(akPu6PFImpactParameterTagInfos *
            (akPu6PFTrackCountingHighEffBJetTags +
             akPu6PFTrackCountingHighPurBJetTags +
             akPu6PFJetProbabilityBJetTags +
             akPu6PFJetBProbabilityBJetTags +
             akPu6PFPositiveOnlyJetProbabilityJetTags +
             akPu6PFNegativeOnlyJetProbabilityJetTags +
             akPu6PFNegativeTrackCountingHighEffJetTags +
             akPu6PFNegativeTrackCountingHighPur +
             akPu6PFNegativeOnlyJetBProbabilityJetTags +
             akPu6PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu6PFJetBtaggingSV = cms.Sequence(akPu6PFImpactParameterTagInfos
            *
            akPu6PFSecondaryVertexTagInfos
            * (akPu6PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu6PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu6PFCombinedSecondaryVertexBJetTags
                +
                akPu6PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu6PFJetBtaggingNegSV = cms.Sequence(akPu6PFImpactParameterTagInfos
            *
            akPu6PFSecondaryVertexNegativeTagInfos
            * (akPu6PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu6PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu6PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu6PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu6PFJetBtaggingMu = cms.Sequence(akPu6PFSoftMuonTagInfos * (akPu6PFSoftMuonBJetTags
                +
                akPu6PFSoftMuonByIP3dBJetTags
                +
                akPu6PFSoftMuonByPtBJetTags
                +
                akPu6PFNegativeSoftMuonByPtBJetTags
                +
                akPu6PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu6PFJetBtagging = cms.Sequence(akPu6PFJetBtaggingIP
            *akPu6PFJetBtaggingSV
            *akPu6PFJetBtaggingNegSV
            *akPu6PFJetBtaggingMu
            )

akPu6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu6PFJets"),
        genJetMatch          = cms.InputTag("akPu6PFmatch"),
        genPartonMatch       = cms.InputTag("akPu6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu6PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu6PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu6PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu6PFJetProbabilityBJetTags"),
            cms.InputTag("akPu6PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu6PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu6PFJetID"),
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

akPu6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu6PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
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
                                                             bTagJetName = cms.untracked.string("akPu6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu6PFJetSequence_mc = cms.Sequence(
                                                  akPu6PFclean
                                                  *
                                                  akPu6PFmatch
                                                  *
                                                  akPu6PFparton
                                                  *
                                                  akPu6PFcorr
                                                  *
                                                  akPu6PFJetID
                                                  *
                                                  akPu6PFPatJetFlavourId
                                                  *
                                                  akPu6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu6PFJetBtagging
                                                  *
                                                  akPu6PFpatJetsWithBtagging
                                                  *
                                                  akPu6PFJetAnalyzer
                                                  )

akPu6PFJetSequence_data = cms.Sequence(akPu6PFcorr
                                                    *
                                                    akPu6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu6PFJetBtagging
                                                    *
                                                    akPu6PFpatJetsWithBtagging
                                                    *
                                                    akPu6PFJetAnalyzer
                                                    )

akPu6PFJetSequence_jec = akPu6PFJetSequence_mc
akPu6PFJetSequence_mix = akPu6PFJetSequence_mc

akPu6PFJetSequence = cms.Sequence(akPu6PFJetSequence_mix)
