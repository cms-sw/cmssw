

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu7PFJets"),
    matched = cms.InputTag("ak7HiGenJetsCleaned")
    )

akPu7PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu7PFJets")
                                                        )

akPu7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu7PFJets"),
    payload = "AKPu7PF_hiIterativeTracks"
    )

akPu7PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu7CaloJets'))

akPu7PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJetsCleaned'))

akPu7PFbTagger = bTaggers("akPu7PF")

#create objects locally since they dont load properly otherwise
akPu7PFmatch = akPu7PFbTagger.match
akPu7PFparton = akPu7PFbTagger.parton
akPu7PFPatJetFlavourAssociation = akPu7PFbTagger.PatJetFlavourAssociation
akPu7PFJetTracksAssociatorAtVertex = akPu7PFbTagger.JetTracksAssociatorAtVertex
akPu7PFSimpleSecondaryVertexHighEffBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7PFSimpleSecondaryVertexHighPurBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7PFCombinedSecondaryVertexBJetTags = akPu7PFbTagger.CombinedSecondaryVertexBJetTags
akPu7PFCombinedSecondaryVertexMVABJetTags = akPu7PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu7PFJetBProbabilityBJetTags = akPu7PFbTagger.JetBProbabilityBJetTags
akPu7PFSoftMuonByPtBJetTags = akPu7PFbTagger.SoftMuonByPtBJetTags
akPu7PFSoftMuonByIP3dBJetTags = akPu7PFbTagger.SoftMuonByIP3dBJetTags
akPu7PFTrackCountingHighEffBJetTags = akPu7PFbTagger.TrackCountingHighEffBJetTags
akPu7PFTrackCountingHighPurBJetTags = akPu7PFbTagger.TrackCountingHighPurBJetTags
akPu7PFPatJetPartonAssociation = akPu7PFbTagger.PatJetPartonAssociation

akPu7PFImpactParameterTagInfos = akPu7PFbTagger.ImpactParameterTagInfos
akPu7PFJetProbabilityBJetTags = akPu7PFbTagger.JetProbabilityBJetTags
akPu7PFPositiveOnlyJetProbabilityJetTags = akPu7PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu7PFNegativeOnlyJetProbabilityJetTags = akPu7PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu7PFNegativeTrackCountingHighEffJetTags = akPu7PFbTagger.NegativeTrackCountingHighEffJetTags
akPu7PFNegativeTrackCountingHighPur = akPu7PFbTagger.NegativeTrackCountingHighPur
akPu7PFNegativeOnlyJetBProbabilityJetTags = akPu7PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu7PFPositiveOnlyJetBProbabilityJetTags = akPu7PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu7PFSecondaryVertexTagInfos = akPu7PFbTagger.SecondaryVertexTagInfos
akPu7PFSimpleSecondaryVertexHighEffBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7PFSimpleSecondaryVertexHighPurBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7PFCombinedSecondaryVertexBJetTags = akPu7PFbTagger.CombinedSecondaryVertexBJetTags
akPu7PFCombinedSecondaryVertexMVABJetTags = akPu7PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu7PFSecondaryVertexNegativeTagInfos = akPu7PFbTagger.SecondaryVertexNegativeTagInfos
akPu7PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu7PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu7PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu7PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu7PFCombinedSecondaryVertexNegativeBJetTags = akPu7PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu7PFCombinedSecondaryVertexPositiveBJetTags = akPu7PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu7PFSoftMuonTagInfos = akPu7PFbTagger.SoftMuonTagInfos
akPu7PFSoftMuonBJetTags = akPu7PFbTagger.SoftMuonBJetTags
akPu7PFSoftMuonByIP3dBJetTags = akPu7PFbTagger.SoftMuonByIP3dBJetTags
akPu7PFSoftMuonByPtBJetTags = akPu7PFbTagger.SoftMuonByPtBJetTags
akPu7PFNegativeSoftMuonByPtBJetTags = akPu7PFbTagger.NegativeSoftMuonByPtBJetTags
akPu7PFPositiveSoftMuonByPtBJetTags = akPu7PFbTagger.PositiveSoftMuonByPtBJetTags

akPu7PFPatJetFlavourId = cms.Sequence(akPu7PFPatJetPartonAssociation*akPu7PFPatJetFlavourAssociation)

akPu7PFJetBtaggingIP       = cms.Sequence(akPu7PFImpactParameterTagInfos *
            (akPu7PFTrackCountingHighEffBJetTags +
             akPu7PFTrackCountingHighPurBJetTags +
             akPu7PFJetProbabilityBJetTags +
             akPu7PFJetBProbabilityBJetTags +
             akPu7PFPositiveOnlyJetProbabilityJetTags +
             akPu7PFNegativeOnlyJetProbabilityJetTags +
             akPu7PFNegativeTrackCountingHighEffJetTags +
             akPu7PFNegativeTrackCountingHighPur +
             akPu7PFNegativeOnlyJetBProbabilityJetTags +
             akPu7PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu7PFJetBtaggingSV = cms.Sequence(akPu7PFImpactParameterTagInfos
            *
            akPu7PFSecondaryVertexTagInfos
            * (akPu7PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu7PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu7PFCombinedSecondaryVertexBJetTags
                +
                akPu7PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu7PFJetBtaggingNegSV = cms.Sequence(akPu7PFImpactParameterTagInfos
            *
            akPu7PFSecondaryVertexNegativeTagInfos
            * (akPu7PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu7PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu7PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu7PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu7PFJetBtaggingMu = cms.Sequence(akPu7PFSoftMuonTagInfos * (akPu7PFSoftMuonBJetTags
                +
                akPu7PFSoftMuonByIP3dBJetTags
                +
                akPu7PFSoftMuonByPtBJetTags
                +
                akPu7PFNegativeSoftMuonByPtBJetTags
                +
                akPu7PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu7PFJetBtagging = cms.Sequence(akPu7PFJetBtaggingIP
            *akPu7PFJetBtaggingSV
            *akPu7PFJetBtaggingNegSV
            *akPu7PFJetBtaggingMu
            )

akPu7PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu7PFJets"),
        genJetMatch          = cms.InputTag("akPu7PFmatch"),
        genPartonMatch       = cms.InputTag("akPu7PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu7PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu7PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu7PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu7PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu7PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu7PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu7PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu7PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu7PFJetProbabilityBJetTags"),
            cms.InputTag("akPu7PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu7PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu7PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu7PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu7PFJetID"),
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

akPu7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu7PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJetsCleaned',
                                                             rParam = 0.7,
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
                                                             bTagJetName = cms.untracked.string("akPu7PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu7PFJetSequence_mc = cms.Sequence(
                                                  akPu7PFclean
                                                  *
                                                  akPu7PFmatch
                                                  *
                                                  akPu7PFparton
                                                  *
                                                  akPu7PFcorr
                                                  *
                                                  akPu7PFJetID
                                                  *
                                                  akPu7PFPatJetFlavourId
                                                  *
                                                  akPu7PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu7PFJetBtagging
                                                  *
                                                  akPu7PFpatJetsWithBtagging
                                                  *
                                                  akPu7PFJetAnalyzer
                                                  )

akPu7PFJetSequence_data = cms.Sequence(akPu7PFcorr
                                                    *
                                                    akPu7PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu7PFJetBtagging
                                                    *
                                                    akPu7PFpatJetsWithBtagging
                                                    *
                                                    akPu7PFJetAnalyzer
                                                    )

akPu7PFJetSequence_jec = akPu7PFJetSequence_mc
akPu7PFJetSequence_mix = akPu7PFJetSequence_mc

akPu7PFJetSequence = cms.Sequence(akPu7PFJetSequence_mix)
