

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu5PFJets"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

akPu5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu5PFJets")
                                                        )

akPu5PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu5PFJets"),
    payload = "AKPu5PF_hiIterativeTracks"
    )

akPu5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu5CaloJets'))

akPu5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJetsCleaned'))

akPu5PFbTagger = bTaggers("akPu5PF")

#create objects locally since they dont load properly otherwise
akPu5PFmatch = akPu5PFbTagger.match
akPu5PFparton = akPu5PFbTagger.parton
akPu5PFPatJetFlavourAssociation = akPu5PFbTagger.PatJetFlavourAssociation
akPu5PFJetTracksAssociatorAtVertex = akPu5PFbTagger.JetTracksAssociatorAtVertex
akPu5PFSimpleSecondaryVertexHighEffBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu5PFSimpleSecondaryVertexHighPurBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu5PFCombinedSecondaryVertexBJetTags = akPu5PFbTagger.CombinedSecondaryVertexBJetTags
akPu5PFCombinedSecondaryVertexMVABJetTags = akPu5PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu5PFJetBProbabilityBJetTags = akPu5PFbTagger.JetBProbabilityBJetTags
akPu5PFSoftMuonByPtBJetTags = akPu5PFbTagger.SoftMuonByPtBJetTags
akPu5PFSoftMuonByIP3dBJetTags = akPu5PFbTagger.SoftMuonByIP3dBJetTags
akPu5PFTrackCountingHighEffBJetTags = akPu5PFbTagger.TrackCountingHighEffBJetTags
akPu5PFTrackCountingHighPurBJetTags = akPu5PFbTagger.TrackCountingHighPurBJetTags
akPu5PFPatJetPartonAssociation = akPu5PFbTagger.PatJetPartonAssociation

akPu5PFImpactParameterTagInfos = akPu5PFbTagger.ImpactParameterTagInfos
akPu5PFJetProbabilityBJetTags = akPu5PFbTagger.JetProbabilityBJetTags
akPu5PFPositiveOnlyJetProbabilityJetTags = akPu5PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu5PFNegativeOnlyJetProbabilityJetTags = akPu5PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu5PFNegativeTrackCountingHighEffJetTags = akPu5PFbTagger.NegativeTrackCountingHighEffJetTags
akPu5PFNegativeTrackCountingHighPur = akPu5PFbTagger.NegativeTrackCountingHighPur
akPu5PFNegativeOnlyJetBProbabilityJetTags = akPu5PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu5PFPositiveOnlyJetBProbabilityJetTags = akPu5PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu5PFSecondaryVertexTagInfos = akPu5PFbTagger.SecondaryVertexTagInfos
akPu5PFSimpleSecondaryVertexHighEffBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu5PFSimpleSecondaryVertexHighPurBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu5PFCombinedSecondaryVertexBJetTags = akPu5PFbTagger.CombinedSecondaryVertexBJetTags
akPu5PFCombinedSecondaryVertexMVABJetTags = akPu5PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu5PFSecondaryVertexNegativeTagInfos = akPu5PFbTagger.SecondaryVertexNegativeTagInfos
akPu5PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu5PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu5PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu5PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu5PFCombinedSecondaryVertexNegativeBJetTags = akPu5PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu5PFCombinedSecondaryVertexPositiveBJetTags = akPu5PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu5PFSoftMuonTagInfos = akPu5PFbTagger.SoftMuonTagInfos
akPu5PFSoftMuonBJetTags = akPu5PFbTagger.SoftMuonBJetTags
akPu5PFSoftMuonByIP3dBJetTags = akPu5PFbTagger.SoftMuonByIP3dBJetTags
akPu5PFSoftMuonByPtBJetTags = akPu5PFbTagger.SoftMuonByPtBJetTags
akPu5PFNegativeSoftMuonByPtBJetTags = akPu5PFbTagger.NegativeSoftMuonByPtBJetTags
akPu5PFPositiveSoftMuonByPtBJetTags = akPu5PFbTagger.PositiveSoftMuonByPtBJetTags

akPu5PFPatJetFlavourId = cms.Sequence(akPu5PFPatJetPartonAssociation*akPu5PFPatJetFlavourAssociation)

akPu5PFJetBtaggingIP       = cms.Sequence(akPu5PFImpactParameterTagInfos *
            (akPu5PFTrackCountingHighEffBJetTags +
             akPu5PFTrackCountingHighPurBJetTags +
             akPu5PFJetProbabilityBJetTags +
             akPu5PFJetBProbabilityBJetTags +
             akPu5PFPositiveOnlyJetProbabilityJetTags +
             akPu5PFNegativeOnlyJetProbabilityJetTags +
             akPu5PFNegativeTrackCountingHighEffJetTags +
             akPu5PFNegativeTrackCountingHighPur +
             akPu5PFNegativeOnlyJetBProbabilityJetTags +
             akPu5PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu5PFJetBtaggingSV = cms.Sequence(akPu5PFImpactParameterTagInfos
            *
            akPu5PFSecondaryVertexTagInfos
            * (akPu5PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu5PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu5PFCombinedSecondaryVertexBJetTags
                +
                akPu5PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu5PFJetBtaggingNegSV = cms.Sequence(akPu5PFImpactParameterTagInfos
            *
            akPu5PFSecondaryVertexNegativeTagInfos
            * (akPu5PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu5PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu5PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu5PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu5PFJetBtaggingMu = cms.Sequence(akPu5PFSoftMuonTagInfos * (akPu5PFSoftMuonBJetTags
                +
                akPu5PFSoftMuonByIP3dBJetTags
                +
                akPu5PFSoftMuonByPtBJetTags
                +
                akPu5PFNegativeSoftMuonByPtBJetTags
                +
                akPu5PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu5PFJetBtagging = cms.Sequence(akPu5PFJetBtaggingIP
            *akPu5PFJetBtaggingSV
            *akPu5PFJetBtaggingNegSV
            *akPu5PFJetBtaggingMu
            )

akPu5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu5PFJets"),
        genJetMatch          = cms.InputTag("akPu5PFmatch"),
        genPartonMatch       = cms.InputTag("akPu5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu5PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu5PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu5PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu5PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu5PFJetProbabilityBJetTags"),
            cms.InputTag("akPu5PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu5PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu5PFJetID"),
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

akPu5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu5PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akPu5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu5PFJetSequence_mc = cms.Sequence(
                                                  akPu5PFclean
                                                  *
                                                  akPu5PFmatch
                                                  *
                                                  akPu5PFparton
                                                  *
                                                  akPu5PFcorr
                                                  *
                                                  akPu5PFJetID
                                                  *
                                                  akPu5PFPatJetFlavourId
                                                  *
                                                  akPu5PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu5PFJetBtagging
                                                  *
                                                  akPu5PFpatJetsWithBtagging
                                                  *
                                                  akPu5PFJetAnalyzer
                                                  )

akPu5PFJetSequence_data = cms.Sequence(akPu5PFcorr
                                                    *
                                                    akPu5PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu5PFJetBtagging
                                                    *
                                                    akPu5PFpatJetsWithBtagging
                                                    *
                                                    akPu5PFJetAnalyzer
                                                    )

akPu5PFJetSequence_jec = akPu5PFJetSequence_mc
akPu5PFJetSequence_mix = akPu5PFJetSequence_mc

akPu5PFJetSequence = cms.Sequence(akPu5PFJetSequence_jec)
akPu5PFJetAnalyzer.genPtMin = cms.untracked.double(1)
