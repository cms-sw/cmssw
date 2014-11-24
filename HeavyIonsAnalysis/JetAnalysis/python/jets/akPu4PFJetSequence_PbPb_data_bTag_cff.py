

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu4PFJets"),
    matched = cms.InputTag("ak4HiGenJetsCleaned")
    )

akPu4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu4PFJets")
                                                        )

akPu4PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu4PFJets"),
    payload = "AKPu4PF_hiIterativeTracks"
    )

akPu4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu4CaloJets'))

akPu4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJetsCleaned'))

akPu4PFbTagger = bTaggers("akPu4PF")

#create objects locally since they dont load properly otherwise
akPu4PFmatch = akPu4PFbTagger.match
akPu4PFparton = akPu4PFbTagger.parton
akPu4PFPatJetFlavourAssociation = akPu4PFbTagger.PatJetFlavourAssociation
akPu4PFJetTracksAssociatorAtVertex = akPu4PFbTagger.JetTracksAssociatorAtVertex
akPu4PFSimpleSecondaryVertexHighEffBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu4PFSimpleSecondaryVertexHighPurBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu4PFCombinedSecondaryVertexBJetTags = akPu4PFbTagger.CombinedSecondaryVertexBJetTags
akPu4PFCombinedSecondaryVertexMVABJetTags = akPu4PFbTagger.CombinedSecondaryVertexMVABJetTags
akPu4PFJetBProbabilityBJetTags = akPu4PFbTagger.JetBProbabilityBJetTags
akPu4PFSoftMuonByPtBJetTags = akPu4PFbTagger.SoftMuonByPtBJetTags
akPu4PFSoftMuonByIP3dBJetTags = akPu4PFbTagger.SoftMuonByIP3dBJetTags
akPu4PFTrackCountingHighEffBJetTags = akPu4PFbTagger.TrackCountingHighEffBJetTags
akPu4PFTrackCountingHighPurBJetTags = akPu4PFbTagger.TrackCountingHighPurBJetTags
akPu4PFPatJetPartonAssociation = akPu4PFbTagger.PatJetPartonAssociation

akPu4PFImpactParameterTagInfos = akPu4PFbTagger.ImpactParameterTagInfos
akPu4PFJetProbabilityBJetTags = akPu4PFbTagger.JetProbabilityBJetTags
akPu4PFPositiveOnlyJetProbabilityJetTags = akPu4PFbTagger.PositiveOnlyJetProbabilityJetTags
akPu4PFNegativeOnlyJetProbabilityJetTags = akPu4PFbTagger.NegativeOnlyJetProbabilityJetTags
akPu4PFNegativeTrackCountingHighEffJetTags = akPu4PFbTagger.NegativeTrackCountingHighEffJetTags
akPu4PFNegativeTrackCountingHighPur = akPu4PFbTagger.NegativeTrackCountingHighPur
akPu4PFNegativeOnlyJetBProbabilityJetTags = akPu4PFbTagger.NegativeOnlyJetBProbabilityJetTags
akPu4PFPositiveOnlyJetBProbabilityJetTags = akPu4PFbTagger.PositiveOnlyJetBProbabilityJetTags

akPu4PFSecondaryVertexTagInfos = akPu4PFbTagger.SecondaryVertexTagInfos
akPu4PFSimpleSecondaryVertexHighEffBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu4PFSimpleSecondaryVertexHighPurBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu4PFCombinedSecondaryVertexBJetTags = akPu4PFbTagger.CombinedSecondaryVertexBJetTags
akPu4PFCombinedSecondaryVertexMVABJetTags = akPu4PFbTagger.CombinedSecondaryVertexMVABJetTags

akPu4PFSecondaryVertexNegativeTagInfos = akPu4PFbTagger.SecondaryVertexNegativeTagInfos
akPu4PFSimpleSecondaryVertexNegativeHighEffBJetTags = akPu4PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu4PFSimpleSecondaryVertexNegativeHighPurBJetTags = akPu4PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu4PFCombinedSecondaryVertexNegativeBJetTags = akPu4PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akPu4PFCombinedSecondaryVertexPositiveBJetTags = akPu4PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akPu4PFSoftMuonTagInfos = akPu4PFbTagger.SoftMuonTagInfos
akPu4PFSoftMuonBJetTags = akPu4PFbTagger.SoftMuonBJetTags
akPu4PFSoftMuonByIP3dBJetTags = akPu4PFbTagger.SoftMuonByIP3dBJetTags
akPu4PFSoftMuonByPtBJetTags = akPu4PFbTagger.SoftMuonByPtBJetTags
akPu4PFNegativeSoftMuonByPtBJetTags = akPu4PFbTagger.NegativeSoftMuonByPtBJetTags
akPu4PFPositiveSoftMuonByPtBJetTags = akPu4PFbTagger.PositiveSoftMuonByPtBJetTags

akPu4PFPatJetFlavourId = cms.Sequence(akPu4PFPatJetPartonAssociation*akPu4PFPatJetFlavourAssociation)

akPu4PFJetBtaggingIP       = cms.Sequence(akPu4PFImpactParameterTagInfos *
            (akPu4PFTrackCountingHighEffBJetTags +
             akPu4PFTrackCountingHighPurBJetTags +
             akPu4PFJetProbabilityBJetTags +
             akPu4PFJetBProbabilityBJetTags +
             akPu4PFPositiveOnlyJetProbabilityJetTags +
             akPu4PFNegativeOnlyJetProbabilityJetTags +
             akPu4PFNegativeTrackCountingHighEffJetTags +
             akPu4PFNegativeTrackCountingHighPur +
             akPu4PFNegativeOnlyJetBProbabilityJetTags +
             akPu4PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu4PFJetBtaggingSV = cms.Sequence(akPu4PFImpactParameterTagInfos
            *
            akPu4PFSecondaryVertexTagInfos
            * (akPu4PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu4PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu4PFCombinedSecondaryVertexBJetTags
                +
                akPu4PFCombinedSecondaryVertexMVABJetTags
              )
            )

akPu4PFJetBtaggingNegSV = cms.Sequence(akPu4PFImpactParameterTagInfos
            *
            akPu4PFSecondaryVertexNegativeTagInfos
            * (akPu4PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu4PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu4PFCombinedSecondaryVertexNegativeBJetTags
                +
                akPu4PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu4PFJetBtaggingMu = cms.Sequence(akPu4PFSoftMuonTagInfos * (akPu4PFSoftMuonBJetTags
                +
                akPu4PFSoftMuonByIP3dBJetTags
                +
                akPu4PFSoftMuonByPtBJetTags
                +
                akPu4PFNegativeSoftMuonByPtBJetTags
                +
                akPu4PFPositiveSoftMuonByPtBJetTags
              )
            )

akPu4PFJetBtagging = cms.Sequence(akPu4PFJetBtaggingIP
            *akPu4PFJetBtaggingSV
            *akPu4PFJetBtaggingNegSV
            *akPu4PFJetBtaggingMu
            )

akPu4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu4PFJets"),
        genJetMatch          = cms.InputTag("akPu4PFmatch"),
        genPartonMatch       = cms.InputTag("akPu4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu4PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu4PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu4PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu4PFJetProbabilityBJetTags"),
            cms.InputTag("akPu4PFSoftMuonByPtBJetTags"),
            cms.InputTag("akPu4PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu4PFJetID"),
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

akPu4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu4PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJetsCleaned',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu4PFJetSequence_mc = cms.Sequence(
                                                  akPu4PFclean
                                                  *
                                                  akPu4PFmatch
                                                  *
                                                  akPu4PFparton
                                                  *
                                                  akPu4PFcorr
                                                  *
                                                  akPu4PFJetID
                                                  *
                                                  akPu4PFPatJetFlavourId
                                                  *
                                                  akPu4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu4PFJetBtagging
                                                  *
                                                  akPu4PFpatJetsWithBtagging
                                                  *
                                                  akPu4PFJetAnalyzer
                                                  )

akPu4PFJetSequence_data = cms.Sequence(akPu4PFcorr
                                                    *
                                                    akPu4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu4PFJetBtagging
                                                    *
                                                    akPu4PFpatJetsWithBtagging
                                                    *
                                                    akPu4PFJetAnalyzer
                                                    )

akPu4PFJetSequence_jec = akPu4PFJetSequence_mc
akPu4PFJetSequence_mix = akPu4PFJetSequence_mc

akPu4PFJetSequence = cms.Sequence(akPu4PFJetSequence_data)
