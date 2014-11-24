

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs1PFJets"),
    matched = cms.InputTag("ak1HiGenJetsCleaned")
    )

akVs1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs1PFJets")
                                                        )

akVs1PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs1PFJets"),
    payload = "AKVs1PF_hiIterativeTracks"
    )

akVs1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs1CaloJets'))

akVs1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJetsCleaned'))

akVs1PFbTagger = bTaggers("akVs1PF")

#create objects locally since they dont load properly otherwise
akVs1PFmatch = akVs1PFbTagger.match
akVs1PFparton = akVs1PFbTagger.parton
akVs1PFPatJetFlavourAssociation = akVs1PFbTagger.PatJetFlavourAssociation
akVs1PFJetTracksAssociatorAtVertex = akVs1PFbTagger.JetTracksAssociatorAtVertex
akVs1PFSimpleSecondaryVertexHighEffBJetTags = akVs1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs1PFSimpleSecondaryVertexHighPurBJetTags = akVs1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs1PFCombinedSecondaryVertexBJetTags = akVs1PFbTagger.CombinedSecondaryVertexBJetTags
akVs1PFCombinedSecondaryVertexMVABJetTags = akVs1PFbTagger.CombinedSecondaryVertexMVABJetTags
akVs1PFJetBProbabilityBJetTags = akVs1PFbTagger.JetBProbabilityBJetTags
akVs1PFSoftMuonByPtBJetTags = akVs1PFbTagger.SoftMuonByPtBJetTags
akVs1PFSoftMuonByIP3dBJetTags = akVs1PFbTagger.SoftMuonByIP3dBJetTags
akVs1PFTrackCountingHighEffBJetTags = akVs1PFbTagger.TrackCountingHighEffBJetTags
akVs1PFTrackCountingHighPurBJetTags = akVs1PFbTagger.TrackCountingHighPurBJetTags
akVs1PFPatJetPartonAssociation = akVs1PFbTagger.PatJetPartonAssociation

akVs1PFImpactParameterTagInfos = akVs1PFbTagger.ImpactParameterTagInfos
akVs1PFJetProbabilityBJetTags = akVs1PFbTagger.JetProbabilityBJetTags
akVs1PFPositiveOnlyJetProbabilityJetTags = akVs1PFbTagger.PositiveOnlyJetProbabilityJetTags
akVs1PFNegativeOnlyJetProbabilityJetTags = akVs1PFbTagger.NegativeOnlyJetProbabilityJetTags
akVs1PFNegativeTrackCountingHighEffJetTags = akVs1PFbTagger.NegativeTrackCountingHighEffJetTags
akVs1PFNegativeTrackCountingHighPur = akVs1PFbTagger.NegativeTrackCountingHighPur
akVs1PFNegativeOnlyJetBProbabilityJetTags = akVs1PFbTagger.NegativeOnlyJetBProbabilityJetTags
akVs1PFPositiveOnlyJetBProbabilityJetTags = akVs1PFbTagger.PositiveOnlyJetBProbabilityJetTags

akVs1PFSecondaryVertexTagInfos = akVs1PFbTagger.SecondaryVertexTagInfos
akVs1PFSimpleSecondaryVertexHighEffBJetTags = akVs1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs1PFSimpleSecondaryVertexHighPurBJetTags = akVs1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs1PFCombinedSecondaryVertexBJetTags = akVs1PFbTagger.CombinedSecondaryVertexBJetTags
akVs1PFCombinedSecondaryVertexMVABJetTags = akVs1PFbTagger.CombinedSecondaryVertexMVABJetTags

akVs1PFSecondaryVertexNegativeTagInfos = akVs1PFbTagger.SecondaryVertexNegativeTagInfos
akVs1PFSimpleSecondaryVertexNegativeHighEffBJetTags = akVs1PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs1PFSimpleSecondaryVertexNegativeHighPurBJetTags = akVs1PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs1PFCombinedSecondaryVertexNegativeBJetTags = akVs1PFbTagger.CombinedSecondaryVertexNegativeBJetTags
akVs1PFCombinedSecondaryVertexPositiveBJetTags = akVs1PFbTagger.CombinedSecondaryVertexPositiveBJetTags

akVs1PFSoftMuonTagInfos = akVs1PFbTagger.SoftMuonTagInfos
akVs1PFSoftMuonBJetTags = akVs1PFbTagger.SoftMuonBJetTags
akVs1PFSoftMuonByIP3dBJetTags = akVs1PFbTagger.SoftMuonByIP3dBJetTags
akVs1PFSoftMuonByPtBJetTags = akVs1PFbTagger.SoftMuonByPtBJetTags
akVs1PFNegativeSoftMuonByPtBJetTags = akVs1PFbTagger.NegativeSoftMuonByPtBJetTags
akVs1PFPositiveSoftMuonByPtBJetTags = akVs1PFbTagger.PositiveSoftMuonByPtBJetTags

akVs1PFPatJetFlavourId = cms.Sequence(akVs1PFPatJetPartonAssociation*akVs1PFPatJetFlavourAssociation)

akVs1PFJetBtaggingIP       = cms.Sequence(akVs1PFImpactParameterTagInfos *
            (akVs1PFTrackCountingHighEffBJetTags +
             akVs1PFTrackCountingHighPurBJetTags +
             akVs1PFJetProbabilityBJetTags +
             akVs1PFJetBProbabilityBJetTags +
             akVs1PFPositiveOnlyJetProbabilityJetTags +
             akVs1PFNegativeOnlyJetProbabilityJetTags +
             akVs1PFNegativeTrackCountingHighEffJetTags +
             akVs1PFNegativeTrackCountingHighPur +
             akVs1PFNegativeOnlyJetBProbabilityJetTags +
             akVs1PFPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs1PFJetBtaggingSV = cms.Sequence(akVs1PFImpactParameterTagInfos
            *
            akVs1PFSecondaryVertexTagInfos
            * (akVs1PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs1PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs1PFCombinedSecondaryVertexBJetTags
                +
                akVs1PFCombinedSecondaryVertexMVABJetTags
              )
            )

akVs1PFJetBtaggingNegSV = cms.Sequence(akVs1PFImpactParameterTagInfos
            *
            akVs1PFSecondaryVertexNegativeTagInfos
            * (akVs1PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs1PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs1PFCombinedSecondaryVertexNegativeBJetTags
                +
                akVs1PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs1PFJetBtaggingMu = cms.Sequence(akVs1PFSoftMuonTagInfos * (akVs1PFSoftMuonBJetTags
                +
                akVs1PFSoftMuonByIP3dBJetTags
                +
                akVs1PFSoftMuonByPtBJetTags
                +
                akVs1PFNegativeSoftMuonByPtBJetTags
                +
                akVs1PFPositiveSoftMuonByPtBJetTags
              )
            )

akVs1PFJetBtagging = cms.Sequence(akVs1PFJetBtaggingIP
            *akVs1PFJetBtaggingSV
            *akVs1PFJetBtaggingNegSV
            *akVs1PFJetBtaggingMu
            )

akVs1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs1PFJets"),
        genJetMatch          = cms.InputTag("akVs1PFmatch"),
        genPartonMatch       = cms.InputTag("akVs1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs1PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs1PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs1PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs1PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs1PFJetProbabilityBJetTags"),
            cms.InputTag("akVs1PFSoftMuonByPtBJetTags"),
            cms.InputTag("akVs1PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs1PFJetID"),
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

akVs1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs1PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJetsCleaned',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akVs1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs1PFJetSequence_mc = cms.Sequence(
                                                  akVs1PFclean
                                                  *
                                                  akVs1PFmatch
                                                  *
                                                  akVs1PFparton
                                                  *
                                                  akVs1PFcorr
                                                  *
                                                  akVs1PFJetID
                                                  *
                                                  akVs1PFPatJetFlavourId
                                                  *
                                                  akVs1PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs1PFJetBtagging
                                                  *
                                                  akVs1PFpatJetsWithBtagging
                                                  *
                                                  akVs1PFJetAnalyzer
                                                  )

akVs1PFJetSequence_data = cms.Sequence(akVs1PFcorr
                                                    *
                                                    akVs1PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs1PFJetBtagging
                                                    *
                                                    akVs1PFpatJetsWithBtagging
                                                    *
                                                    akVs1PFJetAnalyzer
                                                    )

akVs1PFJetSequence_jec = akVs1PFJetSequence_mc
akVs1PFJetSequence_mix = akVs1PFJetSequence_mc

akVs1PFJetSequence = cms.Sequence(akVs1PFJetSequence_data)
