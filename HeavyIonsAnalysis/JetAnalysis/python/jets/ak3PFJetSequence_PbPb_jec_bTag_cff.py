

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak3PFJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

ak3PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak3PFJets")
                                                        )

ak3PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak3PFJets"),
    payload = "AK3PF_hiIterativeTracks"
    )

ak3PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak3CaloJets'))

ak3PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJetsCleaned'))

ak3PFbTagger = bTaggers("ak3PF")

#create objects locally since they dont load properly otherwise
ak3PFmatch = ak3PFbTagger.match
ak3PFparton = ak3PFbTagger.parton
ak3PFPatJetFlavourAssociation = ak3PFbTagger.PatJetFlavourAssociation
ak3PFJetTracksAssociatorAtVertex = ak3PFbTagger.JetTracksAssociatorAtVertex
ak3PFSimpleSecondaryVertexHighEffBJetTags = ak3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak3PFSimpleSecondaryVertexHighPurBJetTags = ak3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak3PFCombinedSecondaryVertexBJetTags = ak3PFbTagger.CombinedSecondaryVertexBJetTags
ak3PFCombinedSecondaryVertexMVABJetTags = ak3PFbTagger.CombinedSecondaryVertexMVABJetTags
ak3PFJetBProbabilityBJetTags = ak3PFbTagger.JetBProbabilityBJetTags
ak3PFSoftMuonByPtBJetTags = ak3PFbTagger.SoftMuonByPtBJetTags
ak3PFSoftMuonByIP3dBJetTags = ak3PFbTagger.SoftMuonByIP3dBJetTags
ak3PFTrackCountingHighEffBJetTags = ak3PFbTagger.TrackCountingHighEffBJetTags
ak3PFTrackCountingHighPurBJetTags = ak3PFbTagger.TrackCountingHighPurBJetTags
ak3PFPatJetPartonAssociation = ak3PFbTagger.PatJetPartonAssociation

ak3PFImpactParameterTagInfos = ak3PFbTagger.ImpactParameterTagInfos
ak3PFJetProbabilityBJetTags = ak3PFbTagger.JetProbabilityBJetTags
ak3PFPositiveOnlyJetProbabilityJetTags = ak3PFbTagger.PositiveOnlyJetProbabilityJetTags
ak3PFNegativeOnlyJetProbabilityJetTags = ak3PFbTagger.NegativeOnlyJetProbabilityJetTags
ak3PFNegativeTrackCountingHighEffJetTags = ak3PFbTagger.NegativeTrackCountingHighEffJetTags
ak3PFNegativeTrackCountingHighPur = ak3PFbTagger.NegativeTrackCountingHighPur
ak3PFNegativeOnlyJetBProbabilityJetTags = ak3PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak3PFPositiveOnlyJetBProbabilityJetTags = ak3PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak3PFSecondaryVertexTagInfos = ak3PFbTagger.SecondaryVertexTagInfos
ak3PFSimpleSecondaryVertexHighEffBJetTags = ak3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak3PFSimpleSecondaryVertexHighPurBJetTags = ak3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak3PFCombinedSecondaryVertexBJetTags = ak3PFbTagger.CombinedSecondaryVertexBJetTags
ak3PFCombinedSecondaryVertexMVABJetTags = ak3PFbTagger.CombinedSecondaryVertexMVABJetTags

ak3PFSecondaryVertexNegativeTagInfos = ak3PFbTagger.SecondaryVertexNegativeTagInfos
ak3PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak3PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak3PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak3PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak3PFCombinedSecondaryVertexNegativeBJetTags = ak3PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak3PFCombinedSecondaryVertexPositiveBJetTags = ak3PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak3PFSoftMuonTagInfos = ak3PFbTagger.SoftMuonTagInfos
ak3PFSoftMuonBJetTags = ak3PFbTagger.SoftMuonBJetTags
ak3PFSoftMuonByIP3dBJetTags = ak3PFbTagger.SoftMuonByIP3dBJetTags
ak3PFSoftMuonByPtBJetTags = ak3PFbTagger.SoftMuonByPtBJetTags
ak3PFNegativeSoftMuonByPtBJetTags = ak3PFbTagger.NegativeSoftMuonByPtBJetTags
ak3PFPositiveSoftMuonByPtBJetTags = ak3PFbTagger.PositiveSoftMuonByPtBJetTags

ak3PFPatJetFlavourId = cms.Sequence(ak3PFPatJetPartonAssociation*ak3PFPatJetFlavourAssociation)

ak3PFJetBtaggingIP       = cms.Sequence(ak3PFImpactParameterTagInfos *
            (ak3PFTrackCountingHighEffBJetTags +
             ak3PFTrackCountingHighPurBJetTags +
             ak3PFJetProbabilityBJetTags +
             ak3PFJetBProbabilityBJetTags +
             ak3PFPositiveOnlyJetProbabilityJetTags +
             ak3PFNegativeOnlyJetProbabilityJetTags +
             ak3PFNegativeTrackCountingHighEffJetTags +
             ak3PFNegativeTrackCountingHighPur +
             ak3PFNegativeOnlyJetBProbabilityJetTags +
             ak3PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak3PFJetBtaggingSV = cms.Sequence(ak3PFImpactParameterTagInfos
            *
            ak3PFSecondaryVertexTagInfos
            * (ak3PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak3PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak3PFCombinedSecondaryVertexBJetTags
                +
                ak3PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak3PFJetBtaggingNegSV = cms.Sequence(ak3PFImpactParameterTagInfos
            *
            ak3PFSecondaryVertexNegativeTagInfos
            * (ak3PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak3PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak3PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak3PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak3PFJetBtaggingMu = cms.Sequence(ak3PFSoftMuonTagInfos * (ak3PFSoftMuonBJetTags
                +
                ak3PFSoftMuonByIP3dBJetTags
                +
                ak3PFSoftMuonByPtBJetTags
                +
                ak3PFNegativeSoftMuonByPtBJetTags
                +
                ak3PFPositiveSoftMuonByPtBJetTags
              )
            )

ak3PFJetBtagging = cms.Sequence(ak3PFJetBtaggingIP
            *ak3PFJetBtaggingSV
            *ak3PFJetBtaggingNegSV
            *ak3PFJetBtaggingMu
            )

ak3PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak3PFJets"),
        genJetMatch          = cms.InputTag("ak3PFmatch"),
        genPartonMatch       = cms.InputTag("ak3PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak3PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak3PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak3PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak3PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak3PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak3PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak3PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak3PFJetBProbabilityBJetTags"),
            cms.InputTag("ak3PFJetProbabilityBJetTags"),
            cms.InputTag("ak3PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak3PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak3PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak3PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak3PFJetID"),
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

ak3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak3PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJetsCleaned',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("ak3PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak3PFJetSequence_mc = cms.Sequence(
                                                  ak3PFclean
                                                  *
                                                  ak3PFmatch
                                                  *
                                                  ak3PFparton
                                                  *
                                                  ak3PFcorr
                                                  *
                                                  ak3PFJetID
                                                  *
                                                  ak3PFPatJetFlavourId
                                                  *
                                                  ak3PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak3PFJetBtagging
                                                  *
                                                  ak3PFpatJetsWithBtagging
                                                  *
                                                  ak3PFJetAnalyzer
                                                  )

ak3PFJetSequence_data = cms.Sequence(ak3PFcorr
                                                    *
                                                    ak3PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak3PFJetBtagging
                                                    *
                                                    ak3PFpatJetsWithBtagging
                                                    *
                                                    ak3PFJetAnalyzer
                                                    )

ak3PFJetSequence_jec = ak3PFJetSequence_mc
ak3PFJetSequence_mix = ak3PFJetSequence_mc

ak3PFJetSequence = cms.Sequence(ak3PFJetSequence_jec)
ak3PFJetAnalyzer.genPtMin = cms.untracked.double(1)
