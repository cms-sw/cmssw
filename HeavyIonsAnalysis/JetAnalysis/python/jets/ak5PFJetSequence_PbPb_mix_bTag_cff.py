

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak5PFJets"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

ak5PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak5PFJets")
                                                        )

ak5PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak5PFJets"),
    payload = "AK5PF_hiIterativeTracks"
    )

ak5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak5CaloJets'))

ak5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJetsCleaned'))

ak5PFbTagger = bTaggers("ak5PF")

#create objects locally since they dont load properly otherwise
ak5PFmatch = ak5PFbTagger.match
ak5PFparton = ak5PFbTagger.parton
ak5PFPatJetFlavourAssociation = ak5PFbTagger.PatJetFlavourAssociation
ak5PFJetTracksAssociatorAtVertex = ak5PFbTagger.JetTracksAssociatorAtVertex
ak5PFSimpleSecondaryVertexHighEffBJetTags = ak5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak5PFSimpleSecondaryVertexHighPurBJetTags = ak5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak5PFCombinedSecondaryVertexBJetTags = ak5PFbTagger.CombinedSecondaryVertexBJetTags
ak5PFCombinedSecondaryVertexMVABJetTags = ak5PFbTagger.CombinedSecondaryVertexMVABJetTags
ak5PFJetBProbabilityBJetTags = ak5PFbTagger.JetBProbabilityBJetTags
ak5PFSoftMuonByPtBJetTags = ak5PFbTagger.SoftMuonByPtBJetTags
ak5PFSoftMuonByIP3dBJetTags = ak5PFbTagger.SoftMuonByIP3dBJetTags
ak5PFTrackCountingHighEffBJetTags = ak5PFbTagger.TrackCountingHighEffBJetTags
ak5PFTrackCountingHighPurBJetTags = ak5PFbTagger.TrackCountingHighPurBJetTags
ak5PFPatJetPartonAssociation = ak5PFbTagger.PatJetPartonAssociation

ak5PFImpactParameterTagInfos = ak5PFbTagger.ImpactParameterTagInfos
ak5PFJetProbabilityBJetTags = ak5PFbTagger.JetProbabilityBJetTags
ak5PFPositiveOnlyJetProbabilityJetTags = ak5PFbTagger.PositiveOnlyJetProbabilityJetTags
ak5PFNegativeOnlyJetProbabilityJetTags = ak5PFbTagger.NegativeOnlyJetProbabilityJetTags
ak5PFNegativeTrackCountingHighEffJetTags = ak5PFbTagger.NegativeTrackCountingHighEffJetTags
ak5PFNegativeTrackCountingHighPur = ak5PFbTagger.NegativeTrackCountingHighPur
ak5PFNegativeOnlyJetBProbabilityJetTags = ak5PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak5PFPositiveOnlyJetBProbabilityJetTags = ak5PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak5PFSecondaryVertexTagInfos = ak5PFbTagger.SecondaryVertexTagInfos
ak5PFSimpleSecondaryVertexHighEffBJetTags = ak5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak5PFSimpleSecondaryVertexHighPurBJetTags = ak5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak5PFCombinedSecondaryVertexBJetTags = ak5PFbTagger.CombinedSecondaryVertexBJetTags
ak5PFCombinedSecondaryVertexMVABJetTags = ak5PFbTagger.CombinedSecondaryVertexMVABJetTags

ak5PFSecondaryVertexNegativeTagInfos = ak5PFbTagger.SecondaryVertexNegativeTagInfos
ak5PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak5PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak5PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak5PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak5PFCombinedSecondaryVertexNegativeBJetTags = ak5PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak5PFCombinedSecondaryVertexPositiveBJetTags = ak5PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak5PFSoftMuonTagInfos = ak5PFbTagger.SoftMuonTagInfos
ak5PFSoftMuonBJetTags = ak5PFbTagger.SoftMuonBJetTags
ak5PFSoftMuonByIP3dBJetTags = ak5PFbTagger.SoftMuonByIP3dBJetTags
ak5PFSoftMuonByPtBJetTags = ak5PFbTagger.SoftMuonByPtBJetTags
ak5PFNegativeSoftMuonByPtBJetTags = ak5PFbTagger.NegativeSoftMuonByPtBJetTags
ak5PFPositiveSoftMuonByPtBJetTags = ak5PFbTagger.PositiveSoftMuonByPtBJetTags

ak5PFPatJetFlavourId = cms.Sequence(ak5PFPatJetPartonAssociation*ak5PFPatJetFlavourAssociation)

ak5PFJetBtaggingIP       = cms.Sequence(ak5PFImpactParameterTagInfos *
            (ak5PFTrackCountingHighEffBJetTags +
             ak5PFTrackCountingHighPurBJetTags +
             ak5PFJetProbabilityBJetTags +
             ak5PFJetBProbabilityBJetTags +
             ak5PFPositiveOnlyJetProbabilityJetTags +
             ak5PFNegativeOnlyJetProbabilityJetTags +
             ak5PFNegativeTrackCountingHighEffJetTags +
             ak5PFNegativeTrackCountingHighPur +
             ak5PFNegativeOnlyJetBProbabilityJetTags +
             ak5PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak5PFJetBtaggingSV = cms.Sequence(ak5PFImpactParameterTagInfos
            *
            ak5PFSecondaryVertexTagInfos
            * (ak5PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak5PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak5PFCombinedSecondaryVertexBJetTags
                +
                ak5PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak5PFJetBtaggingNegSV = cms.Sequence(ak5PFImpactParameterTagInfos
            *
            ak5PFSecondaryVertexNegativeTagInfos
            * (ak5PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak5PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak5PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak5PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak5PFJetBtaggingMu = cms.Sequence(ak5PFSoftMuonTagInfos * (ak5PFSoftMuonBJetTags
                +
                ak5PFSoftMuonByIP3dBJetTags
                +
                ak5PFSoftMuonByPtBJetTags
                +
                ak5PFNegativeSoftMuonByPtBJetTags
                +
                ak5PFPositiveSoftMuonByPtBJetTags
              )
            )

ak5PFJetBtagging = cms.Sequence(ak5PFJetBtaggingIP
            *ak5PFJetBtaggingSV
            *ak5PFJetBtaggingNegSV
            *ak5PFJetBtaggingMu
            )

ak5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak5PFJets"),
        genJetMatch          = cms.InputTag("ak5PFmatch"),
        genPartonMatch       = cms.InputTag("ak5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak5PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak5PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak5PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak5PFJetBProbabilityBJetTags"),
            cms.InputTag("ak5PFJetProbabilityBJetTags"),
            cms.InputTag("ak5PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak5PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak5PFJetID"),
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

ak5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak5PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("ak5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak5PFJetSequence_mc = cms.Sequence(
                                                  ak5PFclean
                                                  *
                                                  ak5PFmatch
                                                  *
                                                  ak5PFparton
                                                  *
                                                  ak5PFcorr
                                                  *
                                                  ak5PFJetID
                                                  *
                                                  ak5PFPatJetFlavourId
                                                  *
                                                  ak5PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak5PFJetBtagging
                                                  *
                                                  ak5PFpatJetsWithBtagging
                                                  *
                                                  ak5PFJetAnalyzer
                                                  )

ak5PFJetSequence_data = cms.Sequence(ak5PFcorr
                                                    *
                                                    ak5PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak5PFJetBtagging
                                                    *
                                                    ak5PFpatJetsWithBtagging
                                                    *
                                                    ak5PFJetAnalyzer
                                                    )

ak5PFJetSequence_jec = ak5PFJetSequence_mc
ak5PFJetSequence_mix = ak5PFJetSequence_mc

ak5PFJetSequence = cms.Sequence(ak5PFJetSequence_mix)
