

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak1PFJets"),
    matched = cms.InputTag("ak1HiGenJets")
    )

ak1PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak1PFJets")
                                                        )

ak1PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak1PFJets"),
    payload = "AK1PF_generalTracks"
    )

ak1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak1CaloJets'))

ak1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJets'))

ak1PFbTagger = bTaggers("ak1PF")

#create objects locally since they dont load properly otherwise
ak1PFmatch = ak1PFbTagger.match
ak1PFparton = ak1PFbTagger.parton
ak1PFPatJetFlavourAssociation = ak1PFbTagger.PatJetFlavourAssociation
ak1PFJetTracksAssociatorAtVertex = ak1PFbTagger.JetTracksAssociatorAtVertex
ak1PFSimpleSecondaryVertexHighEffBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak1PFSimpleSecondaryVertexHighPurBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak1PFCombinedSecondaryVertexBJetTags = ak1PFbTagger.CombinedSecondaryVertexBJetTags
ak1PFCombinedSecondaryVertexMVABJetTags = ak1PFbTagger.CombinedSecondaryVertexMVABJetTags
ak1PFJetBProbabilityBJetTags = ak1PFbTagger.JetBProbabilityBJetTags
ak1PFSoftMuonByPtBJetTags = ak1PFbTagger.SoftMuonByPtBJetTags
ak1PFSoftMuonByIP3dBJetTags = ak1PFbTagger.SoftMuonByIP3dBJetTags
ak1PFTrackCountingHighEffBJetTags = ak1PFbTagger.TrackCountingHighEffBJetTags
ak1PFTrackCountingHighPurBJetTags = ak1PFbTagger.TrackCountingHighPurBJetTags
ak1PFPatJetPartonAssociation = ak1PFbTagger.PatJetPartonAssociation

ak1PFImpactParameterTagInfos = ak1PFbTagger.ImpactParameterTagInfos
ak1PFJetProbabilityBJetTags = ak1PFbTagger.JetProbabilityBJetTags
ak1PFPositiveOnlyJetProbabilityJetTags = ak1PFbTagger.PositiveOnlyJetProbabilityJetTags
ak1PFNegativeOnlyJetProbabilityJetTags = ak1PFbTagger.NegativeOnlyJetProbabilityJetTags
ak1PFNegativeTrackCountingHighEffJetTags = ak1PFbTagger.NegativeTrackCountingHighEffJetTags
ak1PFNegativeTrackCountingHighPur = ak1PFbTagger.NegativeTrackCountingHighPur
ak1PFNegativeOnlyJetBProbabilityJetTags = ak1PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak1PFPositiveOnlyJetBProbabilityJetTags = ak1PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak1PFSecondaryVertexTagInfos = ak1PFbTagger.SecondaryVertexTagInfos
ak1PFSimpleSecondaryVertexHighEffBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak1PFSimpleSecondaryVertexHighPurBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak1PFCombinedSecondaryVertexBJetTags = ak1PFbTagger.CombinedSecondaryVertexBJetTags
ak1PFCombinedSecondaryVertexMVABJetTags = ak1PFbTagger.CombinedSecondaryVertexMVABJetTags

ak1PFSecondaryVertexNegativeTagInfos = ak1PFbTagger.SecondaryVertexNegativeTagInfos
ak1PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak1PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak1PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak1PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak1PFCombinedSecondaryVertexNegativeBJetTags = ak1PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak1PFCombinedSecondaryVertexPositiveBJetTags = ak1PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak1PFSoftMuonTagInfos = ak1PFbTagger.SoftMuonTagInfos
ak1PFSoftMuonBJetTags = ak1PFbTagger.SoftMuonBJetTags
ak1PFSoftMuonByIP3dBJetTags = ak1PFbTagger.SoftMuonByIP3dBJetTags
ak1PFSoftMuonByPtBJetTags = ak1PFbTagger.SoftMuonByPtBJetTags
ak1PFNegativeSoftMuonByPtBJetTags = ak1PFbTagger.NegativeSoftMuonByPtBJetTags
ak1PFPositiveSoftMuonByPtBJetTags = ak1PFbTagger.PositiveSoftMuonByPtBJetTags

ak1PFPatJetFlavourId = cms.Sequence(ak1PFPatJetPartonAssociation*ak1PFPatJetFlavourAssociation)

ak1PFJetBtaggingIP       = cms.Sequence(ak1PFImpactParameterTagInfos *
            (ak1PFTrackCountingHighEffBJetTags +
             ak1PFTrackCountingHighPurBJetTags +
             ak1PFJetProbabilityBJetTags +
             ak1PFJetBProbabilityBJetTags +
             ak1PFPositiveOnlyJetProbabilityJetTags +
             ak1PFNegativeOnlyJetProbabilityJetTags +
             ak1PFNegativeTrackCountingHighEffJetTags +
             ak1PFNegativeTrackCountingHighPur +
             ak1PFNegativeOnlyJetBProbabilityJetTags +
             ak1PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak1PFJetBtaggingSV = cms.Sequence(ak1PFImpactParameterTagInfos
            *
            ak1PFSecondaryVertexTagInfos
            * (ak1PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak1PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak1PFCombinedSecondaryVertexBJetTags
                +
                ak1PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak1PFJetBtaggingNegSV = cms.Sequence(ak1PFImpactParameterTagInfos
            *
            ak1PFSecondaryVertexNegativeTagInfos
            * (ak1PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak1PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak1PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak1PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak1PFJetBtaggingMu = cms.Sequence(ak1PFSoftMuonTagInfos * (ak1PFSoftMuonBJetTags
                +
                ak1PFSoftMuonByIP3dBJetTags
                +
                ak1PFSoftMuonByPtBJetTags
                +
                ak1PFNegativeSoftMuonByPtBJetTags
                +
                ak1PFPositiveSoftMuonByPtBJetTags
              )
            )

ak1PFJetBtagging = cms.Sequence(ak1PFJetBtaggingIP
            *ak1PFJetBtaggingSV
            *ak1PFJetBtaggingNegSV
            *ak1PFJetBtaggingMu
            )

ak1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak1PFJets"),
        genJetMatch          = cms.InputTag("ak1PFmatch"),
        genPartonMatch       = cms.InputTag("ak1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak1PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak1PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak1PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak1PFJetBProbabilityBJetTags"),
            cms.InputTag("ak1PFJetProbabilityBJetTags"),
            cms.InputTag("ak1PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak1PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak1PFJetID"),
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

ak1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak1PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("ak1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak1PFJetSequence_mc = cms.Sequence(
                                                  ak1PFclean
                                                  *
                                                  ak1PFmatch
                                                  *
                                                  ak1PFparton
                                                  *
                                                  ak1PFcorr
                                                  *
                                                  ak1PFJetID
                                                  *
                                                  ak1PFPatJetFlavourId
                                                  *
                                                  ak1PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak1PFJetBtagging
                                                  *
                                                  ak1PFpatJetsWithBtagging
                                                  *
                                                  ak1PFJetAnalyzer
                                                  )

ak1PFJetSequence_data = cms.Sequence(ak1PFcorr
                                                    *
                                                    ak1PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak1PFJetBtagging
                                                    *
                                                    ak1PFpatJetsWithBtagging
                                                    *
                                                    ak1PFJetAnalyzer
                                                    )

ak1PFJetSequence_jec = ak1PFJetSequence_mc
ak1PFJetSequence_mix = ak1PFJetSequence_mc

ak1PFJetSequence = cms.Sequence(ak1PFJetSequence_mix)
