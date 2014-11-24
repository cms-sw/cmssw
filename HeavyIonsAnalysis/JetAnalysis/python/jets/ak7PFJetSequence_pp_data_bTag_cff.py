

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak7PFJets"),
    matched = cms.InputTag("ak7HiGenJets")
    )

ak7PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak7PFJets")
                                                        )

ak7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak7PFJets"),
    payload = "AK7PF_generalTracks"
    )

ak7PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak7CaloJets'))

ak7PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

ak7PFbTagger = bTaggers("ak7PF")

#create objects locally since they dont load properly otherwise
ak7PFmatch = ak7PFbTagger.match
ak7PFparton = ak7PFbTagger.parton
ak7PFPatJetFlavourAssociation = ak7PFbTagger.PatJetFlavourAssociation
ak7PFJetTracksAssociatorAtVertex = ak7PFbTagger.JetTracksAssociatorAtVertex
ak7PFSimpleSecondaryVertexHighEffBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak7PFSimpleSecondaryVertexHighPurBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak7PFCombinedSecondaryVertexBJetTags = ak7PFbTagger.CombinedSecondaryVertexBJetTags
ak7PFCombinedSecondaryVertexMVABJetTags = ak7PFbTagger.CombinedSecondaryVertexMVABJetTags
ak7PFJetBProbabilityBJetTags = ak7PFbTagger.JetBProbabilityBJetTags
ak7PFSoftMuonByPtBJetTags = ak7PFbTagger.SoftMuonByPtBJetTags
ak7PFSoftMuonByIP3dBJetTags = ak7PFbTagger.SoftMuonByIP3dBJetTags
ak7PFTrackCountingHighEffBJetTags = ak7PFbTagger.TrackCountingHighEffBJetTags
ak7PFTrackCountingHighPurBJetTags = ak7PFbTagger.TrackCountingHighPurBJetTags
ak7PFPatJetPartonAssociation = ak7PFbTagger.PatJetPartonAssociation

ak7PFImpactParameterTagInfos = ak7PFbTagger.ImpactParameterTagInfos
ak7PFJetProbabilityBJetTags = ak7PFbTagger.JetProbabilityBJetTags
ak7PFPositiveOnlyJetProbabilityJetTags = ak7PFbTagger.PositiveOnlyJetProbabilityJetTags
ak7PFNegativeOnlyJetProbabilityJetTags = ak7PFbTagger.NegativeOnlyJetProbabilityJetTags
ak7PFNegativeTrackCountingHighEffJetTags = ak7PFbTagger.NegativeTrackCountingHighEffJetTags
ak7PFNegativeTrackCountingHighPur = ak7PFbTagger.NegativeTrackCountingHighPur
ak7PFNegativeOnlyJetBProbabilityJetTags = ak7PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak7PFPositiveOnlyJetBProbabilityJetTags = ak7PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak7PFSecondaryVertexTagInfos = ak7PFbTagger.SecondaryVertexTagInfos
ak7PFSimpleSecondaryVertexHighEffBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak7PFSimpleSecondaryVertexHighPurBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak7PFCombinedSecondaryVertexBJetTags = ak7PFbTagger.CombinedSecondaryVertexBJetTags
ak7PFCombinedSecondaryVertexMVABJetTags = ak7PFbTagger.CombinedSecondaryVertexMVABJetTags

ak7PFSecondaryVertexNegativeTagInfos = ak7PFbTagger.SecondaryVertexNegativeTagInfos
ak7PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak7PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak7PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak7PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak7PFCombinedSecondaryVertexNegativeBJetTags = ak7PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak7PFCombinedSecondaryVertexPositiveBJetTags = ak7PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak7PFSoftMuonTagInfos = ak7PFbTagger.SoftMuonTagInfos
ak7PFSoftMuonBJetTags = ak7PFbTagger.SoftMuonBJetTags
ak7PFSoftMuonByIP3dBJetTags = ak7PFbTagger.SoftMuonByIP3dBJetTags
ak7PFSoftMuonByPtBJetTags = ak7PFbTagger.SoftMuonByPtBJetTags
ak7PFNegativeSoftMuonByPtBJetTags = ak7PFbTagger.NegativeSoftMuonByPtBJetTags
ak7PFPositiveSoftMuonByPtBJetTags = ak7PFbTagger.PositiveSoftMuonByPtBJetTags

ak7PFPatJetFlavourId = cms.Sequence(ak7PFPatJetPartonAssociation*ak7PFPatJetFlavourAssociation)

ak7PFJetBtaggingIP       = cms.Sequence(ak7PFImpactParameterTagInfos *
            (ak7PFTrackCountingHighEffBJetTags +
             ak7PFTrackCountingHighPurBJetTags +
             ak7PFJetProbabilityBJetTags +
             ak7PFJetBProbabilityBJetTags +
             ak7PFPositiveOnlyJetProbabilityJetTags +
             ak7PFNegativeOnlyJetProbabilityJetTags +
             ak7PFNegativeTrackCountingHighEffJetTags +
             ak7PFNegativeTrackCountingHighPur +
             ak7PFNegativeOnlyJetBProbabilityJetTags +
             ak7PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak7PFJetBtaggingSV = cms.Sequence(ak7PFImpactParameterTagInfos
            *
            ak7PFSecondaryVertexTagInfos
            * (ak7PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak7PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak7PFCombinedSecondaryVertexBJetTags
                +
                ak7PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak7PFJetBtaggingNegSV = cms.Sequence(ak7PFImpactParameterTagInfos
            *
            ak7PFSecondaryVertexNegativeTagInfos
            * (ak7PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak7PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak7PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak7PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak7PFJetBtaggingMu = cms.Sequence(ak7PFSoftMuonTagInfos * (ak7PFSoftMuonBJetTags
                +
                ak7PFSoftMuonByIP3dBJetTags
                +
                ak7PFSoftMuonByPtBJetTags
                +
                ak7PFNegativeSoftMuonByPtBJetTags
                +
                ak7PFPositiveSoftMuonByPtBJetTags
              )
            )

ak7PFJetBtagging = cms.Sequence(ak7PFJetBtaggingIP
            *ak7PFJetBtaggingSV
            *ak7PFJetBtaggingNegSV
            *ak7PFJetBtaggingMu
            )

ak7PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak7PFJets"),
        genJetMatch          = cms.InputTag("ak7PFmatch"),
        genPartonMatch       = cms.InputTag("ak7PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak7PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak7PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak7PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak7PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak7PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak7PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak7PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak7PFJetBProbabilityBJetTags"),
            cms.InputTag("ak7PFJetProbabilityBJetTags"),
            cms.InputTag("ak7PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak7PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak7PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak7PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak7PFJetID"),
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

ak7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak7PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak7PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak7PFJetSequence_mc = cms.Sequence(
                                                  ak7PFclean
                                                  *
                                                  ak7PFmatch
                                                  *
                                                  ak7PFparton
                                                  *
                                                  ak7PFcorr
                                                  *
                                                  ak7PFJetID
                                                  *
                                                  ak7PFPatJetFlavourId
                                                  *
                                                  ak7PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak7PFJetBtagging
                                                  *
                                                  ak7PFpatJetsWithBtagging
                                                  *
                                                  ak7PFJetAnalyzer
                                                  )

ak7PFJetSequence_data = cms.Sequence(ak7PFcorr
                                                    *
                                                    ak7PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak7PFJetBtagging
                                                    *
                                                    ak7PFpatJetsWithBtagging
                                                    *
                                                    ak7PFJetAnalyzer
                                                    )

ak7PFJetSequence_jec = ak7PFJetSequence_mc
ak7PFJetSequence_mix = ak7PFJetSequence_mc

ak7PFJetSequence = cms.Sequence(ak7PFJetSequence_data)
