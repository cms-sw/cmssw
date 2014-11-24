

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak6PFJets"),
    matched = cms.InputTag("ak6HiGenJetsCleaned")
    )

ak6PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak6PFJets")
                                                        )

ak6PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak6PFJets"),
    payload = "AK6PF_hiIterativeTracks"
    )

ak6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak6CaloJets'))

ak6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJetsCleaned'))

ak6PFbTagger = bTaggers("ak6PF")

#create objects locally since they dont load properly otherwise
ak6PFmatch = ak6PFbTagger.match
ak6PFparton = ak6PFbTagger.parton
ak6PFPatJetFlavourAssociation = ak6PFbTagger.PatJetFlavourAssociation
ak6PFJetTracksAssociatorAtVertex = ak6PFbTagger.JetTracksAssociatorAtVertex
ak6PFSimpleSecondaryVertexHighEffBJetTags = ak6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak6PFSimpleSecondaryVertexHighPurBJetTags = ak6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak6PFCombinedSecondaryVertexBJetTags = ak6PFbTagger.CombinedSecondaryVertexBJetTags
ak6PFCombinedSecondaryVertexMVABJetTags = ak6PFbTagger.CombinedSecondaryVertexMVABJetTags
ak6PFJetBProbabilityBJetTags = ak6PFbTagger.JetBProbabilityBJetTags
ak6PFSoftMuonByPtBJetTags = ak6PFbTagger.SoftMuonByPtBJetTags
ak6PFSoftMuonByIP3dBJetTags = ak6PFbTagger.SoftMuonByIP3dBJetTags
ak6PFTrackCountingHighEffBJetTags = ak6PFbTagger.TrackCountingHighEffBJetTags
ak6PFTrackCountingHighPurBJetTags = ak6PFbTagger.TrackCountingHighPurBJetTags
ak6PFPatJetPartonAssociation = ak6PFbTagger.PatJetPartonAssociation

ak6PFImpactParameterTagInfos = ak6PFbTagger.ImpactParameterTagInfos
ak6PFJetProbabilityBJetTags = ak6PFbTagger.JetProbabilityBJetTags
ak6PFPositiveOnlyJetProbabilityJetTags = ak6PFbTagger.PositiveOnlyJetProbabilityJetTags
ak6PFNegativeOnlyJetProbabilityJetTags = ak6PFbTagger.NegativeOnlyJetProbabilityJetTags
ak6PFNegativeTrackCountingHighEffJetTags = ak6PFbTagger.NegativeTrackCountingHighEffJetTags
ak6PFNegativeTrackCountingHighPur = ak6PFbTagger.NegativeTrackCountingHighPur
ak6PFNegativeOnlyJetBProbabilityJetTags = ak6PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak6PFPositiveOnlyJetBProbabilityJetTags = ak6PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak6PFSecondaryVertexTagInfos = ak6PFbTagger.SecondaryVertexTagInfos
ak6PFSimpleSecondaryVertexHighEffBJetTags = ak6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak6PFSimpleSecondaryVertexHighPurBJetTags = ak6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak6PFCombinedSecondaryVertexBJetTags = ak6PFbTagger.CombinedSecondaryVertexBJetTags
ak6PFCombinedSecondaryVertexMVABJetTags = ak6PFbTagger.CombinedSecondaryVertexMVABJetTags

ak6PFSecondaryVertexNegativeTagInfos = ak6PFbTagger.SecondaryVertexNegativeTagInfos
ak6PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak6PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak6PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak6PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak6PFCombinedSecondaryVertexNegativeBJetTags = ak6PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak6PFCombinedSecondaryVertexPositiveBJetTags = ak6PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak6PFSoftMuonTagInfos = ak6PFbTagger.SoftMuonTagInfos
ak6PFSoftMuonBJetTags = ak6PFbTagger.SoftMuonBJetTags
ak6PFSoftMuonByIP3dBJetTags = ak6PFbTagger.SoftMuonByIP3dBJetTags
ak6PFSoftMuonByPtBJetTags = ak6PFbTagger.SoftMuonByPtBJetTags
ak6PFNegativeSoftMuonByPtBJetTags = ak6PFbTagger.NegativeSoftMuonByPtBJetTags
ak6PFPositiveSoftMuonByPtBJetTags = ak6PFbTagger.PositiveSoftMuonByPtBJetTags

ak6PFPatJetFlavourId = cms.Sequence(ak6PFPatJetPartonAssociation*ak6PFPatJetFlavourAssociation)

ak6PFJetBtaggingIP       = cms.Sequence(ak6PFImpactParameterTagInfos *
            (ak6PFTrackCountingHighEffBJetTags +
             ak6PFTrackCountingHighPurBJetTags +
             ak6PFJetProbabilityBJetTags +
             ak6PFJetBProbabilityBJetTags +
             ak6PFPositiveOnlyJetProbabilityJetTags +
             ak6PFNegativeOnlyJetProbabilityJetTags +
             ak6PFNegativeTrackCountingHighEffJetTags +
             ak6PFNegativeTrackCountingHighPur +
             ak6PFNegativeOnlyJetBProbabilityJetTags +
             ak6PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak6PFJetBtaggingSV = cms.Sequence(ak6PFImpactParameterTagInfos
            *
            ak6PFSecondaryVertexTagInfos
            * (ak6PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak6PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak6PFCombinedSecondaryVertexBJetTags
                +
                ak6PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak6PFJetBtaggingNegSV = cms.Sequence(ak6PFImpactParameterTagInfos
            *
            ak6PFSecondaryVertexNegativeTagInfos
            * (ak6PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak6PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak6PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak6PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak6PFJetBtaggingMu = cms.Sequence(ak6PFSoftMuonTagInfos * (ak6PFSoftMuonBJetTags
                +
                ak6PFSoftMuonByIP3dBJetTags
                +
                ak6PFSoftMuonByPtBJetTags
                +
                ak6PFNegativeSoftMuonByPtBJetTags
                +
                ak6PFPositiveSoftMuonByPtBJetTags
              )
            )

ak6PFJetBtagging = cms.Sequence(ak6PFJetBtaggingIP
            *ak6PFJetBtaggingSV
            *ak6PFJetBtaggingNegSV
            *ak6PFJetBtaggingMu
            )

ak6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak6PFJets"),
        genJetMatch          = cms.InputTag("ak6PFmatch"),
        genPartonMatch       = cms.InputTag("ak6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak6PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak6PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak6PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak6PFJetBProbabilityBJetTags"),
            cms.InputTag("ak6PFJetProbabilityBJetTags"),
            cms.InputTag("ak6PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak6PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak6PFJetID"),
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

ak6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak6PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJetsCleaned',
                                                             rParam = 0.6,
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
                                                             bTagJetName = cms.untracked.string("ak6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak6PFJetSequence_mc = cms.Sequence(
                                                  ak6PFclean
                                                  *
                                                  ak6PFmatch
                                                  *
                                                  ak6PFparton
                                                  *
                                                  ak6PFcorr
                                                  *
                                                  ak6PFJetID
                                                  *
                                                  ak6PFPatJetFlavourId
                                                  *
                                                  ak6PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak6PFJetBtagging
                                                  *
                                                  ak6PFpatJetsWithBtagging
                                                  *
                                                  ak6PFJetAnalyzer
                                                  )

ak6PFJetSequence_data = cms.Sequence(ak6PFcorr
                                                    *
                                                    ak6PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak6PFJetBtagging
                                                    *
                                                    ak6PFpatJetsWithBtagging
                                                    *
                                                    ak6PFJetAnalyzer
                                                    )

ak6PFJetSequence_jec = ak6PFJetSequence_mc
ak6PFJetSequence_mix = ak6PFJetSequence_mc

ak6PFJetSequence = cms.Sequence(ak6PFJetSequence_jec)
ak6PFJetAnalyzer.genPtMin = cms.untracked.double(1)
