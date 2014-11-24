

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4PFJets"),
    matched = cms.InputTag("ak4HiGenJets")
    )

ak4PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak4PFJets")
                                                        )

ak4PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak4PFJets"),
    payload = "AK4PF_generalTracks"
    )

ak4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak4CaloJets'))

ak4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

ak4PFbTagger = bTaggers("ak4PF")

#create objects locally since they dont load properly otherwise
ak4PFmatch = ak4PFbTagger.match
ak4PFparton = ak4PFbTagger.parton
ak4PFPatJetFlavourAssociation = ak4PFbTagger.PatJetFlavourAssociation
ak4PFJetTracksAssociatorAtVertex = ak4PFbTagger.JetTracksAssociatorAtVertex
ak4PFSimpleSecondaryVertexHighEffBJetTags = ak4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak4PFSimpleSecondaryVertexHighPurBJetTags = ak4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak4PFCombinedSecondaryVertexBJetTags = ak4PFbTagger.CombinedSecondaryVertexBJetTags
ak4PFCombinedSecondaryVertexMVABJetTags = ak4PFbTagger.CombinedSecondaryVertexMVABJetTags
ak4PFJetBProbabilityBJetTags = ak4PFbTagger.JetBProbabilityBJetTags
ak4PFSoftMuonByPtBJetTags = ak4PFbTagger.SoftMuonByPtBJetTags
ak4PFSoftMuonByIP3dBJetTags = ak4PFbTagger.SoftMuonByIP3dBJetTags
ak4PFTrackCountingHighEffBJetTags = ak4PFbTagger.TrackCountingHighEffBJetTags
ak4PFTrackCountingHighPurBJetTags = ak4PFbTagger.TrackCountingHighPurBJetTags
ak4PFPatJetPartonAssociation = ak4PFbTagger.PatJetPartonAssociation

ak4PFImpactParameterTagInfos = ak4PFbTagger.ImpactParameterTagInfos
ak4PFJetProbabilityBJetTags = ak4PFbTagger.JetProbabilityBJetTags
ak4PFPositiveOnlyJetProbabilityJetTags = ak4PFbTagger.PositiveOnlyJetProbabilityJetTags
ak4PFNegativeOnlyJetProbabilityJetTags = ak4PFbTagger.NegativeOnlyJetProbabilityJetTags
ak4PFNegativeTrackCountingHighEffJetTags = ak4PFbTagger.NegativeTrackCountingHighEffJetTags
ak4PFNegativeTrackCountingHighPur = ak4PFbTagger.NegativeTrackCountingHighPur
ak4PFNegativeOnlyJetBProbabilityJetTags = ak4PFbTagger.NegativeOnlyJetBProbabilityJetTags
ak4PFPositiveOnlyJetBProbabilityJetTags = ak4PFbTagger.PositiveOnlyJetBProbabilityJetTags

ak4PFSecondaryVertexTagInfos = ak4PFbTagger.SecondaryVertexTagInfos
ak4PFSimpleSecondaryVertexHighEffBJetTags = ak4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak4PFSimpleSecondaryVertexHighPurBJetTags = ak4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak4PFCombinedSecondaryVertexBJetTags = ak4PFbTagger.CombinedSecondaryVertexBJetTags
ak4PFCombinedSecondaryVertexMVABJetTags = ak4PFbTagger.CombinedSecondaryVertexMVABJetTags

ak4PFSecondaryVertexNegativeTagInfos = ak4PFbTagger.SecondaryVertexNegativeTagInfos
ak4PFSimpleSecondaryVertexNegativeHighEffBJetTags = ak4PFbTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak4PFSimpleSecondaryVertexNegativeHighPurBJetTags = ak4PFbTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak4PFCombinedSecondaryVertexNegativeBJetTags = ak4PFbTagger.CombinedSecondaryVertexNegativeBJetTags
ak4PFCombinedSecondaryVertexPositiveBJetTags = ak4PFbTagger.CombinedSecondaryVertexPositiveBJetTags

ak4PFSoftMuonTagInfos = ak4PFbTagger.SoftMuonTagInfos
ak4PFSoftMuonBJetTags = ak4PFbTagger.SoftMuonBJetTags
ak4PFSoftMuonByIP3dBJetTags = ak4PFbTagger.SoftMuonByIP3dBJetTags
ak4PFSoftMuonByPtBJetTags = ak4PFbTagger.SoftMuonByPtBJetTags
ak4PFNegativeSoftMuonByPtBJetTags = ak4PFbTagger.NegativeSoftMuonByPtBJetTags
ak4PFPositiveSoftMuonByPtBJetTags = ak4PFbTagger.PositiveSoftMuonByPtBJetTags

ak4PFPatJetFlavourId = cms.Sequence(ak4PFPatJetPartonAssociation*ak4PFPatJetFlavourAssociation)

ak4PFJetBtaggingIP       = cms.Sequence(ak4PFImpactParameterTagInfos *
            (ak4PFTrackCountingHighEffBJetTags +
             ak4PFTrackCountingHighPurBJetTags +
             ak4PFJetProbabilityBJetTags +
             ak4PFJetBProbabilityBJetTags +
             ak4PFPositiveOnlyJetProbabilityJetTags +
             ak4PFNegativeOnlyJetProbabilityJetTags +
             ak4PFNegativeTrackCountingHighEffJetTags +
             ak4PFNegativeTrackCountingHighPur +
             ak4PFNegativeOnlyJetBProbabilityJetTags +
             ak4PFPositiveOnlyJetBProbabilityJetTags
            )
            )

ak4PFJetBtaggingSV = cms.Sequence(ak4PFImpactParameterTagInfos
            *
            ak4PFSecondaryVertexTagInfos
            * (ak4PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak4PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak4PFCombinedSecondaryVertexBJetTags
                +
                ak4PFCombinedSecondaryVertexMVABJetTags
              )
            )

ak4PFJetBtaggingNegSV = cms.Sequence(ak4PFImpactParameterTagInfos
            *
            ak4PFSecondaryVertexNegativeTagInfos
            * (ak4PFSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak4PFSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak4PFCombinedSecondaryVertexNegativeBJetTags
                +
                ak4PFCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak4PFJetBtaggingMu = cms.Sequence(ak4PFSoftMuonTagInfos * (ak4PFSoftMuonBJetTags
                +
                ak4PFSoftMuonByIP3dBJetTags
                +
                ak4PFSoftMuonByPtBJetTags
                +
                ak4PFNegativeSoftMuonByPtBJetTags
                +
                ak4PFPositiveSoftMuonByPtBJetTags
              )
            )

ak4PFJetBtagging = cms.Sequence(ak4PFJetBtaggingIP
            *ak4PFJetBtaggingSV
            *ak4PFJetBtaggingNegSV
            *ak4PFJetBtaggingMu
            )

ak4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak4PFJets"),
        genJetMatch          = cms.InputTag("ak4PFmatch"),
        genPartonMatch       = cms.InputTag("ak4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak4PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak4PFJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak4PFCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak4PFJetBProbabilityBJetTags"),
            cms.InputTag("ak4PFJetProbabilityBJetTags"),
            cms.InputTag("ak4PFSoftMuonByPtBJetTags"),
            cms.InputTag("ak4PFSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak4PFJetID"),
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

ak4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak4PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak4PFJetSequence_mc = cms.Sequence(
                                                  ak4PFclean
                                                  *
                                                  ak4PFmatch
                                                  *
                                                  ak4PFparton
                                                  *
                                                  ak4PFcorr
                                                  *
                                                  ak4PFJetID
                                                  *
                                                  ak4PFPatJetFlavourId
                                                  *
                                                  ak4PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak4PFJetBtagging
                                                  *
                                                  ak4PFpatJetsWithBtagging
                                                  *
                                                  ak4PFJetAnalyzer
                                                  )

ak4PFJetSequence_data = cms.Sequence(ak4PFcorr
                                                    *
                                                    ak4PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak4PFJetBtagging
                                                    *
                                                    ak4PFpatJetsWithBtagging
                                                    *
                                                    ak4PFJetAnalyzer
                                                    )

ak4PFJetSequence_jec = ak4PFJetSequence_mc
ak4PFJetSequence_mix = ak4PFJetSequence_mc

ak4PFJetSequence = cms.Sequence(ak4PFJetSequence_jec)
ak4PFJetAnalyzer.genPtMin = cms.untracked.double(1)
