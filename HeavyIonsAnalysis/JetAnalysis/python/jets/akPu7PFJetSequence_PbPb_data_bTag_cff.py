

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu7PFJets"),
    matched = cms.InputTag("ak7HiGenJets"),
    maxDeltaR = 0.7
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

#akPu7PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

akPu7PFbTagger = bTaggers("akPu7PF",0.7)

#create objects locally since they dont load properly otherwise
#akPu7PFmatch = akPu7PFbTagger.match
akPu7PFparton = akPu7PFbTagger.parton
akPu7PFPatJetFlavourAssociationLegacy = akPu7PFbTagger.PatJetFlavourAssociationLegacy
akPu7PFPatJetPartons = akPu7PFbTagger.PatJetPartons
akPu7PFJetTracksAssociatorAtVertex = akPu7PFbTagger.JetTracksAssociatorAtVertex
akPu7PFSimpleSecondaryVertexHighEffBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7PFSimpleSecondaryVertexHighPurBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7PFCombinedSecondaryVertexBJetTags = akPu7PFbTagger.CombinedSecondaryVertexBJetTags
akPu7PFCombinedSecondaryVertexV2BJetTags = akPu7PFbTagger.CombinedSecondaryVertexV2BJetTags
akPu7PFJetBProbabilityBJetTags = akPu7PFbTagger.JetBProbabilityBJetTags
akPu7PFSoftPFMuonByPtBJetTags = akPu7PFbTagger.SoftPFMuonByPtBJetTags
akPu7PFSoftPFMuonByIP3dBJetTags = akPu7PFbTagger.SoftPFMuonByIP3dBJetTags
akPu7PFTrackCountingHighEffBJetTags = akPu7PFbTagger.TrackCountingHighEffBJetTags
akPu7PFTrackCountingHighPurBJetTags = akPu7PFbTagger.TrackCountingHighPurBJetTags
akPu7PFPatJetPartonAssociationLegacy = akPu7PFbTagger.PatJetPartonAssociationLegacy

akPu7PFImpactParameterTagInfos = akPu7PFbTagger.ImpactParameterTagInfos
akPu7PFJetProbabilityBJetTags = akPu7PFbTagger.JetProbabilityBJetTags
akPu7PFPositiveOnlyJetProbabilityBJetTags = akPu7PFbTagger.PositiveOnlyJetProbabilityBJetTags
akPu7PFNegativeOnlyJetProbabilityBJetTags = akPu7PFbTagger.NegativeOnlyJetProbabilityBJetTags
akPu7PFNegativeTrackCountingHighEffBJetTags = akPu7PFbTagger.NegativeTrackCountingHighEffBJetTags
akPu7PFNegativeTrackCountingHighPurBJetTags = akPu7PFbTagger.NegativeTrackCountingHighPurBJetTags
akPu7PFNegativeOnlyJetBProbabilityBJetTags = akPu7PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akPu7PFPositiveOnlyJetBProbabilityBJetTags = akPu7PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akPu7PFSecondaryVertexTagInfos = akPu7PFbTagger.SecondaryVertexTagInfos
akPu7PFSimpleSecondaryVertexHighEffBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7PFSimpleSecondaryVertexHighPurBJetTags = akPu7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7PFCombinedSecondaryVertexBJetTags = akPu7PFbTagger.CombinedSecondaryVertexBJetTags
akPu7PFCombinedSecondaryVertexV2BJetTags = akPu7PFbTagger.CombinedSecondaryVertexV2BJetTags

akPu7PFSecondaryVertexNegativeTagInfos = akPu7PFbTagger.SecondaryVertexNegativeTagInfos
akPu7PFNegativeSimpleSecondaryVertexHighEffBJetTags = akPu7PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu7PFNegativeSimpleSecondaryVertexHighPurBJetTags = akPu7PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu7PFNegativeCombinedSecondaryVertexBJetTags = akPu7PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akPu7PFPositiveCombinedSecondaryVertexBJetTags = akPu7PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akPu7PFSoftPFMuonsTagInfos = akPu7PFbTagger.SoftPFMuonsTagInfos
akPu7PFSoftPFMuonBJetTags = akPu7PFbTagger.SoftPFMuonBJetTags
akPu7PFSoftPFMuonByIP3dBJetTags = akPu7PFbTagger.SoftPFMuonByIP3dBJetTags
akPu7PFSoftPFMuonByPtBJetTags = akPu7PFbTagger.SoftPFMuonByPtBJetTags
akPu7PFNegativeSoftPFMuonByPtBJetTags = akPu7PFbTagger.NegativeSoftPFMuonByPtBJetTags
akPu7PFPositiveSoftPFMuonByPtBJetTags = akPu7PFbTagger.PositiveSoftPFMuonByPtBJetTags
akPu7PFPatJetFlavourIdLegacy = cms.Sequence(akPu7PFPatJetPartonAssociationLegacy*akPu7PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu7PFPatJetFlavourAssociation = akPu7PFbTagger.PatJetFlavourAssociation
#akPu7PFPatJetFlavourId = cms.Sequence(akPu7PFPatJetPartons*akPu7PFPatJetFlavourAssociation)

akPu7PFJetBtaggingIP       = cms.Sequence(akPu7PFImpactParameterTagInfos *
            (akPu7PFTrackCountingHighEffBJetTags +
             akPu7PFTrackCountingHighPurBJetTags +
             akPu7PFJetProbabilityBJetTags +
             akPu7PFJetBProbabilityBJetTags +
             akPu7PFPositiveOnlyJetProbabilityBJetTags +
             akPu7PFNegativeOnlyJetProbabilityBJetTags +
             akPu7PFNegativeTrackCountingHighEffBJetTags +
             akPu7PFNegativeTrackCountingHighPurBJetTags +
             akPu7PFNegativeOnlyJetBProbabilityBJetTags +
             akPu7PFPositiveOnlyJetBProbabilityBJetTags
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
                akPu7PFCombinedSecondaryVertexV2BJetTags
              )
            )

akPu7PFJetBtaggingNegSV = cms.Sequence(akPu7PFImpactParameterTagInfos
            *
            akPu7PFSecondaryVertexNegativeTagInfos
            * (akPu7PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu7PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu7PFNegativeCombinedSecondaryVertexBJetTags
                +
                akPu7PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu7PFJetBtaggingMu = cms.Sequence(akPu7PFSoftPFMuonsTagInfos * (akPu7PFSoftPFMuonBJetTags
                +
                akPu7PFSoftPFMuonByIP3dBJetTags
                +
                akPu7PFSoftPFMuonByPtBJetTags
                +
                akPu7PFNegativeSoftPFMuonByPtBJetTags
                +
                akPu7PFPositiveSoftPFMuonByPtBJetTags
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
        JetPartonMapSource   = cms.InputTag("akPu7PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu7PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu7PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu7PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu7PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu7PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu7PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu7PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu7PFJetProbabilityBJetTags"),
            cms.InputTag("akPu7PFSoftPFMuonByPtBJetTags"),
            cms.InputTag("akPu7PFSoftPFMuonByIP3dBJetTags"),
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
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akPu7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu7PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu7PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu7PFJetSequence_mc = cms.Sequence(
                                                  #akPu7PFclean
                                                  #*
                                                  akPu7PFmatch
                                                  *
                                                  akPu7PFparton
                                                  *
                                                  akPu7PFcorr
                                                  *
                                                  akPu7PFJetID
                                                  *
                                                  akPu7PFPatJetFlavourIdLegacy
                                                  #*
			                          #akPu7PFPatJetFlavourId  # Use legacy algo till PU implemented
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

akPu7PFJetSequence_jec = cms.Sequence(akPu7PFJetSequence_mc)
akPu7PFJetSequence_mix = cms.Sequence(akPu7PFJetSequence_mc)

akPu7PFJetSequence = cms.Sequence(akPu7PFJetSequence_data)
