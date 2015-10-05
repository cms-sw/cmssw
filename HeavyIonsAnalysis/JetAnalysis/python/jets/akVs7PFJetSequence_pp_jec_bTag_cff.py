

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs7PFJets"),
    matched = cms.InputTag("ak7HiGenJets"),
    maxDeltaR = 0.7
    )

akVs7PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs7PFJets")
                                                        )

akVs7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs7PFJets"),
    payload = "AKVs7PF_generalTracks"
    )

akVs7PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs7CaloJets'))

#akVs7PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

akVs7PFbTagger = bTaggers("akVs7PF",0.7)

#create objects locally since they dont load properly otherwise
#akVs7PFmatch = akVs7PFbTagger.match
akVs7PFparton = akVs7PFbTagger.parton
akVs7PFPatJetFlavourAssociationLegacy = akVs7PFbTagger.PatJetFlavourAssociationLegacy
akVs7PFPatJetPartons = akVs7PFbTagger.PatJetPartons
akVs7PFJetTracksAssociatorAtVertex = akVs7PFbTagger.JetTracksAssociatorAtVertex
akVs7PFSimpleSecondaryVertexHighEffBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7PFSimpleSecondaryVertexHighPurBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7PFCombinedSecondaryVertexBJetTags = akVs7PFbTagger.CombinedSecondaryVertexBJetTags
akVs7PFCombinedSecondaryVertexV2BJetTags = akVs7PFbTagger.CombinedSecondaryVertexV2BJetTags
akVs7PFJetBProbabilityBJetTags = akVs7PFbTagger.JetBProbabilityBJetTags
akVs7PFSoftPFMuonByPtBJetTags = akVs7PFbTagger.SoftPFMuonByPtBJetTags
akVs7PFSoftPFMuonByIP3dBJetTags = akVs7PFbTagger.SoftPFMuonByIP3dBJetTags
akVs7PFTrackCountingHighEffBJetTags = akVs7PFbTagger.TrackCountingHighEffBJetTags
akVs7PFTrackCountingHighPurBJetTags = akVs7PFbTagger.TrackCountingHighPurBJetTags
akVs7PFPatJetPartonAssociationLegacy = akVs7PFbTagger.PatJetPartonAssociationLegacy

akVs7PFImpactParameterTagInfos = akVs7PFbTagger.ImpactParameterTagInfos
akVs7PFJetProbabilityBJetTags = akVs7PFbTagger.JetProbabilityBJetTags
akVs7PFPositiveOnlyJetProbabilityBJetTags = akVs7PFbTagger.PositiveOnlyJetProbabilityBJetTags
akVs7PFNegativeOnlyJetProbabilityBJetTags = akVs7PFbTagger.NegativeOnlyJetProbabilityBJetTags
akVs7PFNegativeTrackCountingHighEffBJetTags = akVs7PFbTagger.NegativeTrackCountingHighEffBJetTags
akVs7PFNegativeTrackCountingHighPurBJetTags = akVs7PFbTagger.NegativeTrackCountingHighPurBJetTags
akVs7PFNegativeOnlyJetBProbabilityBJetTags = akVs7PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akVs7PFPositiveOnlyJetBProbabilityBJetTags = akVs7PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akVs7PFSecondaryVertexTagInfos = akVs7PFbTagger.SecondaryVertexTagInfos
akVs7PFSimpleSecondaryVertexHighEffBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7PFSimpleSecondaryVertexHighPurBJetTags = akVs7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7PFCombinedSecondaryVertexBJetTags = akVs7PFbTagger.CombinedSecondaryVertexBJetTags
akVs7PFCombinedSecondaryVertexV2BJetTags = akVs7PFbTagger.CombinedSecondaryVertexV2BJetTags

akVs7PFSecondaryVertexNegativeTagInfos = akVs7PFbTagger.SecondaryVertexNegativeTagInfos
akVs7PFNegativeSimpleSecondaryVertexHighEffBJetTags = akVs7PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs7PFNegativeSimpleSecondaryVertexHighPurBJetTags = akVs7PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs7PFNegativeCombinedSecondaryVertexBJetTags = akVs7PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akVs7PFPositiveCombinedSecondaryVertexBJetTags = akVs7PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akVs7PFSoftPFMuonsTagInfos = akVs7PFbTagger.SoftPFMuonsTagInfos
akVs7PFSoftPFMuonBJetTags = akVs7PFbTagger.SoftPFMuonBJetTags
akVs7PFSoftPFMuonByIP3dBJetTags = akVs7PFbTagger.SoftPFMuonByIP3dBJetTags
akVs7PFSoftPFMuonByPtBJetTags = akVs7PFbTagger.SoftPFMuonByPtBJetTags
akVs7PFNegativeSoftPFMuonByPtBJetTags = akVs7PFbTagger.NegativeSoftPFMuonByPtBJetTags
akVs7PFPositiveSoftPFMuonByPtBJetTags = akVs7PFbTagger.PositiveSoftPFMuonByPtBJetTags
akVs7PFPatJetFlavourIdLegacy = cms.Sequence(akVs7PFPatJetPartonAssociationLegacy*akVs7PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs7PFPatJetFlavourAssociation = akVs7PFbTagger.PatJetFlavourAssociation
#akVs7PFPatJetFlavourId = cms.Sequence(akVs7PFPatJetPartons*akVs7PFPatJetFlavourAssociation)

akVs7PFJetBtaggingIP       = cms.Sequence(akVs7PFImpactParameterTagInfos *
            (akVs7PFTrackCountingHighEffBJetTags +
             akVs7PFTrackCountingHighPurBJetTags +
             akVs7PFJetProbabilityBJetTags +
             akVs7PFJetBProbabilityBJetTags +
             akVs7PFPositiveOnlyJetProbabilityBJetTags +
             akVs7PFNegativeOnlyJetProbabilityBJetTags +
             akVs7PFNegativeTrackCountingHighEffBJetTags +
             akVs7PFNegativeTrackCountingHighPurBJetTags +
             akVs7PFNegativeOnlyJetBProbabilityBJetTags +
             akVs7PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs7PFJetBtaggingSV = cms.Sequence(akVs7PFImpactParameterTagInfos
            *
            akVs7PFSecondaryVertexTagInfos
            * (akVs7PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs7PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs7PFCombinedSecondaryVertexBJetTags
                +
                akVs7PFCombinedSecondaryVertexV2BJetTags
              )
            )

akVs7PFJetBtaggingNegSV = cms.Sequence(akVs7PFImpactParameterTagInfos
            *
            akVs7PFSecondaryVertexNegativeTagInfos
            * (akVs7PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs7PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs7PFNegativeCombinedSecondaryVertexBJetTags
                +
                akVs7PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs7PFJetBtaggingMu = cms.Sequence(akVs7PFSoftPFMuonsTagInfos * (akVs7PFSoftPFMuonBJetTags
                +
                akVs7PFSoftPFMuonByIP3dBJetTags
                +
                akVs7PFSoftPFMuonByPtBJetTags
                +
                akVs7PFNegativeSoftPFMuonByPtBJetTags
                +
                akVs7PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs7PFJetBtagging = cms.Sequence(akVs7PFJetBtaggingIP
            *akVs7PFJetBtaggingSV
            *akVs7PFJetBtaggingNegSV
            *akVs7PFJetBtaggingMu
            )

akVs7PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs7PFJets"),
        genJetMatch          = cms.InputTag("akVs7PFmatch"),
        genPartonMatch       = cms.InputTag("akVs7PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs7PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs7PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs7PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs7PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs7PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs7PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs7PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs7PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs7PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs7PFJetProbabilityBJetTags"),
            cms.InputTag("akVs7PFSoftPFMuonByPtBJetTags"),
            cms.InputTag("akVs7PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs7PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs7PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs7PFJetID"),
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
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akVs7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs7PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs7PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs7PFJetSequence_mc = cms.Sequence(
                                                  #akVs7PFclean
                                                  #*
                                                  akVs7PFmatch
                                                  *
                                                  akVs7PFparton
                                                  *
                                                  akVs7PFcorr
                                                  *
                                                  akVs7PFJetID
                                                  *
                                                  akVs7PFPatJetFlavourIdLegacy
                                                  #*
			                          #akVs7PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs7PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs7PFJetBtagging
                                                  *
                                                  akVs7PFpatJetsWithBtagging
                                                  *
                                                  akVs7PFJetAnalyzer
                                                  )

akVs7PFJetSequence_data = cms.Sequence(akVs7PFcorr
                                                    *
                                                    akVs7PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs7PFJetBtagging
                                                    *
                                                    akVs7PFpatJetsWithBtagging
                                                    *
                                                    akVs7PFJetAnalyzer
                                                    )

akVs7PFJetSequence_jec = cms.Sequence(akVs7PFJetSequence_mc)
akVs7PFJetSequence_mix = cms.Sequence(akVs7PFJetSequence_mc)

akVs7PFJetSequence = cms.Sequence(akVs7PFJetSequence_jec)
akVs7PFJetAnalyzer.genPtMin = cms.untracked.double(1)
