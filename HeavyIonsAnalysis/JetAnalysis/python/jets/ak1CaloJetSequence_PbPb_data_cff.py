

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

ak1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets"),
    maxDeltaR = 0.1
    )

ak1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak1CaloJets")
                                                        )

ak1Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak1CaloJets"),
    payload = "AK1Calo_offline"
    )

ak1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak1CaloJets'))

#ak1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJets'))

ak1CalobTagger = bTaggers("ak1Calo",0.1)

#create objects locally since they dont load properly otherwise
#ak1Calomatch = ak1CalobTagger.match
ak1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak1CaloJets"), matched = cms.InputTag("genParticles"))
ak1CaloPatJetFlavourAssociationLegacy = ak1CalobTagger.PatJetFlavourAssociationLegacy
ak1CaloPatJetPartons = ak1CalobTagger.PatJetPartons
ak1CaloJetTracksAssociatorAtVertex = ak1CalobTagger.JetTracksAssociatorAtVertex
ak1CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak1CaloSimpleSecondaryVertexHighEffBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak1CaloSimpleSecondaryVertexHighPurBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak1CaloCombinedSecondaryVertexBJetTags = ak1CalobTagger.CombinedSecondaryVertexBJetTags
ak1CaloCombinedSecondaryVertexV2BJetTags = ak1CalobTagger.CombinedSecondaryVertexV2BJetTags
ak1CaloJetBProbabilityBJetTags = ak1CalobTagger.JetBProbabilityBJetTags
ak1CaloSoftPFMuonByPtBJetTags = ak1CalobTagger.SoftPFMuonByPtBJetTags
ak1CaloSoftPFMuonByIP3dBJetTags = ak1CalobTagger.SoftPFMuonByIP3dBJetTags
ak1CaloTrackCountingHighEffBJetTags = ak1CalobTagger.TrackCountingHighEffBJetTags
ak1CaloTrackCountingHighPurBJetTags = ak1CalobTagger.TrackCountingHighPurBJetTags
ak1CaloPatJetPartonAssociationLegacy = ak1CalobTagger.PatJetPartonAssociationLegacy

ak1CaloImpactParameterTagInfos = ak1CalobTagger.ImpactParameterTagInfos
ak1CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak1CaloJetProbabilityBJetTags = ak1CalobTagger.JetProbabilityBJetTags
ak1CaloPositiveOnlyJetProbabilityBJetTags = ak1CalobTagger.PositiveOnlyJetProbabilityBJetTags
ak1CaloNegativeOnlyJetProbabilityBJetTags = ak1CalobTagger.NegativeOnlyJetProbabilityBJetTags
ak1CaloNegativeTrackCountingHighEffBJetTags = ak1CalobTagger.NegativeTrackCountingHighEffBJetTags
ak1CaloNegativeTrackCountingHighPurBJetTags = ak1CalobTagger.NegativeTrackCountingHighPurBJetTags
ak1CaloNegativeOnlyJetBProbabilityBJetTags = ak1CalobTagger.NegativeOnlyJetBProbabilityBJetTags
ak1CaloPositiveOnlyJetBProbabilityBJetTags = ak1CalobTagger.PositiveOnlyJetBProbabilityBJetTags

ak1CaloSecondaryVertexTagInfos = ak1CalobTagger.SecondaryVertexTagInfos
ak1CaloSimpleSecondaryVertexHighEffBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak1CaloSimpleSecondaryVertexHighPurBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak1CaloCombinedSecondaryVertexBJetTags = ak1CalobTagger.CombinedSecondaryVertexBJetTags
ak1CaloCombinedSecondaryVertexV2BJetTags = ak1CalobTagger.CombinedSecondaryVertexV2BJetTags

ak1CaloSecondaryVertexNegativeTagInfos = ak1CalobTagger.SecondaryVertexNegativeTagInfos
ak1CaloNegativeSimpleSecondaryVertexHighEffBJetTags = ak1CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak1CaloNegativeSimpleSecondaryVertexHighPurBJetTags = ak1CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak1CaloNegativeCombinedSecondaryVertexBJetTags = ak1CalobTagger.NegativeCombinedSecondaryVertexBJetTags
ak1CaloPositiveCombinedSecondaryVertexBJetTags = ak1CalobTagger.PositiveCombinedSecondaryVertexBJetTags

ak1CaloSoftPFMuonsTagInfos = ak1CalobTagger.SoftPFMuonsTagInfos
ak1CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak1CaloSoftPFMuonBJetTags = ak1CalobTagger.SoftPFMuonBJetTags
ak1CaloSoftPFMuonByIP3dBJetTags = ak1CalobTagger.SoftPFMuonByIP3dBJetTags
ak1CaloSoftPFMuonByPtBJetTags = ak1CalobTagger.SoftPFMuonByPtBJetTags
ak1CaloNegativeSoftPFMuonByPtBJetTags = ak1CalobTagger.NegativeSoftPFMuonByPtBJetTags
ak1CaloPositiveSoftPFMuonByPtBJetTags = ak1CalobTagger.PositiveSoftPFMuonByPtBJetTags
ak1CaloPatJetFlavourIdLegacy = cms.Sequence(ak1CaloPatJetPartonAssociationLegacy*ak1CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak1CaloPatJetFlavourAssociation = ak1CalobTagger.PatJetFlavourAssociation
#ak1CaloPatJetFlavourId = cms.Sequence(ak1CaloPatJetPartons*ak1CaloPatJetFlavourAssociation)

ak1CaloJetBtaggingIP       = cms.Sequence(ak1CaloImpactParameterTagInfos *
            (ak1CaloTrackCountingHighEffBJetTags +
             ak1CaloTrackCountingHighPurBJetTags +
             ak1CaloJetProbabilityBJetTags +
             ak1CaloJetBProbabilityBJetTags +
             ak1CaloPositiveOnlyJetProbabilityBJetTags +
             ak1CaloNegativeOnlyJetProbabilityBJetTags +
             ak1CaloNegativeTrackCountingHighEffBJetTags +
             ak1CaloNegativeTrackCountingHighPurBJetTags +
             ak1CaloNegativeOnlyJetBProbabilityBJetTags +
             ak1CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak1CaloJetBtaggingSV = cms.Sequence(ak1CaloImpactParameterTagInfos
            *
            ak1CaloSecondaryVertexTagInfos
            * (ak1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak1CaloCombinedSecondaryVertexBJetTags
                +
                ak1CaloCombinedSecondaryVertexV2BJetTags
              )
            )

ak1CaloJetBtaggingNegSV = cms.Sequence(ak1CaloImpactParameterTagInfos
            *
            ak1CaloSecondaryVertexNegativeTagInfos
            * (ak1CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak1CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak1CaloNegativeCombinedSecondaryVertexBJetTags
                +
                ak1CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak1CaloJetBtaggingMu = cms.Sequence(ak1CaloSoftPFMuonsTagInfos * (ak1CaloSoftPFMuonBJetTags
                +
                ak1CaloSoftPFMuonByIP3dBJetTags
                +
                ak1CaloSoftPFMuonByPtBJetTags
                +
                ak1CaloNegativeSoftPFMuonByPtBJetTags
                +
                ak1CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

ak1CaloJetBtagging = cms.Sequence(ak1CaloJetBtaggingIP
            *ak1CaloJetBtaggingSV
            *ak1CaloJetBtaggingNegSV
#            *ak1CaloJetBtaggingMu
            )

ak1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak1CaloJets"),
        genJetMatch          = cms.InputTag("ak1Calomatch"),
        genPartonMatch       = cms.InputTag("ak1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak1Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak1CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak1CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak1CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak1CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak1CaloJetProbabilityBJetTags"),
            #cms.InputTag("ak1CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak1CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak1CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

ak1CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("ak1CaloJets"),
           	    R0  = cms.double( 0.1)
)
ak1CalopatJetsWithBtagging.userData.userFloats.src += ['ak1CaloNjettiness:tau1','ak1CaloNjettiness:tau2','ak1CaloNjettiness:tau3']

ak1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("ak1Calo"),
                                                             jetName = cms.untracked.string("ak1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

ak1CaloJetSequence_mc = cms.Sequence(
                                                  #ak1Caloclean
                                                  #*
                                                  ak1Calomatch
                                                  *
                                                  ak1Caloparton
                                                  *
                                                  ak1Calocorr
                                                  *
                                                  #ak1CaloJetID
                                                  #*
                                                  ak1CaloPatJetFlavourIdLegacy
                                                  #*
			                          #ak1CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak1CaloJetBtagging
                                                  *
                                                  ak1CaloNjettiness
                                                  *
                                                  ak1CalopatJetsWithBtagging
                                                  *
                                                  ak1CaloJetAnalyzer
                                                  )

ak1CaloJetSequence_data = cms.Sequence(ak1Calocorr
                                                    *
                                                    #ak1CaloJetID
                                                    #*
                                                    ak1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak1CaloJetBtagging
                                                    *
                                                    ak1CaloNjettiness 
                                                    *
                                                    ak1CalopatJetsWithBtagging
                                                    *
                                                    ak1CaloJetAnalyzer
                                                    )

ak1CaloJetSequence_jec = cms.Sequence(ak1CaloJetSequence_mc)
ak1CaloJetSequence_mix = cms.Sequence(ak1CaloJetSequence_mc)

ak1CaloJetSequence = cms.Sequence(ak1CaloJetSequence_data)
