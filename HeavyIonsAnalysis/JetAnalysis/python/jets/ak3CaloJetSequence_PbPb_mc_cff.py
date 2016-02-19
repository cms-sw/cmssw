

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

ak3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak3CaloJets"),
    matched = cms.InputTag("ak3HiGenJets"),
    maxDeltaR = 0.3
    )

ak3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak3CaloJets")
                                                        )

ak3Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak3CaloJets"),
    payload = "AK3Calo_offline"
    )

ak3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak3CaloJets'))

#ak3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJets'))

ak3CalobTagger = bTaggers("ak3Calo",0.3)

#create objects locally since they dont load properly otherwise
#ak3Calomatch = ak3CalobTagger.match
ak3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak3CaloJets"), matched = cms.InputTag("genParticles"))
ak3CaloPatJetFlavourAssociationLegacy = ak3CalobTagger.PatJetFlavourAssociationLegacy
ak3CaloPatJetPartons = ak3CalobTagger.PatJetPartons
ak3CaloJetTracksAssociatorAtVertex = ak3CalobTagger.JetTracksAssociatorAtVertex
ak3CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak3CaloSimpleSecondaryVertexHighEffBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak3CaloSimpleSecondaryVertexHighPurBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak3CaloCombinedSecondaryVertexBJetTags = ak3CalobTagger.CombinedSecondaryVertexBJetTags
ak3CaloCombinedSecondaryVertexV2BJetTags = ak3CalobTagger.CombinedSecondaryVertexV2BJetTags
ak3CaloJetBProbabilityBJetTags = ak3CalobTagger.JetBProbabilityBJetTags
ak3CaloSoftPFMuonByPtBJetTags = ak3CalobTagger.SoftPFMuonByPtBJetTags
ak3CaloSoftPFMuonByIP3dBJetTags = ak3CalobTagger.SoftPFMuonByIP3dBJetTags
ak3CaloTrackCountingHighEffBJetTags = ak3CalobTagger.TrackCountingHighEffBJetTags
ak3CaloTrackCountingHighPurBJetTags = ak3CalobTagger.TrackCountingHighPurBJetTags
ak3CaloPatJetPartonAssociationLegacy = ak3CalobTagger.PatJetPartonAssociationLegacy

ak3CaloImpactParameterTagInfos = ak3CalobTagger.ImpactParameterTagInfos
ak3CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak3CaloJetProbabilityBJetTags = ak3CalobTagger.JetProbabilityBJetTags
ak3CaloPositiveOnlyJetProbabilityBJetTags = ak3CalobTagger.PositiveOnlyJetProbabilityBJetTags
ak3CaloNegativeOnlyJetProbabilityBJetTags = ak3CalobTagger.NegativeOnlyJetProbabilityBJetTags
ak3CaloNegativeTrackCountingHighEffBJetTags = ak3CalobTagger.NegativeTrackCountingHighEffBJetTags
ak3CaloNegativeTrackCountingHighPurBJetTags = ak3CalobTagger.NegativeTrackCountingHighPurBJetTags
ak3CaloNegativeOnlyJetBProbabilityBJetTags = ak3CalobTagger.NegativeOnlyJetBProbabilityBJetTags
ak3CaloPositiveOnlyJetBProbabilityBJetTags = ak3CalobTagger.PositiveOnlyJetBProbabilityBJetTags

ak3CaloSecondaryVertexTagInfos = ak3CalobTagger.SecondaryVertexTagInfos
ak3CaloSimpleSecondaryVertexHighEffBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak3CaloSimpleSecondaryVertexHighPurBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak3CaloCombinedSecondaryVertexBJetTags = ak3CalobTagger.CombinedSecondaryVertexBJetTags
ak3CaloCombinedSecondaryVertexV2BJetTags = ak3CalobTagger.CombinedSecondaryVertexV2BJetTags

ak3CaloSecondaryVertexNegativeTagInfos = ak3CalobTagger.SecondaryVertexNegativeTagInfos
ak3CaloNegativeSimpleSecondaryVertexHighEffBJetTags = ak3CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak3CaloNegativeSimpleSecondaryVertexHighPurBJetTags = ak3CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak3CaloNegativeCombinedSecondaryVertexBJetTags = ak3CalobTagger.NegativeCombinedSecondaryVertexBJetTags
ak3CaloPositiveCombinedSecondaryVertexBJetTags = ak3CalobTagger.PositiveCombinedSecondaryVertexBJetTags

ak3CaloSoftPFMuonsTagInfos = ak3CalobTagger.SoftPFMuonsTagInfos
ak3CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak3CaloSoftPFMuonBJetTags = ak3CalobTagger.SoftPFMuonBJetTags
ak3CaloSoftPFMuonByIP3dBJetTags = ak3CalobTagger.SoftPFMuonByIP3dBJetTags
ak3CaloSoftPFMuonByPtBJetTags = ak3CalobTagger.SoftPFMuonByPtBJetTags
ak3CaloNegativeSoftPFMuonByPtBJetTags = ak3CalobTagger.NegativeSoftPFMuonByPtBJetTags
ak3CaloPositiveSoftPFMuonByPtBJetTags = ak3CalobTagger.PositiveSoftPFMuonByPtBJetTags
ak3CaloPatJetFlavourIdLegacy = cms.Sequence(ak3CaloPatJetPartonAssociationLegacy*ak3CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak3CaloPatJetFlavourAssociation = ak3CalobTagger.PatJetFlavourAssociation
#ak3CaloPatJetFlavourId = cms.Sequence(ak3CaloPatJetPartons*ak3CaloPatJetFlavourAssociation)

ak3CaloJetBtaggingIP       = cms.Sequence(ak3CaloImpactParameterTagInfos *
            (ak3CaloTrackCountingHighEffBJetTags +
             ak3CaloTrackCountingHighPurBJetTags +
             ak3CaloJetProbabilityBJetTags +
             ak3CaloJetBProbabilityBJetTags +
             ak3CaloPositiveOnlyJetProbabilityBJetTags +
             ak3CaloNegativeOnlyJetProbabilityBJetTags +
             ak3CaloNegativeTrackCountingHighEffBJetTags +
             ak3CaloNegativeTrackCountingHighPurBJetTags +
             ak3CaloNegativeOnlyJetBProbabilityBJetTags +
             ak3CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak3CaloJetBtaggingSV = cms.Sequence(ak3CaloImpactParameterTagInfos
            *
            ak3CaloSecondaryVertexTagInfos
            * (ak3CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak3CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak3CaloCombinedSecondaryVertexBJetTags
                +
                ak3CaloCombinedSecondaryVertexV2BJetTags
              )
            )

ak3CaloJetBtaggingNegSV = cms.Sequence(ak3CaloImpactParameterTagInfos
            *
            ak3CaloSecondaryVertexNegativeTagInfos
            * (ak3CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak3CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak3CaloNegativeCombinedSecondaryVertexBJetTags
                +
                ak3CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak3CaloJetBtaggingMu = cms.Sequence(ak3CaloSoftPFMuonsTagInfos * (ak3CaloSoftPFMuonBJetTags
                +
                ak3CaloSoftPFMuonByIP3dBJetTags
                +
                ak3CaloSoftPFMuonByPtBJetTags
                +
                ak3CaloNegativeSoftPFMuonByPtBJetTags
                +
                ak3CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

ak3CaloJetBtagging = cms.Sequence(ak3CaloJetBtaggingIP
            *ak3CaloJetBtaggingSV
            *ak3CaloJetBtaggingNegSV
#            *ak3CaloJetBtaggingMu
            )

ak3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak3CaloJets"),
        genJetMatch          = cms.InputTag("ak3Calomatch"),
        genPartonMatch       = cms.InputTag("ak3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak3Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak3CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak3CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak3CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak3CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak3CaloJetProbabilityBJetTags"),
            #cms.InputTag("ak3CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak3CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak3CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

ak3CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("ak3CaloJets"),
           	    R0  = cms.double( 0.3)
)
ak3CalopatJetsWithBtagging.userData.userFloats.src += ['ak3CaloNjettiness:tau1','ak3CaloNjettiness:tau2','ak3CaloNjettiness:tau3']

ak3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("ak3Calo"),
                                                             jetName = cms.untracked.string("ak3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

ak3CaloJetSequence_mc = cms.Sequence(
                                                  #ak3Caloclean
                                                  #*
                                                  ak3Calomatch
                                                  *
                                                  ak3Caloparton
                                                  *
                                                  ak3Calocorr
                                                  *
                                                  #ak3CaloJetID
                                                  #*
                                                  ak3CaloPatJetFlavourIdLegacy
                                                  #*
			                          #ak3CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak3CaloJetBtagging
                                                  *
                                                  ak3CaloNjettiness
                                                  *
                                                  ak3CalopatJetsWithBtagging
                                                  *
                                                  ak3CaloJetAnalyzer
                                                  )

ak3CaloJetSequence_data = cms.Sequence(ak3Calocorr
                                                    *
                                                    #ak3CaloJetID
                                                    #*
                                                    ak3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak3CaloJetBtagging
                                                    *
                                                    ak3CaloNjettiness 
                                                    *
                                                    ak3CalopatJetsWithBtagging
                                                    *
                                                    ak3CaloJetAnalyzer
                                                    )

ak3CaloJetSequence_jec = cms.Sequence(ak3CaloJetSequence_mc)
ak3CaloJetSequence_mix = cms.Sequence(ak3CaloJetSequence_mc)

ak3CaloJetSequence = cms.Sequence(ak3CaloJetSequence_mc)
