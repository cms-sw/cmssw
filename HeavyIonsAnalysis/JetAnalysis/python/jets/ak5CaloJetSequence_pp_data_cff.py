

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

ak5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak5CaloJets"),
    matched = cms.InputTag("ak5GenJets"),
    maxDeltaR = 0.5
    )

ak5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak5CaloJets")
                                                        )

ak5Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak5CaloJets"),
    payload = "AK5Calo_offline"
    )

ak5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak5CaloJets'))

#ak5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5GenJets'))

ak5CalobTagger = bTaggers("ak5Calo",0.5)

#create objects locally since they dont load properly otherwise
#ak5Calomatch = ak5CalobTagger.match
ak5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak5CaloJets"), matched = cms.InputTag("genParticles"))
ak5CaloPatJetFlavourAssociationLegacy = ak5CalobTagger.PatJetFlavourAssociationLegacy
ak5CaloPatJetPartons = ak5CalobTagger.PatJetPartons
ak5CaloJetTracksAssociatorAtVertex = ak5CalobTagger.JetTracksAssociatorAtVertex
ak5CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak5CaloSimpleSecondaryVertexHighEffBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak5CaloSimpleSecondaryVertexHighPurBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak5CaloCombinedSecondaryVertexBJetTags = ak5CalobTagger.CombinedSecondaryVertexBJetTags
ak5CaloCombinedSecondaryVertexV2BJetTags = ak5CalobTagger.CombinedSecondaryVertexV2BJetTags
ak5CaloJetBProbabilityBJetTags = ak5CalobTagger.JetBProbabilityBJetTags
ak5CaloSoftPFMuonByPtBJetTags = ak5CalobTagger.SoftPFMuonByPtBJetTags
ak5CaloSoftPFMuonByIP3dBJetTags = ak5CalobTagger.SoftPFMuonByIP3dBJetTags
ak5CaloTrackCountingHighEffBJetTags = ak5CalobTagger.TrackCountingHighEffBJetTags
ak5CaloTrackCountingHighPurBJetTags = ak5CalobTagger.TrackCountingHighPurBJetTags
ak5CaloPatJetPartonAssociationLegacy = ak5CalobTagger.PatJetPartonAssociationLegacy

ak5CaloImpactParameterTagInfos = ak5CalobTagger.ImpactParameterTagInfos
ak5CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak5CaloJetProbabilityBJetTags = ak5CalobTagger.JetProbabilityBJetTags
ak5CaloPositiveOnlyJetProbabilityBJetTags = ak5CalobTagger.PositiveOnlyJetProbabilityBJetTags
ak5CaloNegativeOnlyJetProbabilityBJetTags = ak5CalobTagger.NegativeOnlyJetProbabilityBJetTags
ak5CaloNegativeTrackCountingHighEffBJetTags = ak5CalobTagger.NegativeTrackCountingHighEffBJetTags
ak5CaloNegativeTrackCountingHighPurBJetTags = ak5CalobTagger.NegativeTrackCountingHighPurBJetTags
ak5CaloNegativeOnlyJetBProbabilityBJetTags = ak5CalobTagger.NegativeOnlyJetBProbabilityBJetTags
ak5CaloPositiveOnlyJetBProbabilityBJetTags = ak5CalobTagger.PositiveOnlyJetBProbabilityBJetTags

ak5CaloSecondaryVertexTagInfos = ak5CalobTagger.SecondaryVertexTagInfos
ak5CaloSimpleSecondaryVertexHighEffBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak5CaloSimpleSecondaryVertexHighPurBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak5CaloCombinedSecondaryVertexBJetTags = ak5CalobTagger.CombinedSecondaryVertexBJetTags
ak5CaloCombinedSecondaryVertexV2BJetTags = ak5CalobTagger.CombinedSecondaryVertexV2BJetTags

ak5CaloSecondaryVertexNegativeTagInfos = ak5CalobTagger.SecondaryVertexNegativeTagInfos
ak5CaloNegativeSimpleSecondaryVertexHighEffBJetTags = ak5CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak5CaloNegativeSimpleSecondaryVertexHighPurBJetTags = ak5CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak5CaloNegativeCombinedSecondaryVertexBJetTags = ak5CalobTagger.NegativeCombinedSecondaryVertexBJetTags
ak5CaloPositiveCombinedSecondaryVertexBJetTags = ak5CalobTagger.PositiveCombinedSecondaryVertexBJetTags

ak5CaloSoftPFMuonsTagInfos = ak5CalobTagger.SoftPFMuonsTagInfos
ak5CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak5CaloSoftPFMuonBJetTags = ak5CalobTagger.SoftPFMuonBJetTags
ak5CaloSoftPFMuonByIP3dBJetTags = ak5CalobTagger.SoftPFMuonByIP3dBJetTags
ak5CaloSoftPFMuonByPtBJetTags = ak5CalobTagger.SoftPFMuonByPtBJetTags
ak5CaloNegativeSoftPFMuonByPtBJetTags = ak5CalobTagger.NegativeSoftPFMuonByPtBJetTags
ak5CaloPositiveSoftPFMuonByPtBJetTags = ak5CalobTagger.PositiveSoftPFMuonByPtBJetTags
ak5CaloPatJetFlavourIdLegacy = cms.Sequence(ak5CaloPatJetPartonAssociationLegacy*ak5CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak5CaloPatJetFlavourAssociation = ak5CalobTagger.PatJetFlavourAssociation
#ak5CaloPatJetFlavourId = cms.Sequence(ak5CaloPatJetPartons*ak5CaloPatJetFlavourAssociation)

ak5CaloJetBtaggingIP       = cms.Sequence(ak5CaloImpactParameterTagInfos *
            (ak5CaloTrackCountingHighEffBJetTags +
             ak5CaloTrackCountingHighPurBJetTags +
             ak5CaloJetProbabilityBJetTags +
             ak5CaloJetBProbabilityBJetTags +
             ak5CaloPositiveOnlyJetProbabilityBJetTags +
             ak5CaloNegativeOnlyJetProbabilityBJetTags +
             ak5CaloNegativeTrackCountingHighEffBJetTags +
             ak5CaloNegativeTrackCountingHighPurBJetTags +
             ak5CaloNegativeOnlyJetBProbabilityBJetTags +
             ak5CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak5CaloJetBtaggingSV = cms.Sequence(ak5CaloImpactParameterTagInfos
            *
            ak5CaloSecondaryVertexTagInfos
            * (ak5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak5CaloCombinedSecondaryVertexBJetTags
                +
                ak5CaloCombinedSecondaryVertexV2BJetTags
              )
            )

ak5CaloJetBtaggingNegSV = cms.Sequence(ak5CaloImpactParameterTagInfos
            *
            ak5CaloSecondaryVertexNegativeTagInfos
            * (ak5CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak5CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak5CaloNegativeCombinedSecondaryVertexBJetTags
                +
                ak5CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak5CaloJetBtaggingMu = cms.Sequence(ak5CaloSoftPFMuonsTagInfos * (ak5CaloSoftPFMuonBJetTags
                +
                ak5CaloSoftPFMuonByIP3dBJetTags
                +
                ak5CaloSoftPFMuonByPtBJetTags
                +
                ak5CaloNegativeSoftPFMuonByPtBJetTags
                +
                ak5CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

ak5CaloJetBtagging = cms.Sequence(ak5CaloJetBtaggingIP
            *ak5CaloJetBtaggingSV
            *ak5CaloJetBtaggingNegSV
#            *ak5CaloJetBtaggingMu
            )

ak5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak5CaloJets"),
        genJetMatch          = cms.InputTag("ak5Calomatch"),
        genPartonMatch       = cms.InputTag("ak5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak5Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak5CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak5CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak5CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak5CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak5CaloJetProbabilityBJetTags"),
            #cms.InputTag("ak5CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak5CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak5CaloJetID"),
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

ak5CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("ak5CaloJets"),
           	    R0  = cms.double( 0.5)
)
ak5CalopatJetsWithBtagging.userData.userFloats.src += ['ak5CaloNjettiness:tau1','ak5CaloNjettiness:tau2','ak5CaloNjettiness:tau3']

ak5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak5CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak5GenJets',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("ak5Calo"),
                                                             jetName = cms.untracked.string("ak5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

ak5CaloJetSequence_mc = cms.Sequence(
                                                  #ak5Caloclean
                                                  #*
                                                  ak5Calomatch
                                                  *
                                                  ak5Caloparton
                                                  *
                                                  ak5Calocorr
                                                  *
                                                  #ak5CaloJetID
                                                  #*
                                                  ak5CaloPatJetFlavourIdLegacy
                                                  #*
			                          #ak5CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak5CaloJetBtagging
                                                  *
                                                  ak5CaloNjettiness
                                                  *
                                                  ak5CalopatJetsWithBtagging
                                                  *
                                                  ak5CaloJetAnalyzer
                                                  )

ak5CaloJetSequence_data = cms.Sequence(ak5Calocorr
                                                    *
                                                    #ak5CaloJetID
                                                    #*
                                                    ak5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak5CaloJetBtagging
                                                    *
                                                    ak5CaloNjettiness 
                                                    *
                                                    ak5CalopatJetsWithBtagging
                                                    *
                                                    ak5CaloJetAnalyzer
                                                    )

ak5CaloJetSequence_jec = cms.Sequence(ak5CaloJetSequence_mc)
ak5CaloJetSequence_mix = cms.Sequence(ak5CaloJetSequence_mc)

ak5CaloJetSequence = cms.Sequence(ak5CaloJetSequence_data)
