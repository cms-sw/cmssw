

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop2CaloJets"),
    matched = cms.InputTag("ak2GenJets"),
    maxDeltaR = 0.2
    )

akCsSoftDrop2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop2CaloJets")
                                                        )

akCsSoftDrop2Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop2CaloJets"),
    payload = "AK2Calo_offline"
    )

akCsSoftDrop2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop2CaloJets'))

#akCsSoftDrop2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2GenJets'))

akCsSoftDrop2CalobTagger = bTaggers("akCsSoftDrop2Calo",0.2)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop2Calomatch = akCsSoftDrop2CalobTagger.match
akCsSoftDrop2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop2CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop2CaloPatJetFlavourAssociationLegacy = akCsSoftDrop2CalobTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop2CaloPatJetPartons = akCsSoftDrop2CalobTagger.PatJetPartons
akCsSoftDrop2CaloJetTracksAssociatorAtVertex = akCsSoftDrop2CalobTagger.JetTracksAssociatorAtVertex
akCsSoftDrop2CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop2CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop2CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop2CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop2CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop2CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop2CaloJetBProbabilityBJetTags = akCsSoftDrop2CalobTagger.JetBProbabilityBJetTags
akCsSoftDrop2CaloSoftPFMuonByPtBJetTags = akCsSoftDrop2CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop2CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop2CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop2CaloTrackCountingHighEffBJetTags = akCsSoftDrop2CalobTagger.TrackCountingHighEffBJetTags
akCsSoftDrop2CaloTrackCountingHighPurBJetTags = akCsSoftDrop2CalobTagger.TrackCountingHighPurBJetTags
akCsSoftDrop2CaloPatJetPartonAssociationLegacy = akCsSoftDrop2CalobTagger.PatJetPartonAssociationLegacy

akCsSoftDrop2CaloImpactParameterTagInfos = akCsSoftDrop2CalobTagger.ImpactParameterTagInfos
akCsSoftDrop2CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop2CaloJetProbabilityBJetTags = akCsSoftDrop2CalobTagger.JetProbabilityBJetTags

akCsSoftDrop2CaloSecondaryVertexTagInfos = akCsSoftDrop2CalobTagger.SecondaryVertexTagInfos
akCsSoftDrop2CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop2CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop2CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop2CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop2CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop2CaloSecondaryVertexNegativeTagInfos = akCsSoftDrop2CalobTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop2CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop2CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop2CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop2CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop2CaloNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop2CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop2CaloPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop2CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop2CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop2CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop2CaloSoftPFMuonsTagInfos = akCsSoftDrop2CalobTagger.SoftPFMuonsTagInfos
akCsSoftDrop2CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop2CaloSoftPFMuonBJetTags = akCsSoftDrop2CalobTagger.SoftPFMuonBJetTags
akCsSoftDrop2CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop2CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop2CaloSoftPFMuonByPtBJetTags = akCsSoftDrop2CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop2CaloNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop2CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop2CaloPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop2CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop2CaloPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop2CaloPatJetPartonAssociationLegacy*akCsSoftDrop2CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop2CaloPatJetFlavourAssociation = akCsSoftDrop2CalobTagger.PatJetFlavourAssociation
#akCsSoftDrop2CaloPatJetFlavourId = cms.Sequence(akCsSoftDrop2CaloPatJetPartons*akCsSoftDrop2CaloPatJetFlavourAssociation)

akCsSoftDrop2CaloJetBtaggingIP       = cms.Sequence(akCsSoftDrop2CaloImpactParameterTagInfos *
            (akCsSoftDrop2CaloTrackCountingHighEffBJetTags +
             akCsSoftDrop2CaloTrackCountingHighPurBJetTags +
             akCsSoftDrop2CaloJetProbabilityBJetTags +
             akCsSoftDrop2CaloJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop2CaloJetBtaggingSV = cms.Sequence(akCsSoftDrop2CaloImpactParameterTagInfos
            *
            akCsSoftDrop2CaloSecondaryVertexTagInfos
            * (akCsSoftDrop2CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop2CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop2CaloCombinedSecondaryVertexBJetTags+
                akCsSoftDrop2CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop2CaloJetBtaggingNegSV = cms.Sequence(akCsSoftDrop2CaloImpactParameterTagInfos
            *
            akCsSoftDrop2CaloSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop2CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop2CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop2CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop2CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop2CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop2CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop2CaloJetBtaggingMu = cms.Sequence(akCsSoftDrop2CaloSoftPFMuonsTagInfos * (akCsSoftDrop2CaloSoftPFMuonBJetTags
                +
                akCsSoftDrop2CaloSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop2CaloSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop2CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop2CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop2CaloJetBtagging = cms.Sequence(akCsSoftDrop2CaloJetBtaggingIP
            *akCsSoftDrop2CaloJetBtaggingSV
            *akCsSoftDrop2CaloJetBtaggingNegSV
#            *akCsSoftDrop2CaloJetBtaggingMu
            )

akCsSoftDrop2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop2CaloJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop2Calomatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop2CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop2CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop2CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop2CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop2CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop2CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop2CaloJetID"),
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

akCsSoftDrop2CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop2CaloJets"),
           	    R0  = cms.double( 0.2)
)
akCsSoftDrop2CalopatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop2CaloNjettiness:tau1','akCsSoftDrop2CaloNjettiness:tau2','akCsSoftDrop2CaloNjettiness:tau3']

akCsSoftDrop2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2GenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop2Calo"),
                                                             jetName = cms.untracked.string("akCsSoftDrop2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop2CaloJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop2Caloclean
                                                  #*
                                                  akCsSoftDrop2Calomatch
                                                  *
                                                  akCsSoftDrop2Caloparton
                                                  *
                                                  akCsSoftDrop2Calocorr
                                                  *
                                                  #akCsSoftDrop2CaloJetID
                                                  #*
                                                  akCsSoftDrop2CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop2CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop2CaloJetBtagging
                                                  *
                                                  akCsSoftDrop2CaloNjettiness
                                                  *
                                                  akCsSoftDrop2CalopatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop2CaloJetAnalyzer
                                                  )

akCsSoftDrop2CaloJetSequence_data = cms.Sequence(akCsSoftDrop2Calocorr
                                                    *
                                                    #akCsSoftDrop2CaloJetID
                                                    #*
                                                    akCsSoftDrop2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop2CaloJetBtagging
                                                    *
                                                    akCsSoftDrop2CaloNjettiness 
                                                    *
                                                    akCsSoftDrop2CalopatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop2CaloJetAnalyzer
                                                    )

akCsSoftDrop2CaloJetSequence_jec = cms.Sequence(akCsSoftDrop2CaloJetSequence_mc)
akCsSoftDrop2CaloJetSequence_mb = cms.Sequence(akCsSoftDrop2CaloJetSequence_mc)

akCsSoftDrop2CaloJetSequence = cms.Sequence(akCsSoftDrop2CaloJetSequence_mc)
