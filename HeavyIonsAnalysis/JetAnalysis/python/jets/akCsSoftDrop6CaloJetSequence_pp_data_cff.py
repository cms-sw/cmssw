

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop6CaloJets"),
    matched = cms.InputTag("ak6GenJets"),
    maxDeltaR = 0.6
    )

akCsSoftDrop6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop6CaloJets")
                                                        )

akCsSoftDrop6Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop6CaloJets"),
    payload = "AK6Calo_offline"
    )

akCsSoftDrop6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop6CaloJets'))

#akCsSoftDrop6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6GenJets'))

akCsSoftDrop6CalobTagger = bTaggers("akCsSoftDrop6Calo",0.6)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop6Calomatch = akCsSoftDrop6CalobTagger.match
akCsSoftDrop6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop6CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop6CaloPatJetFlavourAssociationLegacy = akCsSoftDrop6CalobTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop6CaloPatJetPartons = akCsSoftDrop6CalobTagger.PatJetPartons
akCsSoftDrop6CaloJetTracksAssociatorAtVertex = akCsSoftDrop6CalobTagger.JetTracksAssociatorAtVertex
akCsSoftDrop6CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop6CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop6CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop6CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop6CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop6CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop6CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop6CaloJetBProbabilityBJetTags = akCsSoftDrop6CalobTagger.JetBProbabilityBJetTags
akCsSoftDrop6CaloSoftPFMuonByPtBJetTags = akCsSoftDrop6CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop6CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop6CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop6CaloTrackCountingHighEffBJetTags = akCsSoftDrop6CalobTagger.TrackCountingHighEffBJetTags
akCsSoftDrop6CaloTrackCountingHighPurBJetTags = akCsSoftDrop6CalobTagger.TrackCountingHighPurBJetTags
akCsSoftDrop6CaloPatJetPartonAssociationLegacy = akCsSoftDrop6CalobTagger.PatJetPartonAssociationLegacy

akCsSoftDrop6CaloImpactParameterTagInfos = akCsSoftDrop6CalobTagger.ImpactParameterTagInfos
akCsSoftDrop6CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop6CaloJetProbabilityBJetTags = akCsSoftDrop6CalobTagger.JetProbabilityBJetTags

akCsSoftDrop6CaloSecondaryVertexTagInfos = akCsSoftDrop6CalobTagger.SecondaryVertexTagInfos
akCsSoftDrop6CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop6CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop6CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop6CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop6CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop6CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop6CaloSecondaryVertexNegativeTagInfos = akCsSoftDrop6CalobTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop6CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop6CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop6CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop6CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop6CaloNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop6CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop6CaloPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop6CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop6CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop6CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop6CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop6CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop6CaloSoftPFMuonsTagInfos = akCsSoftDrop6CalobTagger.SoftPFMuonsTagInfos
akCsSoftDrop6CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop6CaloSoftPFMuonBJetTags = akCsSoftDrop6CalobTagger.SoftPFMuonBJetTags
akCsSoftDrop6CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop6CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop6CaloSoftPFMuonByPtBJetTags = akCsSoftDrop6CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop6CaloNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop6CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop6CaloPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop6CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop6CaloPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop6CaloPatJetPartonAssociationLegacy*akCsSoftDrop6CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop6CaloPatJetFlavourAssociation = akCsSoftDrop6CalobTagger.PatJetFlavourAssociation
#akCsSoftDrop6CaloPatJetFlavourId = cms.Sequence(akCsSoftDrop6CaloPatJetPartons*akCsSoftDrop6CaloPatJetFlavourAssociation)

akCsSoftDrop6CaloJetBtaggingIP       = cms.Sequence(akCsSoftDrop6CaloImpactParameterTagInfos *
            (akCsSoftDrop6CaloTrackCountingHighEffBJetTags +
             akCsSoftDrop6CaloTrackCountingHighPurBJetTags +
             akCsSoftDrop6CaloJetProbabilityBJetTags +
             akCsSoftDrop6CaloJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop6CaloJetBtaggingSV = cms.Sequence(akCsSoftDrop6CaloImpactParameterTagInfos
            *
            akCsSoftDrop6CaloSecondaryVertexTagInfos
            * (akCsSoftDrop6CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop6CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop6CaloCombinedSecondaryVertexBJetTags+
                akCsSoftDrop6CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop6CaloJetBtaggingNegSV = cms.Sequence(akCsSoftDrop6CaloImpactParameterTagInfos
            *
            akCsSoftDrop6CaloSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop6CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop6CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop6CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop6CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop6CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop6CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop6CaloJetBtaggingMu = cms.Sequence(akCsSoftDrop6CaloSoftPFMuonsTagInfos * (akCsSoftDrop6CaloSoftPFMuonBJetTags
                +
                akCsSoftDrop6CaloSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop6CaloSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop6CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop6CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop6CaloJetBtagging = cms.Sequence(akCsSoftDrop6CaloJetBtaggingIP
            *akCsSoftDrop6CaloJetBtaggingSV
            *akCsSoftDrop6CaloJetBtaggingNegSV
#            *akCsSoftDrop6CaloJetBtaggingMu
            )

akCsSoftDrop6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop6CaloJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop6Calomatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop6Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop6CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop6CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop6CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop6CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop6CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop6CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop6CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop6CaloJetID"),
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

akCsSoftDrop6CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop6CaloJets"),
           	    R0  = cms.double( 0.6)
)
akCsSoftDrop6CalopatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop6CaloNjettiness:tau1','akCsSoftDrop6CaloNjettiness:tau2','akCsSoftDrop6CaloNjettiness:tau3']

akCsSoftDrop6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6GenJets',
                                                             rParam = 0.6,
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop6Calo"),
                                                             jetName = cms.untracked.string("akCsSoftDrop6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop6CaloJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop6Caloclean
                                                  #*
                                                  akCsSoftDrop6Calomatch
                                                  *
                                                  akCsSoftDrop6Caloparton
                                                  *
                                                  akCsSoftDrop6Calocorr
                                                  *
                                                  #akCsSoftDrop6CaloJetID
                                                  #*
                                                  akCsSoftDrop6CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop6CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop6CaloJetBtagging
                                                  *
                                                  akCsSoftDrop6CaloNjettiness
                                                  *
                                                  akCsSoftDrop6CalopatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop6CaloJetAnalyzer
                                                  )

akCsSoftDrop6CaloJetSequence_data = cms.Sequence(akCsSoftDrop6Calocorr
                                                    *
                                                    #akCsSoftDrop6CaloJetID
                                                    #*
                                                    akCsSoftDrop6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop6CaloJetBtagging
                                                    *
                                                    akCsSoftDrop6CaloNjettiness 
                                                    *
                                                    akCsSoftDrop6CalopatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop6CaloJetAnalyzer
                                                    )

akCsSoftDrop6CaloJetSequence_jec = cms.Sequence(akCsSoftDrop6CaloJetSequence_mc)
akCsSoftDrop6CaloJetSequence_mb = cms.Sequence(akCsSoftDrop6CaloJetSequence_mc)

akCsSoftDrop6CaloJetSequence = cms.Sequence(akCsSoftDrop6CaloJetSequence_data)
