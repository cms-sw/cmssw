

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter2CaloJets"),
    matched = cms.InputTag("ak2GenJets"),
    maxDeltaR = 0.2
    )

akCsFilter2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter2CaloJets")
                                                        )

akCsFilter2Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter2CaloJets"),
    payload = "AK2Calo_offline"
    )

akCsFilter2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter2CaloJets'))

#akCsFilter2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2GenJets'))

akCsFilter2CalobTagger = bTaggers("akCsFilter2Calo",0.2)

#create objects locally since they dont load properly otherwise
#akCsFilter2Calomatch = akCsFilter2CalobTagger.match
akCsFilter2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter2CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter2CaloPatJetFlavourAssociationLegacy = akCsFilter2CalobTagger.PatJetFlavourAssociationLegacy
akCsFilter2CaloPatJetPartons = akCsFilter2CalobTagger.PatJetPartons
akCsFilter2CaloJetTracksAssociatorAtVertex = akCsFilter2CalobTagger.JetTracksAssociatorAtVertex
akCsFilter2CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter2CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter2CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter2CaloCombinedSecondaryVertexBJetTags = akCsFilter2CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter2CaloCombinedSecondaryVertexV2BJetTags = akCsFilter2CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter2CaloJetBProbabilityBJetTags = akCsFilter2CalobTagger.JetBProbabilityBJetTags
akCsFilter2CaloSoftPFMuonByPtBJetTags = akCsFilter2CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter2CaloSoftPFMuonByIP3dBJetTags = akCsFilter2CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter2CaloTrackCountingHighEffBJetTags = akCsFilter2CalobTagger.TrackCountingHighEffBJetTags
akCsFilter2CaloTrackCountingHighPurBJetTags = akCsFilter2CalobTagger.TrackCountingHighPurBJetTags
akCsFilter2CaloPatJetPartonAssociationLegacy = akCsFilter2CalobTagger.PatJetPartonAssociationLegacy

akCsFilter2CaloImpactParameterTagInfos = akCsFilter2CalobTagger.ImpactParameterTagInfos
akCsFilter2CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter2CaloJetProbabilityBJetTags = akCsFilter2CalobTagger.JetProbabilityBJetTags

akCsFilter2CaloSecondaryVertexTagInfos = akCsFilter2CalobTagger.SecondaryVertexTagInfos
akCsFilter2CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter2CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter2CaloCombinedSecondaryVertexBJetTags = akCsFilter2CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter2CaloCombinedSecondaryVertexV2BJetTags = akCsFilter2CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter2CaloSecondaryVertexNegativeTagInfos = akCsFilter2CalobTagger.SecondaryVertexNegativeTagInfos
akCsFilter2CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter2CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter2CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter2CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter2CaloNegativeCombinedSecondaryVertexBJetTags = akCsFilter2CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter2CaloPositiveCombinedSecondaryVertexBJetTags = akCsFilter2CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter2CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter2CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter2CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter2CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter2CaloSoftPFMuonsTagInfos = akCsFilter2CalobTagger.SoftPFMuonsTagInfos
akCsFilter2CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter2CaloSoftPFMuonBJetTags = akCsFilter2CalobTagger.SoftPFMuonBJetTags
akCsFilter2CaloSoftPFMuonByIP3dBJetTags = akCsFilter2CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter2CaloSoftPFMuonByPtBJetTags = akCsFilter2CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter2CaloNegativeSoftPFMuonByPtBJetTags = akCsFilter2CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter2CaloPositiveSoftPFMuonByPtBJetTags = akCsFilter2CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter2CaloPatJetFlavourIdLegacy = cms.Sequence(akCsFilter2CaloPatJetPartonAssociationLegacy*akCsFilter2CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter2CaloPatJetFlavourAssociation = akCsFilter2CalobTagger.PatJetFlavourAssociation
#akCsFilter2CaloPatJetFlavourId = cms.Sequence(akCsFilter2CaloPatJetPartons*akCsFilter2CaloPatJetFlavourAssociation)

akCsFilter2CaloJetBtaggingIP       = cms.Sequence(akCsFilter2CaloImpactParameterTagInfos *
            (akCsFilter2CaloTrackCountingHighEffBJetTags +
             akCsFilter2CaloTrackCountingHighPurBJetTags +
             akCsFilter2CaloJetProbabilityBJetTags +
             akCsFilter2CaloJetBProbabilityBJetTags 
            )
            )

akCsFilter2CaloJetBtaggingSV = cms.Sequence(akCsFilter2CaloImpactParameterTagInfos
            *
            akCsFilter2CaloSecondaryVertexTagInfos
            * (akCsFilter2CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter2CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter2CaloCombinedSecondaryVertexBJetTags+
                akCsFilter2CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter2CaloJetBtaggingNegSV = cms.Sequence(akCsFilter2CaloImpactParameterTagInfos
            *
            akCsFilter2CaloSecondaryVertexNegativeTagInfos
            * (akCsFilter2CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter2CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter2CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter2CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter2CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter2CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter2CaloJetBtaggingMu = cms.Sequence(akCsFilter2CaloSoftPFMuonsTagInfos * (akCsFilter2CaloSoftPFMuonBJetTags
                +
                akCsFilter2CaloSoftPFMuonByIP3dBJetTags
                +
                akCsFilter2CaloSoftPFMuonByPtBJetTags
                +
                akCsFilter2CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter2CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter2CaloJetBtagging = cms.Sequence(akCsFilter2CaloJetBtaggingIP
            *akCsFilter2CaloJetBtaggingSV
            *akCsFilter2CaloJetBtaggingNegSV
#            *akCsFilter2CaloJetBtaggingMu
            )

akCsFilter2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter2CaloJets"),
        genJetMatch          = cms.InputTag("akCsFilter2Calomatch"),
        genPartonMatch       = cms.InputTag("akCsFilter2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter2CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter2CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter2CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter2CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter2CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter2CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter2CaloJetID"),
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

akCsFilter2CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter2CaloJets"),
           	    R0  = cms.double( 0.2)
)
akCsFilter2CalopatJetsWithBtagging.userData.userFloats.src += ['akCsFilter2CaloNjettiness:tau1','akCsFilter2CaloNjettiness:tau2','akCsFilter2CaloNjettiness:tau3']

akCsFilter2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter2CalopatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsFilter2Calo"),
                                                             jetName = cms.untracked.string("akCsFilter2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter2CaloJetSequence_mc = cms.Sequence(
                                                  #akCsFilter2Caloclean
                                                  #*
                                                  akCsFilter2Calomatch
                                                  *
                                                  akCsFilter2Caloparton
                                                  *
                                                  akCsFilter2Calocorr
                                                  *
                                                  #akCsFilter2CaloJetID
                                                  #*
                                                  akCsFilter2CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter2CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter2CaloJetBtagging
                                                  *
                                                  akCsFilter2CaloNjettiness
                                                  *
                                                  akCsFilter2CalopatJetsWithBtagging
                                                  *
                                                  akCsFilter2CaloJetAnalyzer
                                                  )

akCsFilter2CaloJetSequence_data = cms.Sequence(akCsFilter2Calocorr
                                                    *
                                                    #akCsFilter2CaloJetID
                                                    #*
                                                    akCsFilter2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter2CaloJetBtagging
                                                    *
                                                    akCsFilter2CaloNjettiness 
                                                    *
                                                    akCsFilter2CalopatJetsWithBtagging
                                                    *
                                                    akCsFilter2CaloJetAnalyzer
                                                    )

akCsFilter2CaloJetSequence_jec = cms.Sequence(akCsFilter2CaloJetSequence_mc)
akCsFilter2CaloJetSequence_mb = cms.Sequence(akCsFilter2CaloJetSequence_mc)

akCsFilter2CaloJetSequence = cms.Sequence(akCsFilter2CaloJetSequence_mc)
