

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter5CaloJets"),
    matched = cms.InputTag("ak5GenJets"),
    maxDeltaR = 0.5
    )

akCsFilter5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter5CaloJets")
                                                        )

akCsFilter5Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter5CaloJets"),
    payload = "AK5Calo_offline"
    )

akCsFilter5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter5CaloJets'))

#akCsFilter5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5GenJets'))

akCsFilter5CalobTagger = bTaggers("akCsFilter5Calo",0.5)

#create objects locally since they dont load properly otherwise
#akCsFilter5Calomatch = akCsFilter5CalobTagger.match
akCsFilter5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter5CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter5CaloPatJetFlavourAssociationLegacy = akCsFilter5CalobTagger.PatJetFlavourAssociationLegacy
akCsFilter5CaloPatJetPartons = akCsFilter5CalobTagger.PatJetPartons
akCsFilter5CaloJetTracksAssociatorAtVertex = akCsFilter5CalobTagger.JetTracksAssociatorAtVertex
akCsFilter5CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter5CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter5CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter5CaloCombinedSecondaryVertexBJetTags = akCsFilter5CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter5CaloCombinedSecondaryVertexV2BJetTags = akCsFilter5CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter5CaloJetBProbabilityBJetTags = akCsFilter5CalobTagger.JetBProbabilityBJetTags
akCsFilter5CaloSoftPFMuonByPtBJetTags = akCsFilter5CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter5CaloSoftPFMuonByIP3dBJetTags = akCsFilter5CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter5CaloTrackCountingHighEffBJetTags = akCsFilter5CalobTagger.TrackCountingHighEffBJetTags
akCsFilter5CaloTrackCountingHighPurBJetTags = akCsFilter5CalobTagger.TrackCountingHighPurBJetTags
akCsFilter5CaloPatJetPartonAssociationLegacy = akCsFilter5CalobTagger.PatJetPartonAssociationLegacy

akCsFilter5CaloImpactParameterTagInfos = akCsFilter5CalobTagger.ImpactParameterTagInfos
akCsFilter5CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter5CaloJetProbabilityBJetTags = akCsFilter5CalobTagger.JetProbabilityBJetTags

akCsFilter5CaloSecondaryVertexTagInfos = akCsFilter5CalobTagger.SecondaryVertexTagInfos
akCsFilter5CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter5CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter5CaloCombinedSecondaryVertexBJetTags = akCsFilter5CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter5CaloCombinedSecondaryVertexV2BJetTags = akCsFilter5CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter5CaloSecondaryVertexNegativeTagInfos = akCsFilter5CalobTagger.SecondaryVertexNegativeTagInfos
akCsFilter5CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter5CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter5CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter5CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter5CaloNegativeCombinedSecondaryVertexBJetTags = akCsFilter5CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter5CaloPositiveCombinedSecondaryVertexBJetTags = akCsFilter5CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter5CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter5CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter5CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter5CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter5CaloSoftPFMuonsTagInfos = akCsFilter5CalobTagger.SoftPFMuonsTagInfos
akCsFilter5CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter5CaloSoftPFMuonBJetTags = akCsFilter5CalobTagger.SoftPFMuonBJetTags
akCsFilter5CaloSoftPFMuonByIP3dBJetTags = akCsFilter5CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter5CaloSoftPFMuonByPtBJetTags = akCsFilter5CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter5CaloNegativeSoftPFMuonByPtBJetTags = akCsFilter5CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter5CaloPositiveSoftPFMuonByPtBJetTags = akCsFilter5CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter5CaloPatJetFlavourIdLegacy = cms.Sequence(akCsFilter5CaloPatJetPartonAssociationLegacy*akCsFilter5CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter5CaloPatJetFlavourAssociation = akCsFilter5CalobTagger.PatJetFlavourAssociation
#akCsFilter5CaloPatJetFlavourId = cms.Sequence(akCsFilter5CaloPatJetPartons*akCsFilter5CaloPatJetFlavourAssociation)

akCsFilter5CaloJetBtaggingIP       = cms.Sequence(akCsFilter5CaloImpactParameterTagInfos *
            (akCsFilter5CaloTrackCountingHighEffBJetTags +
             akCsFilter5CaloTrackCountingHighPurBJetTags +
             akCsFilter5CaloJetProbabilityBJetTags +
             akCsFilter5CaloJetBProbabilityBJetTags 
            )
            )

akCsFilter5CaloJetBtaggingSV = cms.Sequence(akCsFilter5CaloImpactParameterTagInfos
            *
            akCsFilter5CaloSecondaryVertexTagInfos
            * (akCsFilter5CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter5CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter5CaloCombinedSecondaryVertexBJetTags+
                akCsFilter5CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter5CaloJetBtaggingNegSV = cms.Sequence(akCsFilter5CaloImpactParameterTagInfos
            *
            akCsFilter5CaloSecondaryVertexNegativeTagInfos
            * (akCsFilter5CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter5CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter5CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter5CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter5CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter5CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter5CaloJetBtaggingMu = cms.Sequence(akCsFilter5CaloSoftPFMuonsTagInfos * (akCsFilter5CaloSoftPFMuonBJetTags
                +
                akCsFilter5CaloSoftPFMuonByIP3dBJetTags
                +
                akCsFilter5CaloSoftPFMuonByPtBJetTags
                +
                akCsFilter5CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter5CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter5CaloJetBtagging = cms.Sequence(akCsFilter5CaloJetBtaggingIP
            *akCsFilter5CaloJetBtaggingSV
            *akCsFilter5CaloJetBtaggingNegSV
#            *akCsFilter5CaloJetBtaggingMu
            )

akCsFilter5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter5CaloJets"),
        genJetMatch          = cms.InputTag("akCsFilter5Calomatch"),
        genPartonMatch       = cms.InputTag("akCsFilter5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter5Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter5CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter5CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter5CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter5CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter5CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter5CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter5CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter5CaloJetID"),
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

akCsFilter5CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter5CaloJets"),
           	    R0  = cms.double( 0.5)
)
akCsFilter5CalopatJetsWithBtagging.userData.userFloats.src += ['akCsFilter5CaloNjettiness:tau1','akCsFilter5CaloNjettiness:tau2','akCsFilter5CaloNjettiness:tau3']

akCsFilter5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter5CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak5GenJets',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akCsFilter5Calo"),
                                                             jetName = cms.untracked.string("akCsFilter5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter5CaloJetSequence_mc = cms.Sequence(
                                                  #akCsFilter5Caloclean
                                                  #*
                                                  akCsFilter5Calomatch
                                                  *
                                                  akCsFilter5Caloparton
                                                  *
                                                  akCsFilter5Calocorr
                                                  *
                                                  #akCsFilter5CaloJetID
                                                  #*
                                                  akCsFilter5CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter5CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter5CaloJetBtagging
                                                  *
                                                  akCsFilter5CaloNjettiness
                                                  *
                                                  akCsFilter5CalopatJetsWithBtagging
                                                  *
                                                  akCsFilter5CaloJetAnalyzer
                                                  )

akCsFilter5CaloJetSequence_data = cms.Sequence(akCsFilter5Calocorr
                                                    *
                                                    #akCsFilter5CaloJetID
                                                    #*
                                                    akCsFilter5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter5CaloJetBtagging
                                                    *
                                                    akCsFilter5CaloNjettiness 
                                                    *
                                                    akCsFilter5CalopatJetsWithBtagging
                                                    *
                                                    akCsFilter5CaloJetAnalyzer
                                                    )

akCsFilter5CaloJetSequence_jec = cms.Sequence(akCsFilter5CaloJetSequence_mc)
akCsFilter5CaloJetSequence_mb = cms.Sequence(akCsFilter5CaloJetSequence_mc)

akCsFilter5CaloJetSequence = cms.Sequence(akCsFilter5CaloJetSequence_mc)
