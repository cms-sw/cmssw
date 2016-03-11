

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop3CaloJets"),
    matched = cms.InputTag("ak3GenJets"),
    maxDeltaR = 0.3
    )

akCsSoftDrop3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop3CaloJets")
                                                        )

akCsSoftDrop3Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop3CaloJets"),
    payload = "AK3Calo_offline"
    )

akCsSoftDrop3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop3CaloJets'))

#akCsSoftDrop3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3GenJets'))

akCsSoftDrop3CalobTagger = bTaggers("akCsSoftDrop3Calo",0.3)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop3Calomatch = akCsSoftDrop3CalobTagger.match
akCsSoftDrop3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop3CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop3CaloPatJetFlavourAssociationLegacy = akCsSoftDrop3CalobTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop3CaloPatJetPartons = akCsSoftDrop3CalobTagger.PatJetPartons
akCsSoftDrop3CaloJetTracksAssociatorAtVertex = akCsSoftDrop3CalobTagger.JetTracksAssociatorAtVertex
akCsSoftDrop3CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop3CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop3CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop3CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop3CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop3CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop3CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop3CaloJetBProbabilityBJetTags = akCsSoftDrop3CalobTagger.JetBProbabilityBJetTags
akCsSoftDrop3CaloSoftPFMuonByPtBJetTags = akCsSoftDrop3CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop3CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop3CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop3CaloTrackCountingHighEffBJetTags = akCsSoftDrop3CalobTagger.TrackCountingHighEffBJetTags
akCsSoftDrop3CaloTrackCountingHighPurBJetTags = akCsSoftDrop3CalobTagger.TrackCountingHighPurBJetTags
akCsSoftDrop3CaloPatJetPartonAssociationLegacy = akCsSoftDrop3CalobTagger.PatJetPartonAssociationLegacy

akCsSoftDrop3CaloImpactParameterTagInfos = akCsSoftDrop3CalobTagger.ImpactParameterTagInfos
akCsSoftDrop3CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop3CaloJetProbabilityBJetTags = akCsSoftDrop3CalobTagger.JetProbabilityBJetTags

akCsSoftDrop3CaloSecondaryVertexTagInfos = akCsSoftDrop3CalobTagger.SecondaryVertexTagInfos
akCsSoftDrop3CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop3CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop3CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop3CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop3CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop3CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop3CaloSecondaryVertexNegativeTagInfos = akCsSoftDrop3CalobTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop3CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop3CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop3CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop3CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop3CaloNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop3CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop3CaloPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop3CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop3CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop3CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop3CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop3CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop3CaloSoftPFMuonsTagInfos = akCsSoftDrop3CalobTagger.SoftPFMuonsTagInfos
akCsSoftDrop3CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop3CaloSoftPFMuonBJetTags = akCsSoftDrop3CalobTagger.SoftPFMuonBJetTags
akCsSoftDrop3CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop3CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop3CaloSoftPFMuonByPtBJetTags = akCsSoftDrop3CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop3CaloNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop3CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop3CaloPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop3CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop3CaloPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop3CaloPatJetPartonAssociationLegacy*akCsSoftDrop3CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop3CaloPatJetFlavourAssociation = akCsSoftDrop3CalobTagger.PatJetFlavourAssociation
#akCsSoftDrop3CaloPatJetFlavourId = cms.Sequence(akCsSoftDrop3CaloPatJetPartons*akCsSoftDrop3CaloPatJetFlavourAssociation)

akCsSoftDrop3CaloJetBtaggingIP       = cms.Sequence(akCsSoftDrop3CaloImpactParameterTagInfos *
            (akCsSoftDrop3CaloTrackCountingHighEffBJetTags +
             akCsSoftDrop3CaloTrackCountingHighPurBJetTags +
             akCsSoftDrop3CaloJetProbabilityBJetTags +
             akCsSoftDrop3CaloJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop3CaloJetBtaggingSV = cms.Sequence(akCsSoftDrop3CaloImpactParameterTagInfos
            *
            akCsSoftDrop3CaloSecondaryVertexTagInfos
            * (akCsSoftDrop3CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop3CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop3CaloCombinedSecondaryVertexBJetTags+
                akCsSoftDrop3CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop3CaloJetBtaggingNegSV = cms.Sequence(akCsSoftDrop3CaloImpactParameterTagInfos
            *
            akCsSoftDrop3CaloSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop3CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop3CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop3CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop3CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop3CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop3CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop3CaloJetBtaggingMu = cms.Sequence(akCsSoftDrop3CaloSoftPFMuonsTagInfos * (akCsSoftDrop3CaloSoftPFMuonBJetTags
                +
                akCsSoftDrop3CaloSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop3CaloSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop3CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop3CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop3CaloJetBtagging = cms.Sequence(akCsSoftDrop3CaloJetBtaggingIP
            *akCsSoftDrop3CaloJetBtaggingSV
            *akCsSoftDrop3CaloJetBtaggingNegSV
#            *akCsSoftDrop3CaloJetBtaggingMu
            )

akCsSoftDrop3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop3CaloJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop3Calomatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop3Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop3CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop3CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop3CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop3CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop3CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop3CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop3CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop3CaloJetID"),
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

akCsSoftDrop3CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop3CaloJets"),
           	    R0  = cms.double( 0.3)
)
akCsSoftDrop3CalopatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop3CaloNjettiness:tau1','akCsSoftDrop3CaloNjettiness:tau2','akCsSoftDrop3CaloNjettiness:tau3']

akCsSoftDrop3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3GenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop3Calo"),
                                                             jetName = cms.untracked.string("akCsSoftDrop3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop3CaloJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop3Caloclean
                                                  #*
                                                  akCsSoftDrop3Calomatch
                                                  *
                                                  akCsSoftDrop3Caloparton
                                                  *
                                                  akCsSoftDrop3Calocorr
                                                  *
                                                  #akCsSoftDrop3CaloJetID
                                                  #*
                                                  akCsSoftDrop3CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop3CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop3CaloJetBtagging
                                                  *
                                                  akCsSoftDrop3CaloNjettiness
                                                  *
                                                  akCsSoftDrop3CalopatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop3CaloJetAnalyzer
                                                  )

akCsSoftDrop3CaloJetSequence_data = cms.Sequence(akCsSoftDrop3Calocorr
                                                    *
                                                    #akCsSoftDrop3CaloJetID
                                                    #*
                                                    akCsSoftDrop3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop3CaloJetBtagging
                                                    *
                                                    akCsSoftDrop3CaloNjettiness 
                                                    *
                                                    akCsSoftDrop3CalopatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop3CaloJetAnalyzer
                                                    )

akCsSoftDrop3CaloJetSequence_jec = cms.Sequence(akCsSoftDrop3CaloJetSequence_mc)
akCsSoftDrop3CaloJetSequence_mb = cms.Sequence(akCsSoftDrop3CaloJetSequence_mc)

akCsSoftDrop3CaloJetSequence = cms.Sequence(akCsSoftDrop3CaloJetSequence_jec)
akCsSoftDrop3CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
akCsSoftDrop3CaloJetAnalyzer.jetPtMin = cms.double(1)
