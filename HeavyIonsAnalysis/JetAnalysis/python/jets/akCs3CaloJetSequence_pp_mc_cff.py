

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs3CaloJets"),
    matched = cms.InputTag("ak3GenJets"),
    maxDeltaR = 0.3
    )

akCs3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs3CaloJets")
                                                        )

akCs3Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs3CaloJets"),
    payload = "AK3Calo_offline"
    )

akCs3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs3CaloJets'))

#akCs3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3GenJets'))

akCs3CalobTagger = bTaggers("akCs3Calo",0.3)

#create objects locally since they dont load properly otherwise
#akCs3Calomatch = akCs3CalobTagger.match
akCs3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs3CaloJets"), matched = cms.InputTag("genParticles"))
akCs3CaloPatJetFlavourAssociationLegacy = akCs3CalobTagger.PatJetFlavourAssociationLegacy
akCs3CaloPatJetPartons = akCs3CalobTagger.PatJetPartons
akCs3CaloJetTracksAssociatorAtVertex = akCs3CalobTagger.JetTracksAssociatorAtVertex
akCs3CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs3CaloSimpleSecondaryVertexHighEffBJetTags = akCs3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs3CaloSimpleSecondaryVertexHighPurBJetTags = akCs3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs3CaloCombinedSecondaryVertexBJetTags = akCs3CalobTagger.CombinedSecondaryVertexBJetTags
akCs3CaloCombinedSecondaryVertexV2BJetTags = akCs3CalobTagger.CombinedSecondaryVertexV2BJetTags
akCs3CaloJetBProbabilityBJetTags = akCs3CalobTagger.JetBProbabilityBJetTags
akCs3CaloSoftPFMuonByPtBJetTags = akCs3CalobTagger.SoftPFMuonByPtBJetTags
akCs3CaloSoftPFMuonByIP3dBJetTags = akCs3CalobTagger.SoftPFMuonByIP3dBJetTags
akCs3CaloTrackCountingHighEffBJetTags = akCs3CalobTagger.TrackCountingHighEffBJetTags
akCs3CaloTrackCountingHighPurBJetTags = akCs3CalobTagger.TrackCountingHighPurBJetTags
akCs3CaloPatJetPartonAssociationLegacy = akCs3CalobTagger.PatJetPartonAssociationLegacy

akCs3CaloImpactParameterTagInfos = akCs3CalobTagger.ImpactParameterTagInfos
akCs3CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs3CaloJetProbabilityBJetTags = akCs3CalobTagger.JetProbabilityBJetTags

akCs3CaloSecondaryVertexTagInfos = akCs3CalobTagger.SecondaryVertexTagInfos
akCs3CaloSimpleSecondaryVertexHighEffBJetTags = akCs3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs3CaloSimpleSecondaryVertexHighPurBJetTags = akCs3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs3CaloCombinedSecondaryVertexBJetTags = akCs3CalobTagger.CombinedSecondaryVertexBJetTags
akCs3CaloCombinedSecondaryVertexV2BJetTags = akCs3CalobTagger.CombinedSecondaryVertexV2BJetTags

akCs3CaloSecondaryVertexNegativeTagInfos = akCs3CalobTagger.SecondaryVertexNegativeTagInfos
akCs3CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCs3CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs3CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCs3CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs3CaloNegativeCombinedSecondaryVertexBJetTags = akCs3CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCs3CaloPositiveCombinedSecondaryVertexBJetTags = akCs3CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCs3CaloNegativeCombinedSecondaryVertexV2BJetTags = akCs3CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCs3CaloPositiveCombinedSecondaryVertexV2BJetTags = akCs3CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCs3CaloSoftPFMuonsTagInfos = akCs3CalobTagger.SoftPFMuonsTagInfos
akCs3CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs3CaloSoftPFMuonBJetTags = akCs3CalobTagger.SoftPFMuonBJetTags
akCs3CaloSoftPFMuonByIP3dBJetTags = akCs3CalobTagger.SoftPFMuonByIP3dBJetTags
akCs3CaloSoftPFMuonByPtBJetTags = akCs3CalobTagger.SoftPFMuonByPtBJetTags
akCs3CaloNegativeSoftPFMuonByPtBJetTags = akCs3CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCs3CaloPositiveSoftPFMuonByPtBJetTags = akCs3CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCs3CaloPatJetFlavourIdLegacy = cms.Sequence(akCs3CaloPatJetPartonAssociationLegacy*akCs3CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs3CaloPatJetFlavourAssociation = akCs3CalobTagger.PatJetFlavourAssociation
#akCs3CaloPatJetFlavourId = cms.Sequence(akCs3CaloPatJetPartons*akCs3CaloPatJetFlavourAssociation)

akCs3CaloJetBtaggingIP       = cms.Sequence(akCs3CaloImpactParameterTagInfos *
            (akCs3CaloTrackCountingHighEffBJetTags +
             akCs3CaloTrackCountingHighPurBJetTags +
             akCs3CaloJetProbabilityBJetTags +
             akCs3CaloJetBProbabilityBJetTags 
            )
            )

akCs3CaloJetBtaggingSV = cms.Sequence(akCs3CaloImpactParameterTagInfos
            *
            akCs3CaloSecondaryVertexTagInfos
            * (akCs3CaloSimpleSecondaryVertexHighEffBJetTags+
                akCs3CaloSimpleSecondaryVertexHighPurBJetTags+
                akCs3CaloCombinedSecondaryVertexBJetTags+
                akCs3CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCs3CaloJetBtaggingNegSV = cms.Sequence(akCs3CaloImpactParameterTagInfos
            *
            akCs3CaloSecondaryVertexNegativeTagInfos
            * (akCs3CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCs3CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCs3CaloNegativeCombinedSecondaryVertexBJetTags+
                akCs3CaloPositiveCombinedSecondaryVertexBJetTags+
                akCs3CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCs3CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCs3CaloJetBtaggingMu = cms.Sequence(akCs3CaloSoftPFMuonsTagInfos * (akCs3CaloSoftPFMuonBJetTags
                +
                akCs3CaloSoftPFMuonByIP3dBJetTags
                +
                akCs3CaloSoftPFMuonByPtBJetTags
                +
                akCs3CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCs3CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs3CaloJetBtagging = cms.Sequence(akCs3CaloJetBtaggingIP
            *akCs3CaloJetBtaggingSV
            *akCs3CaloJetBtaggingNegSV
#            *akCs3CaloJetBtaggingMu
            )

akCs3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs3CaloJets"),
        genJetMatch          = cms.InputTag("akCs3Calomatch"),
        genPartonMatch       = cms.InputTag("akCs3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs3Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCs3CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs3CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs3CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs3CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCs3CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCs3CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs3CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs3CaloJetID"),
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

akCs3CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs3CaloJets"),
           	    R0  = cms.double( 0.3)
)
akCs3CalopatJetsWithBtagging.userData.userFloats.src += ['akCs3CaloNjettiness:tau1','akCs3CaloNjettiness:tau2','akCs3CaloNjettiness:tau3']

akCs3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs3CalopatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCs3Calo"),
                                                             jetName = cms.untracked.string("akCs3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs3CaloJetSequence_mc = cms.Sequence(
                                                  #akCs3Caloclean
                                                  #*
                                                  akCs3Calomatch
                                                  *
                                                  akCs3Caloparton
                                                  *
                                                  akCs3Calocorr
                                                  *
                                                  #akCs3CaloJetID
                                                  #*
                                                  akCs3CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCs3CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCs3CaloJetBtagging
                                                  *
                                                  akCs3CaloNjettiness
                                                  *
                                                  akCs3CalopatJetsWithBtagging
                                                  *
                                                  akCs3CaloJetAnalyzer
                                                  )

akCs3CaloJetSequence_data = cms.Sequence(akCs3Calocorr
                                                    *
                                                    #akCs3CaloJetID
                                                    #*
                                                    akCs3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCs3CaloJetBtagging
                                                    *
                                                    akCs3CaloNjettiness 
                                                    *
                                                    akCs3CalopatJetsWithBtagging
                                                    *
                                                    akCs3CaloJetAnalyzer
                                                    )

akCs3CaloJetSequence_jec = cms.Sequence(akCs3CaloJetSequence_mc)
akCs3CaloJetSequence_mb = cms.Sequence(akCs3CaloJetSequence_mc)

akCs3CaloJetSequence = cms.Sequence(akCs3CaloJetSequence_mc)
