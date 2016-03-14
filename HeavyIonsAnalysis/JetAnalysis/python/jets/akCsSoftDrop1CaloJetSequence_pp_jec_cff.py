

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop1CaloJets"),
    matched = cms.InputTag("ak1GenJets"),
    maxDeltaR = 0.1
    )

akCsSoftDrop1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop1CaloJets")
                                                        )

akCsSoftDrop1Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop1CaloJets"),
    payload = "AK1Calo_offline"
    )

akCsSoftDrop1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop1CaloJets'))

#akCsSoftDrop1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1GenJets'))

akCsSoftDrop1CalobTagger = bTaggers("akCsSoftDrop1Calo",0.1)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop1Calomatch = akCsSoftDrop1CalobTagger.match
akCsSoftDrop1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop1CaloJets"), matched = cms.InputTag("genParticles"))
akCsSoftDrop1CaloPatJetFlavourAssociationLegacy = akCsSoftDrop1CalobTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop1CaloPatJetPartons = akCsSoftDrop1CalobTagger.PatJetPartons
akCsSoftDrop1CaloJetTracksAssociatorAtVertex = akCsSoftDrop1CalobTagger.JetTracksAssociatorAtVertex
akCsSoftDrop1CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop1CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop1CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop1CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop1CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop1CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop1CaloJetBProbabilityBJetTags = akCsSoftDrop1CalobTagger.JetBProbabilityBJetTags
akCsSoftDrop1CaloSoftPFMuonByPtBJetTags = akCsSoftDrop1CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop1CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop1CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop1CaloTrackCountingHighEffBJetTags = akCsSoftDrop1CalobTagger.TrackCountingHighEffBJetTags
akCsSoftDrop1CaloTrackCountingHighPurBJetTags = akCsSoftDrop1CalobTagger.TrackCountingHighPurBJetTags
akCsSoftDrop1CaloPatJetPartonAssociationLegacy = akCsSoftDrop1CalobTagger.PatJetPartonAssociationLegacy

akCsSoftDrop1CaloImpactParameterTagInfos = akCsSoftDrop1CalobTagger.ImpactParameterTagInfos
akCsSoftDrop1CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop1CaloJetProbabilityBJetTags = akCsSoftDrop1CalobTagger.JetProbabilityBJetTags

akCsSoftDrop1CaloSecondaryVertexTagInfos = akCsSoftDrop1CalobTagger.SecondaryVertexTagInfos
akCsSoftDrop1CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop1CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop1CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop1CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop1CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop1CaloSecondaryVertexNegativeTagInfos = akCsSoftDrop1CalobTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop1CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop1CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop1CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop1CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop1CaloNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop1CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop1CaloPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop1CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop1CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop1CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop1CaloSoftPFMuonsTagInfos = akCsSoftDrop1CalobTagger.SoftPFMuonsTagInfos
akCsSoftDrop1CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop1CaloSoftPFMuonBJetTags = akCsSoftDrop1CalobTagger.SoftPFMuonBJetTags
akCsSoftDrop1CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop1CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop1CaloSoftPFMuonByPtBJetTags = akCsSoftDrop1CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop1CaloNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop1CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop1CaloPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop1CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop1CaloPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop1CaloPatJetPartonAssociationLegacy*akCsSoftDrop1CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop1CaloPatJetFlavourAssociation = akCsSoftDrop1CalobTagger.PatJetFlavourAssociation
#akCsSoftDrop1CaloPatJetFlavourId = cms.Sequence(akCsSoftDrop1CaloPatJetPartons*akCsSoftDrop1CaloPatJetFlavourAssociation)

akCsSoftDrop1CaloJetBtaggingIP       = cms.Sequence(akCsSoftDrop1CaloImpactParameterTagInfos *
            (akCsSoftDrop1CaloTrackCountingHighEffBJetTags +
             akCsSoftDrop1CaloTrackCountingHighPurBJetTags +
             akCsSoftDrop1CaloJetProbabilityBJetTags +
             akCsSoftDrop1CaloJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop1CaloJetBtaggingSV = cms.Sequence(akCsSoftDrop1CaloImpactParameterTagInfos
            *
            akCsSoftDrop1CaloSecondaryVertexTagInfos
            * (akCsSoftDrop1CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop1CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop1CaloCombinedSecondaryVertexBJetTags+
                akCsSoftDrop1CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop1CaloJetBtaggingNegSV = cms.Sequence(akCsSoftDrop1CaloImpactParameterTagInfos
            *
            akCsSoftDrop1CaloSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop1CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop1CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop1CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop1CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop1CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop1CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop1CaloJetBtaggingMu = cms.Sequence(akCsSoftDrop1CaloSoftPFMuonsTagInfos * (akCsSoftDrop1CaloSoftPFMuonBJetTags
                +
                akCsSoftDrop1CaloSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop1CaloSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop1CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop1CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop1CaloJetBtagging = cms.Sequence(akCsSoftDrop1CaloJetBtaggingIP
            *akCsSoftDrop1CaloJetBtaggingSV
            *akCsSoftDrop1CaloJetBtaggingNegSV
#            *akCsSoftDrop1CaloJetBtaggingMu
            )

akCsSoftDrop1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop1CaloJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop1Calomatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop1CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop1CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop1CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop1CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop1CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop1CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop1CaloJetID"),
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

akCsSoftDrop1CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop1CaloJets"),
           	    R0  = cms.double( 0.1)
)
akCsSoftDrop1CalopatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop1CaloNjettiness:tau1','akCsSoftDrop1CaloNjettiness:tau2','akCsSoftDrop1CaloNjettiness:tau3']

akCsSoftDrop1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1GenJets',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop1Calo"),
                                                             jetName = cms.untracked.string("akCsSoftDrop1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop1CaloJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop1Caloclean
                                                  #*
                                                  akCsSoftDrop1Calomatch
                                                  *
                                                  akCsSoftDrop1Caloparton
                                                  *
                                                  akCsSoftDrop1Calocorr
                                                  *
                                                  #akCsSoftDrop1CaloJetID
                                                  #*
                                                  akCsSoftDrop1CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop1CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop1CaloJetBtagging
                                                  *
                                                  akCsSoftDrop1CaloNjettiness
                                                  *
                                                  akCsSoftDrop1CalopatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop1CaloJetAnalyzer
                                                  )

akCsSoftDrop1CaloJetSequence_data = cms.Sequence(akCsSoftDrop1Calocorr
                                                    *
                                                    #akCsSoftDrop1CaloJetID
                                                    #*
                                                    akCsSoftDrop1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop1CaloJetBtagging
                                                    *
                                                    akCsSoftDrop1CaloNjettiness 
                                                    *
                                                    akCsSoftDrop1CalopatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop1CaloJetAnalyzer
                                                    )

akCsSoftDrop1CaloJetSequence_jec = cms.Sequence(akCsSoftDrop1CaloJetSequence_mc)
akCsSoftDrop1CaloJetSequence_mb = cms.Sequence(akCsSoftDrop1CaloJetSequence_mc)

akCsSoftDrop1CaloJetSequence = cms.Sequence(akCsSoftDrop1CaloJetSequence_jec)
akCsSoftDrop1CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
akCsSoftDrop1CaloJetAnalyzer.jetPtMin = cms.double(1)
