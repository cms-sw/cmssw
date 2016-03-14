

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter1CaloJets"),
    matched = cms.InputTag("ak1GenJets"),
    maxDeltaR = 0.1
    )

akCsFilter1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter1CaloJets")
                                                        )

akCsFilter1Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter1CaloJets"),
    payload = "AK1Calo_offline"
    )

akCsFilter1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter1CaloJets'))

#akCsFilter1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1GenJets'))

akCsFilter1CalobTagger = bTaggers("akCsFilter1Calo",0.1)

#create objects locally since they dont load properly otherwise
#akCsFilter1Calomatch = akCsFilter1CalobTagger.match
akCsFilter1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter1CaloJets"), matched = cms.InputTag("genParticles"))
akCsFilter1CaloPatJetFlavourAssociationLegacy = akCsFilter1CalobTagger.PatJetFlavourAssociationLegacy
akCsFilter1CaloPatJetPartons = akCsFilter1CalobTagger.PatJetPartons
akCsFilter1CaloJetTracksAssociatorAtVertex = akCsFilter1CalobTagger.JetTracksAssociatorAtVertex
akCsFilter1CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter1CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter1CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter1CaloCombinedSecondaryVertexBJetTags = akCsFilter1CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter1CaloCombinedSecondaryVertexV2BJetTags = akCsFilter1CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter1CaloJetBProbabilityBJetTags = akCsFilter1CalobTagger.JetBProbabilityBJetTags
akCsFilter1CaloSoftPFMuonByPtBJetTags = akCsFilter1CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter1CaloSoftPFMuonByIP3dBJetTags = akCsFilter1CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter1CaloTrackCountingHighEffBJetTags = akCsFilter1CalobTagger.TrackCountingHighEffBJetTags
akCsFilter1CaloTrackCountingHighPurBJetTags = akCsFilter1CalobTagger.TrackCountingHighPurBJetTags
akCsFilter1CaloPatJetPartonAssociationLegacy = akCsFilter1CalobTagger.PatJetPartonAssociationLegacy

akCsFilter1CaloImpactParameterTagInfos = akCsFilter1CalobTagger.ImpactParameterTagInfos
akCsFilter1CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter1CaloJetProbabilityBJetTags = akCsFilter1CalobTagger.JetProbabilityBJetTags

akCsFilter1CaloSecondaryVertexTagInfos = akCsFilter1CalobTagger.SecondaryVertexTagInfos
akCsFilter1CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter1CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter1CaloCombinedSecondaryVertexBJetTags = akCsFilter1CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter1CaloCombinedSecondaryVertexV2BJetTags = akCsFilter1CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter1CaloSecondaryVertexNegativeTagInfos = akCsFilter1CalobTagger.SecondaryVertexNegativeTagInfos
akCsFilter1CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter1CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter1CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter1CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter1CaloNegativeCombinedSecondaryVertexBJetTags = akCsFilter1CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter1CaloPositiveCombinedSecondaryVertexBJetTags = akCsFilter1CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter1CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter1CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter1CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter1CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter1CaloSoftPFMuonsTagInfos = akCsFilter1CalobTagger.SoftPFMuonsTagInfos
akCsFilter1CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter1CaloSoftPFMuonBJetTags = akCsFilter1CalobTagger.SoftPFMuonBJetTags
akCsFilter1CaloSoftPFMuonByIP3dBJetTags = akCsFilter1CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter1CaloSoftPFMuonByPtBJetTags = akCsFilter1CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter1CaloNegativeSoftPFMuonByPtBJetTags = akCsFilter1CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter1CaloPositiveSoftPFMuonByPtBJetTags = akCsFilter1CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter1CaloPatJetFlavourIdLegacy = cms.Sequence(akCsFilter1CaloPatJetPartonAssociationLegacy*akCsFilter1CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter1CaloPatJetFlavourAssociation = akCsFilter1CalobTagger.PatJetFlavourAssociation
#akCsFilter1CaloPatJetFlavourId = cms.Sequence(akCsFilter1CaloPatJetPartons*akCsFilter1CaloPatJetFlavourAssociation)

akCsFilter1CaloJetBtaggingIP       = cms.Sequence(akCsFilter1CaloImpactParameterTagInfos *
            (akCsFilter1CaloTrackCountingHighEffBJetTags +
             akCsFilter1CaloTrackCountingHighPurBJetTags +
             akCsFilter1CaloJetProbabilityBJetTags +
             akCsFilter1CaloJetBProbabilityBJetTags 
            )
            )

akCsFilter1CaloJetBtaggingSV = cms.Sequence(akCsFilter1CaloImpactParameterTagInfos
            *
            akCsFilter1CaloSecondaryVertexTagInfos
            * (akCsFilter1CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter1CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter1CaloCombinedSecondaryVertexBJetTags+
                akCsFilter1CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter1CaloJetBtaggingNegSV = cms.Sequence(akCsFilter1CaloImpactParameterTagInfos
            *
            akCsFilter1CaloSecondaryVertexNegativeTagInfos
            * (akCsFilter1CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter1CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter1CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter1CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter1CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter1CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter1CaloJetBtaggingMu = cms.Sequence(akCsFilter1CaloSoftPFMuonsTagInfos * (akCsFilter1CaloSoftPFMuonBJetTags
                +
                akCsFilter1CaloSoftPFMuonByIP3dBJetTags
                +
                akCsFilter1CaloSoftPFMuonByPtBJetTags
                +
                akCsFilter1CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter1CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter1CaloJetBtagging = cms.Sequence(akCsFilter1CaloJetBtaggingIP
            *akCsFilter1CaloJetBtaggingSV
            *akCsFilter1CaloJetBtaggingNegSV
#            *akCsFilter1CaloJetBtaggingMu
            )

akCsFilter1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter1CaloJets"),
        genJetMatch          = cms.InputTag("akCsFilter1Calomatch"),
        genPartonMatch       = cms.InputTag("akCsFilter1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter1CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter1CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter1CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter1CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter1CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter1CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter1CaloJetID"),
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

akCsFilter1CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter1CaloJets"),
           	    R0  = cms.double( 0.1)
)
akCsFilter1CalopatJetsWithBtagging.userData.userFloats.src += ['akCsFilter1CaloNjettiness:tau1','akCsFilter1CaloNjettiness:tau2','akCsFilter1CaloNjettiness:tau3']

akCsFilter1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter1CalopatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsFilter1Calo"),
                                                             jetName = cms.untracked.string("akCsFilter1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter1CaloJetSequence_mc = cms.Sequence(
                                                  #akCsFilter1Caloclean
                                                  #*
                                                  akCsFilter1Calomatch
                                                  *
                                                  akCsFilter1Caloparton
                                                  *
                                                  akCsFilter1Calocorr
                                                  *
                                                  #akCsFilter1CaloJetID
                                                  #*
                                                  akCsFilter1CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter1CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter1CaloJetBtagging
                                                  *
                                                  akCsFilter1CaloNjettiness
                                                  *
                                                  akCsFilter1CalopatJetsWithBtagging
                                                  *
                                                  akCsFilter1CaloJetAnalyzer
                                                  )

akCsFilter1CaloJetSequence_data = cms.Sequence(akCsFilter1Calocorr
                                                    *
                                                    #akCsFilter1CaloJetID
                                                    #*
                                                    akCsFilter1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter1CaloJetBtagging
                                                    *
                                                    akCsFilter1CaloNjettiness 
                                                    *
                                                    akCsFilter1CalopatJetsWithBtagging
                                                    *
                                                    akCsFilter1CaloJetAnalyzer
                                                    )

akCsFilter1CaloJetSequence_jec = cms.Sequence(akCsFilter1CaloJetSequence_mc)
akCsFilter1CaloJetSequence_mb = cms.Sequence(akCsFilter1CaloJetSequence_mc)

akCsFilter1CaloJetSequence = cms.Sequence(akCsFilter1CaloJetSequence_jec)
akCsFilter1CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
akCsFilter1CaloJetAnalyzer.jetPtMin = cms.double(1)
