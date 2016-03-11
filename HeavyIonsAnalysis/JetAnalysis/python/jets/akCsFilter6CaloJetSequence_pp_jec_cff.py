

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter6CaloJets"),
    matched = cms.InputTag("ak6GenJets"),
    maxDeltaR = 0.6
    )

akCsFilter6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter6CaloJets")
                                                        )

akCsFilter6Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter6CaloJets"),
    payload = "AK6Calo_offline"
    )

akCsFilter6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter6CaloJets'))

#akCsFilter6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6GenJets'))

akCsFilter6CalobTagger = bTaggers("akCsFilter6Calo",0.6)

#create objects locally since they dont load properly otherwise
#akCsFilter6Calomatch = akCsFilter6CalobTagger.match
akCsFilter6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter6CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter6CaloPatJetFlavourAssociationLegacy = akCsFilter6CalobTagger.PatJetFlavourAssociationLegacy
akCsFilter6CaloPatJetPartons = akCsFilter6CalobTagger.PatJetPartons
akCsFilter6CaloJetTracksAssociatorAtVertex = akCsFilter6CalobTagger.JetTracksAssociatorAtVertex
akCsFilter6CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter6CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter6CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter6CaloCombinedSecondaryVertexBJetTags = akCsFilter6CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter6CaloCombinedSecondaryVertexV2BJetTags = akCsFilter6CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter6CaloJetBProbabilityBJetTags = akCsFilter6CalobTagger.JetBProbabilityBJetTags
akCsFilter6CaloSoftPFMuonByPtBJetTags = akCsFilter6CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter6CaloSoftPFMuonByIP3dBJetTags = akCsFilter6CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter6CaloTrackCountingHighEffBJetTags = akCsFilter6CalobTagger.TrackCountingHighEffBJetTags
akCsFilter6CaloTrackCountingHighPurBJetTags = akCsFilter6CalobTagger.TrackCountingHighPurBJetTags
akCsFilter6CaloPatJetPartonAssociationLegacy = akCsFilter6CalobTagger.PatJetPartonAssociationLegacy

akCsFilter6CaloImpactParameterTagInfos = akCsFilter6CalobTagger.ImpactParameterTagInfos
akCsFilter6CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter6CaloJetProbabilityBJetTags = akCsFilter6CalobTagger.JetProbabilityBJetTags

akCsFilter6CaloSecondaryVertexTagInfos = akCsFilter6CalobTagger.SecondaryVertexTagInfos
akCsFilter6CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter6CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter6CaloCombinedSecondaryVertexBJetTags = akCsFilter6CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter6CaloCombinedSecondaryVertexV2BJetTags = akCsFilter6CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter6CaloSecondaryVertexNegativeTagInfos = akCsFilter6CalobTagger.SecondaryVertexNegativeTagInfos
akCsFilter6CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter6CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter6CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter6CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter6CaloNegativeCombinedSecondaryVertexBJetTags = akCsFilter6CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter6CaloPositiveCombinedSecondaryVertexBJetTags = akCsFilter6CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter6CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter6CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter6CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter6CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter6CaloSoftPFMuonsTagInfos = akCsFilter6CalobTagger.SoftPFMuonsTagInfos
akCsFilter6CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter6CaloSoftPFMuonBJetTags = akCsFilter6CalobTagger.SoftPFMuonBJetTags
akCsFilter6CaloSoftPFMuonByIP3dBJetTags = akCsFilter6CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter6CaloSoftPFMuonByPtBJetTags = akCsFilter6CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter6CaloNegativeSoftPFMuonByPtBJetTags = akCsFilter6CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter6CaloPositiveSoftPFMuonByPtBJetTags = akCsFilter6CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter6CaloPatJetFlavourIdLegacy = cms.Sequence(akCsFilter6CaloPatJetPartonAssociationLegacy*akCsFilter6CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter6CaloPatJetFlavourAssociation = akCsFilter6CalobTagger.PatJetFlavourAssociation
#akCsFilter6CaloPatJetFlavourId = cms.Sequence(akCsFilter6CaloPatJetPartons*akCsFilter6CaloPatJetFlavourAssociation)

akCsFilter6CaloJetBtaggingIP       = cms.Sequence(akCsFilter6CaloImpactParameterTagInfos *
            (akCsFilter6CaloTrackCountingHighEffBJetTags +
             akCsFilter6CaloTrackCountingHighPurBJetTags +
             akCsFilter6CaloJetProbabilityBJetTags +
             akCsFilter6CaloJetBProbabilityBJetTags 
            )
            )

akCsFilter6CaloJetBtaggingSV = cms.Sequence(akCsFilter6CaloImpactParameterTagInfos
            *
            akCsFilter6CaloSecondaryVertexTagInfos
            * (akCsFilter6CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter6CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter6CaloCombinedSecondaryVertexBJetTags+
                akCsFilter6CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter6CaloJetBtaggingNegSV = cms.Sequence(akCsFilter6CaloImpactParameterTagInfos
            *
            akCsFilter6CaloSecondaryVertexNegativeTagInfos
            * (akCsFilter6CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter6CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter6CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter6CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter6CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter6CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter6CaloJetBtaggingMu = cms.Sequence(akCsFilter6CaloSoftPFMuonsTagInfos * (akCsFilter6CaloSoftPFMuonBJetTags
                +
                akCsFilter6CaloSoftPFMuonByIP3dBJetTags
                +
                akCsFilter6CaloSoftPFMuonByPtBJetTags
                +
                akCsFilter6CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter6CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter6CaloJetBtagging = cms.Sequence(akCsFilter6CaloJetBtaggingIP
            *akCsFilter6CaloJetBtaggingSV
            *akCsFilter6CaloJetBtaggingNegSV
#            *akCsFilter6CaloJetBtaggingMu
            )

akCsFilter6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter6CaloJets"),
        genJetMatch          = cms.InputTag("akCsFilter6Calomatch"),
        genPartonMatch       = cms.InputTag("akCsFilter6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter6Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter6CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter6CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter6CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter6CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter6CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter6CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter6CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter6CaloJetID"),
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

akCsFilter6CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter6CaloJets"),
           	    R0  = cms.double( 0.6)
)
akCsFilter6CalopatJetsWithBtagging.userData.userFloats.src += ['akCsFilter6CaloNjettiness:tau1','akCsFilter6CaloNjettiness:tau2','akCsFilter6CaloNjettiness:tau3']

akCsFilter6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6GenJets',
                                                             rParam = 0.6,
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
                                                             bTagJetName = cms.untracked.string("akCsFilter6Calo"),
                                                             jetName = cms.untracked.string("akCsFilter6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter6CaloJetSequence_mc = cms.Sequence(
                                                  #akCsFilter6Caloclean
                                                  #*
                                                  akCsFilter6Calomatch
                                                  *
                                                  akCsFilter6Caloparton
                                                  *
                                                  akCsFilter6Calocorr
                                                  *
                                                  #akCsFilter6CaloJetID
                                                  #*
                                                  akCsFilter6CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter6CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter6CaloJetBtagging
                                                  *
                                                  akCsFilter6CaloNjettiness
                                                  *
                                                  akCsFilter6CalopatJetsWithBtagging
                                                  *
                                                  akCsFilter6CaloJetAnalyzer
                                                  )

akCsFilter6CaloJetSequence_data = cms.Sequence(akCsFilter6Calocorr
                                                    *
                                                    #akCsFilter6CaloJetID
                                                    #*
                                                    akCsFilter6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter6CaloJetBtagging
                                                    *
                                                    akCsFilter6CaloNjettiness 
                                                    *
                                                    akCsFilter6CalopatJetsWithBtagging
                                                    *
                                                    akCsFilter6CaloJetAnalyzer
                                                    )

akCsFilter6CaloJetSequence_jec = cms.Sequence(akCsFilter6CaloJetSequence_mc)
akCsFilter6CaloJetSequence_mb = cms.Sequence(akCsFilter6CaloJetSequence_mc)

akCsFilter6CaloJetSequence = cms.Sequence(akCsFilter6CaloJetSequence_jec)
akCsFilter6CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
akCsFilter6CaloJetAnalyzer.jetPtMin = cms.double(1)
