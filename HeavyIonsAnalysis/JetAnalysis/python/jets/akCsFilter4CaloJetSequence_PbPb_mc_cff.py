

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter4CaloJets"),
    matched = cms.InputTag("ak4HiSignalGenJets"),
    maxDeltaR = 0.4
    )

akCsFilter4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter4CaloJets")
                                                        )

akCsFilter4Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter4CaloJets"),
    payload = "AK4Calo_offline"
    )

akCsFilter4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter4CaloJets'))

#akCsFilter4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiSignalGenJets'))

akCsFilter4CalobTagger = bTaggers("akCsFilter4Calo",0.4)

#create objects locally since they dont load properly otherwise
#akCsFilter4Calomatch = akCsFilter4CalobTagger.match
akCsFilter4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter4CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter4CaloPatJetFlavourAssociationLegacy = akCsFilter4CalobTagger.PatJetFlavourAssociationLegacy
akCsFilter4CaloPatJetPartons = akCsFilter4CalobTagger.PatJetPartons
akCsFilter4CaloJetTracksAssociatorAtVertex = akCsFilter4CalobTagger.JetTracksAssociatorAtVertex
akCsFilter4CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter4CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter4CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter4CaloCombinedSecondaryVertexBJetTags = akCsFilter4CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter4CaloCombinedSecondaryVertexV2BJetTags = akCsFilter4CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter4CaloJetBProbabilityBJetTags = akCsFilter4CalobTagger.JetBProbabilityBJetTags
akCsFilter4CaloSoftPFMuonByPtBJetTags = akCsFilter4CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter4CaloSoftPFMuonByIP3dBJetTags = akCsFilter4CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter4CaloTrackCountingHighEffBJetTags = akCsFilter4CalobTagger.TrackCountingHighEffBJetTags
akCsFilter4CaloTrackCountingHighPurBJetTags = akCsFilter4CalobTagger.TrackCountingHighPurBJetTags
akCsFilter4CaloPatJetPartonAssociationLegacy = akCsFilter4CalobTagger.PatJetPartonAssociationLegacy

akCsFilter4CaloImpactParameterTagInfos = akCsFilter4CalobTagger.ImpactParameterTagInfos
akCsFilter4CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter4CaloJetProbabilityBJetTags = akCsFilter4CalobTagger.JetProbabilityBJetTags

akCsFilter4CaloSecondaryVertexTagInfos = akCsFilter4CalobTagger.SecondaryVertexTagInfos
akCsFilter4CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter4CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter4CaloCombinedSecondaryVertexBJetTags = akCsFilter4CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter4CaloCombinedSecondaryVertexV2BJetTags = akCsFilter4CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter4CaloSecondaryVertexNegativeTagInfos = akCsFilter4CalobTagger.SecondaryVertexNegativeTagInfos
akCsFilter4CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter4CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter4CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter4CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter4CaloNegativeCombinedSecondaryVertexBJetTags = akCsFilter4CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter4CaloPositiveCombinedSecondaryVertexBJetTags = akCsFilter4CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter4CaloNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter4CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter4CaloPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter4CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter4CaloSoftPFMuonsTagInfos = akCsFilter4CalobTagger.SoftPFMuonsTagInfos
akCsFilter4CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter4CaloSoftPFMuonBJetTags = akCsFilter4CalobTagger.SoftPFMuonBJetTags
akCsFilter4CaloSoftPFMuonByIP3dBJetTags = akCsFilter4CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter4CaloSoftPFMuonByPtBJetTags = akCsFilter4CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter4CaloNegativeSoftPFMuonByPtBJetTags = akCsFilter4CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter4CaloPositiveSoftPFMuonByPtBJetTags = akCsFilter4CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter4CaloPatJetFlavourIdLegacy = cms.Sequence(akCsFilter4CaloPatJetPartonAssociationLegacy*akCsFilter4CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter4CaloPatJetFlavourAssociation = akCsFilter4CalobTagger.PatJetFlavourAssociation
#akCsFilter4CaloPatJetFlavourId = cms.Sequence(akCsFilter4CaloPatJetPartons*akCsFilter4CaloPatJetFlavourAssociation)

akCsFilter4CaloJetBtaggingIP       = cms.Sequence(akCsFilter4CaloImpactParameterTagInfos *
            (akCsFilter4CaloTrackCountingHighEffBJetTags +
             akCsFilter4CaloTrackCountingHighPurBJetTags +
             akCsFilter4CaloJetProbabilityBJetTags +
             akCsFilter4CaloJetBProbabilityBJetTags 
            )
            )

akCsFilter4CaloJetBtaggingSV = cms.Sequence(akCsFilter4CaloImpactParameterTagInfos
            *
            akCsFilter4CaloSecondaryVertexTagInfos
            * (akCsFilter4CaloSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter4CaloSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter4CaloCombinedSecondaryVertexBJetTags+
                akCsFilter4CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter4CaloJetBtaggingNegSV = cms.Sequence(akCsFilter4CaloImpactParameterTagInfos
            *
            akCsFilter4CaloSecondaryVertexNegativeTagInfos
            * (akCsFilter4CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter4CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter4CaloNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter4CaloPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter4CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter4CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter4CaloJetBtaggingMu = cms.Sequence(akCsFilter4CaloSoftPFMuonsTagInfos * (akCsFilter4CaloSoftPFMuonBJetTags
                +
                akCsFilter4CaloSoftPFMuonByIP3dBJetTags
                +
                akCsFilter4CaloSoftPFMuonByPtBJetTags
                +
                akCsFilter4CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter4CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter4CaloJetBtagging = cms.Sequence(akCsFilter4CaloJetBtaggingIP
            *akCsFilter4CaloJetBtaggingSV
            *akCsFilter4CaloJetBtaggingNegSV
#            *akCsFilter4CaloJetBtaggingMu
            )

akCsFilter4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter4CaloJets"),
        genJetMatch          = cms.InputTag("akCsFilter4Calomatch"),
        genPartonMatch       = cms.InputTag("akCsFilter4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter4Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter4CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter4CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter4CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter4CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter4CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter4CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter4CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter4CaloJetID"),
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

akCsFilter4CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter4CaloJets"),
           	    R0  = cms.double( 0.4)
)
akCsFilter4CalopatJetsWithBtagging.userData.userFloats.src += ['akCsFilter4CaloNjettiness:tau1','akCsFilter4CaloNjettiness:tau2','akCsFilter4CaloNjettiness:tau3']

akCsFilter4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCsFilter4Calo"),
                                                             jetName = cms.untracked.string("akCsFilter4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter4CaloJetSequence_mc = cms.Sequence(
                                                  #akCsFilter4Caloclean
                                                  #*
                                                  akCsFilter4Calomatch
                                                  *
                                                  akCsFilter4Caloparton
                                                  *
                                                  akCsFilter4Calocorr
                                                  *
                                                  #akCsFilter4CaloJetID
                                                  #*
                                                  akCsFilter4CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter4CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter4CaloJetBtagging
                                                  *
                                                  akCsFilter4CaloNjettiness
                                                  *
                                                  akCsFilter4CalopatJetsWithBtagging
                                                  *
                                                  akCsFilter4CaloJetAnalyzer
                                                  )

akCsFilter4CaloJetSequence_data = cms.Sequence(akCsFilter4Calocorr
                                                    *
                                                    #akCsFilter4CaloJetID
                                                    #*
                                                    akCsFilter4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter4CaloJetBtagging
                                                    *
                                                    akCsFilter4CaloNjettiness 
                                                    *
                                                    akCsFilter4CalopatJetsWithBtagging
                                                    *
                                                    akCsFilter4CaloJetAnalyzer
                                                    )

akCsFilter4CaloJetSequence_jec = cms.Sequence(akCsFilter4CaloJetSequence_mc)
akCsFilter4CaloJetSequence_mb = cms.Sequence(akCsFilter4CaloJetSequence_mc)

akCsFilter4CaloJetSequence = cms.Sequence(akCsFilter4CaloJetSequence_mc)
