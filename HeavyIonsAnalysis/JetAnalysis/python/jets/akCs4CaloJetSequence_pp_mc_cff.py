

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs4CaloJets"),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = 0.4
    )

akCs4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs4CaloJets")
                                                        )

akCs4Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs4CaloJets"),
    payload = "AK4Calo_offline"
    )

akCs4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs4CaloJets'))

#akCs4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4GenJets'))

akCs4CalobTagger = bTaggers("akCs4Calo",0.4)

#create objects locally since they dont load properly otherwise
#akCs4Calomatch = akCs4CalobTagger.match
akCs4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs4CaloJets"), matched = cms.InputTag("genParticles"))
akCs4CaloPatJetFlavourAssociationLegacy = akCs4CalobTagger.PatJetFlavourAssociationLegacy
akCs4CaloPatJetPartons = akCs4CalobTagger.PatJetPartons
akCs4CaloJetTracksAssociatorAtVertex = akCs4CalobTagger.JetTracksAssociatorAtVertex
akCs4CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs4CaloSimpleSecondaryVertexHighEffBJetTags = akCs4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs4CaloSimpleSecondaryVertexHighPurBJetTags = akCs4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs4CaloCombinedSecondaryVertexBJetTags = akCs4CalobTagger.CombinedSecondaryVertexBJetTags
akCs4CaloCombinedSecondaryVertexV2BJetTags = akCs4CalobTagger.CombinedSecondaryVertexV2BJetTags
akCs4CaloJetBProbabilityBJetTags = akCs4CalobTagger.JetBProbabilityBJetTags
akCs4CaloSoftPFMuonByPtBJetTags = akCs4CalobTagger.SoftPFMuonByPtBJetTags
akCs4CaloSoftPFMuonByIP3dBJetTags = akCs4CalobTagger.SoftPFMuonByIP3dBJetTags
akCs4CaloTrackCountingHighEffBJetTags = akCs4CalobTagger.TrackCountingHighEffBJetTags
akCs4CaloTrackCountingHighPurBJetTags = akCs4CalobTagger.TrackCountingHighPurBJetTags
akCs4CaloPatJetPartonAssociationLegacy = akCs4CalobTagger.PatJetPartonAssociationLegacy

akCs4CaloImpactParameterTagInfos = akCs4CalobTagger.ImpactParameterTagInfos
akCs4CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs4CaloJetProbabilityBJetTags = akCs4CalobTagger.JetProbabilityBJetTags

akCs4CaloSecondaryVertexTagInfos = akCs4CalobTagger.SecondaryVertexTagInfos
akCs4CaloSimpleSecondaryVertexHighEffBJetTags = akCs4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs4CaloSimpleSecondaryVertexHighPurBJetTags = akCs4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs4CaloCombinedSecondaryVertexBJetTags = akCs4CalobTagger.CombinedSecondaryVertexBJetTags
akCs4CaloCombinedSecondaryVertexV2BJetTags = akCs4CalobTagger.CombinedSecondaryVertexV2BJetTags

akCs4CaloSecondaryVertexNegativeTagInfos = akCs4CalobTagger.SecondaryVertexNegativeTagInfos
akCs4CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCs4CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs4CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCs4CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs4CaloNegativeCombinedSecondaryVertexBJetTags = akCs4CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCs4CaloPositiveCombinedSecondaryVertexBJetTags = akCs4CalobTagger.PositiveCombinedSecondaryVertexBJetTags
akCs4CaloNegativeCombinedSecondaryVertexV2BJetTags = akCs4CalobTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCs4CaloPositiveCombinedSecondaryVertexV2BJetTags = akCs4CalobTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCs4CaloSoftPFMuonsTagInfos = akCs4CalobTagger.SoftPFMuonsTagInfos
akCs4CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs4CaloSoftPFMuonBJetTags = akCs4CalobTagger.SoftPFMuonBJetTags
akCs4CaloSoftPFMuonByIP3dBJetTags = akCs4CalobTagger.SoftPFMuonByIP3dBJetTags
akCs4CaloSoftPFMuonByPtBJetTags = akCs4CalobTagger.SoftPFMuonByPtBJetTags
akCs4CaloNegativeSoftPFMuonByPtBJetTags = akCs4CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCs4CaloPositiveSoftPFMuonByPtBJetTags = akCs4CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCs4CaloPatJetFlavourIdLegacy = cms.Sequence(akCs4CaloPatJetPartonAssociationLegacy*akCs4CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs4CaloPatJetFlavourAssociation = akCs4CalobTagger.PatJetFlavourAssociation
#akCs4CaloPatJetFlavourId = cms.Sequence(akCs4CaloPatJetPartons*akCs4CaloPatJetFlavourAssociation)

akCs4CaloJetBtaggingIP       = cms.Sequence(akCs4CaloImpactParameterTagInfos *
            (akCs4CaloTrackCountingHighEffBJetTags +
             akCs4CaloTrackCountingHighPurBJetTags +
             akCs4CaloJetProbabilityBJetTags +
             akCs4CaloJetBProbabilityBJetTags 
            )
            )

akCs4CaloJetBtaggingSV = cms.Sequence(akCs4CaloImpactParameterTagInfos
            *
            akCs4CaloSecondaryVertexTagInfos
            * (akCs4CaloSimpleSecondaryVertexHighEffBJetTags+
                akCs4CaloSimpleSecondaryVertexHighPurBJetTags+
                akCs4CaloCombinedSecondaryVertexBJetTags+
                akCs4CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCs4CaloJetBtaggingNegSV = cms.Sequence(akCs4CaloImpactParameterTagInfos
            *
            akCs4CaloSecondaryVertexNegativeTagInfos
            * (akCs4CaloNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCs4CaloNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCs4CaloNegativeCombinedSecondaryVertexBJetTags+
                akCs4CaloPositiveCombinedSecondaryVertexBJetTags+
                akCs4CaloNegativeCombinedSecondaryVertexV2BJetTags+
                akCs4CaloPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCs4CaloJetBtaggingMu = cms.Sequence(akCs4CaloSoftPFMuonsTagInfos * (akCs4CaloSoftPFMuonBJetTags
                +
                akCs4CaloSoftPFMuonByIP3dBJetTags
                +
                akCs4CaloSoftPFMuonByPtBJetTags
                +
                akCs4CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCs4CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs4CaloJetBtagging = cms.Sequence(akCs4CaloJetBtaggingIP
            *akCs4CaloJetBtaggingSV
            *akCs4CaloJetBtaggingNegSV
#            *akCs4CaloJetBtaggingMu
            )

akCs4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs4CaloJets"),
        genJetMatch          = cms.InputTag("akCs4Calomatch"),
        genPartonMatch       = cms.InputTag("akCs4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs4Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCs4CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs4CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs4CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs4CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCs4CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCs4CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs4CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs4CaloJetID"),
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

akCs4CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs4CaloJets"),
           	    R0  = cms.double( 0.4)
)
akCs4CalopatJetsWithBtagging.userData.userFloats.src += ['akCs4CaloNjettiness:tau1','akCs4CaloNjettiness:tau2','akCs4CaloNjettiness:tau3']

akCs4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4GenJets',
                                                             rParam = 0.4,
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
                                                             bTagJetName = cms.untracked.string("akCs4Calo"),
                                                             jetName = cms.untracked.string("akCs4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs4CaloJetSequence_mc = cms.Sequence(
                                                  #akCs4Caloclean
                                                  #*
                                                  akCs4Calomatch
                                                  *
                                                  akCs4Caloparton
                                                  *
                                                  akCs4Calocorr
                                                  *
                                                  #akCs4CaloJetID
                                                  #*
                                                  akCs4CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCs4CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCs4CaloJetBtagging
                                                  *
                                                  akCs4CaloNjettiness
                                                  *
                                                  akCs4CalopatJetsWithBtagging
                                                  *
                                                  akCs4CaloJetAnalyzer
                                                  )

akCs4CaloJetSequence_data = cms.Sequence(akCs4Calocorr
                                                    *
                                                    #akCs4CaloJetID
                                                    #*
                                                    akCs4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCs4CaloJetBtagging
                                                    *
                                                    akCs4CaloNjettiness 
                                                    *
                                                    akCs4CalopatJetsWithBtagging
                                                    *
                                                    akCs4CaloJetAnalyzer
                                                    )

akCs4CaloJetSequence_jec = cms.Sequence(akCs4CaloJetSequence_mc)
akCs4CaloJetSequence_mb = cms.Sequence(akCs4CaloJetSequence_mc)

akCs4CaloJetSequence = cms.Sequence(akCs4CaloJetSequence_mc)
