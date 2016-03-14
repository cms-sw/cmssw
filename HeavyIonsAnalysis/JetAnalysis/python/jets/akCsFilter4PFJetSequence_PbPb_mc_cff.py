

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter4PFJets"),
    matched = cms.InputTag("ak4HiSignalGenJets"),
    maxDeltaR = 0.4
    )

akCsFilter4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter4PFJets")
                                                        )

akCsFilter4PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter4PFJets"),
    payload = "AK4PF_offline"
    )

akCsFilter4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter4CaloJets'))

#akCsFilter4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiSignalGenJets'))

akCsFilter4PFbTagger = bTaggers("akCsFilter4PF",0.4)

#create objects locally since they dont load properly otherwise
#akCsFilter4PFmatch = akCsFilter4PFbTagger.match
akCsFilter4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter4PFJets"), matched = cms.InputTag("hiSignalGenParticles"))
akCsFilter4PFPatJetFlavourAssociationLegacy = akCsFilter4PFbTagger.PatJetFlavourAssociationLegacy
akCsFilter4PFPatJetPartons = akCsFilter4PFbTagger.PatJetPartons
akCsFilter4PFJetTracksAssociatorAtVertex = akCsFilter4PFbTagger.JetTracksAssociatorAtVertex
akCsFilter4PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter4PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter4PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter4PFCombinedSecondaryVertexBJetTags = akCsFilter4PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter4PFCombinedSecondaryVertexV2BJetTags = akCsFilter4PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter4PFJetBProbabilityBJetTags = akCsFilter4PFbTagger.JetBProbabilityBJetTags
akCsFilter4PFSoftPFMuonByPtBJetTags = akCsFilter4PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter4PFSoftPFMuonByIP3dBJetTags = akCsFilter4PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter4PFTrackCountingHighEffBJetTags = akCsFilter4PFbTagger.TrackCountingHighEffBJetTags
akCsFilter4PFTrackCountingHighPurBJetTags = akCsFilter4PFbTagger.TrackCountingHighPurBJetTags
akCsFilter4PFPatJetPartonAssociationLegacy = akCsFilter4PFbTagger.PatJetPartonAssociationLegacy

akCsFilter4PFImpactParameterTagInfos = akCsFilter4PFbTagger.ImpactParameterTagInfos
akCsFilter4PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter4PFJetProbabilityBJetTags = akCsFilter4PFbTagger.JetProbabilityBJetTags

akCsFilter4PFSecondaryVertexTagInfos = akCsFilter4PFbTagger.SecondaryVertexTagInfos
akCsFilter4PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter4PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter4PFCombinedSecondaryVertexBJetTags = akCsFilter4PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter4PFCombinedSecondaryVertexV2BJetTags = akCsFilter4PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter4PFSecondaryVertexNegativeTagInfos = akCsFilter4PFbTagger.SecondaryVertexNegativeTagInfos
akCsFilter4PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter4PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter4PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter4PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter4PFNegativeCombinedSecondaryVertexBJetTags = akCsFilter4PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter4PFPositiveCombinedSecondaryVertexBJetTags = akCsFilter4PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter4PFNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter4PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter4PFPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter4PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter4PFSoftPFMuonsTagInfos = akCsFilter4PFbTagger.SoftPFMuonsTagInfos
akCsFilter4PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter4PFSoftPFMuonBJetTags = akCsFilter4PFbTagger.SoftPFMuonBJetTags
akCsFilter4PFSoftPFMuonByIP3dBJetTags = akCsFilter4PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter4PFSoftPFMuonByPtBJetTags = akCsFilter4PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter4PFNegativeSoftPFMuonByPtBJetTags = akCsFilter4PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter4PFPositiveSoftPFMuonByPtBJetTags = akCsFilter4PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter4PFPatJetFlavourIdLegacy = cms.Sequence(akCsFilter4PFPatJetPartonAssociationLegacy*akCsFilter4PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter4PFPatJetFlavourAssociation = akCsFilter4PFbTagger.PatJetFlavourAssociation
#akCsFilter4PFPatJetFlavourId = cms.Sequence(akCsFilter4PFPatJetPartons*akCsFilter4PFPatJetFlavourAssociation)

akCsFilter4PFJetBtaggingIP       = cms.Sequence(akCsFilter4PFImpactParameterTagInfos *
            (akCsFilter4PFTrackCountingHighEffBJetTags +
             akCsFilter4PFTrackCountingHighPurBJetTags +
             akCsFilter4PFJetProbabilityBJetTags +
             akCsFilter4PFJetBProbabilityBJetTags 
            )
            )

akCsFilter4PFJetBtaggingSV = cms.Sequence(akCsFilter4PFImpactParameterTagInfos
            *
            akCsFilter4PFSecondaryVertexTagInfos
            * (akCsFilter4PFSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter4PFSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter4PFCombinedSecondaryVertexBJetTags+
                akCsFilter4PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter4PFJetBtaggingNegSV = cms.Sequence(akCsFilter4PFImpactParameterTagInfos
            *
            akCsFilter4PFSecondaryVertexNegativeTagInfos
            * (akCsFilter4PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter4PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter4PFNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter4PFPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter4PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter4PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter4PFJetBtaggingMu = cms.Sequence(akCsFilter4PFSoftPFMuonsTagInfos * (akCsFilter4PFSoftPFMuonBJetTags
                +
                akCsFilter4PFSoftPFMuonByIP3dBJetTags
                +
                akCsFilter4PFSoftPFMuonByPtBJetTags
                +
                akCsFilter4PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter4PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter4PFJetBtagging = cms.Sequence(akCsFilter4PFJetBtaggingIP
            *akCsFilter4PFJetBtaggingSV
            *akCsFilter4PFJetBtaggingNegSV
#            *akCsFilter4PFJetBtaggingMu
            )

akCsFilter4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter4PFJets"),
        genJetMatch          = cms.InputTag("akCsFilter4PFmatch"),
        genPartonMatch       = cms.InputTag("akCsFilter4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter4PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter4PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter4PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter4PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter4PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter4PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter4PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter4PFJetID"),
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

akCsFilter4PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter4PFJets"),
           	    R0  = cms.double( 0.4)
)
akCsFilter4PFpatJetsWithBtagging.userData.userFloats.src += ['akCsFilter4PFNjettiness:tau1','akCsFilter4PFNjettiness:tau2','akCsFilter4PFNjettiness:tau3']

akCsFilter4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter4PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsFilter4PF"),
                                                             jetName = cms.untracked.string("akCsFilter4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter4PFJetSequence_mc = cms.Sequence(
                                                  #akCsFilter4PFclean
                                                  #*
                                                  akCsFilter4PFmatch
                                                  *
                                                  akCsFilter4PFparton
                                                  *
                                                  akCsFilter4PFcorr
                                                  *
                                                  #akCsFilter4PFJetID
                                                  #*
                                                  akCsFilter4PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter4PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter4PFJetBtagging
                                                  *
                                                  akCsFilter4PFNjettiness
                                                  *
                                                  akCsFilter4PFpatJetsWithBtagging
                                                  *
                                                  akCsFilter4PFJetAnalyzer
                                                  )

akCsFilter4PFJetSequence_data = cms.Sequence(akCsFilter4PFcorr
                                                    *
                                                    #akCsFilter4PFJetID
                                                    #*
                                                    akCsFilter4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter4PFJetBtagging
                                                    *
                                                    akCsFilter4PFNjettiness 
                                                    *
                                                    akCsFilter4PFpatJetsWithBtagging
                                                    *
                                                    akCsFilter4PFJetAnalyzer
                                                    )

akCsFilter4PFJetSequence_jec = cms.Sequence(akCsFilter4PFJetSequence_mc)
akCsFilter4PFJetSequence_mb = cms.Sequence(akCsFilter4PFJetSequence_mc)

akCsFilter4PFJetSequence = cms.Sequence(akCsFilter4PFJetSequence_mc)
