

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs6PFJets"),
    matched = cms.InputTag("ak6HiSignalGenJets"),
    maxDeltaR = 0.6
    )

akCs6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs6PFJets")
                                                        )

akCs6PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs6PFJets"),
    payload = "AK6PF_offline"
    )

akCs6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs6CaloJets'))

#akCs6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiSignalGenJets'))

akCs6PFbTagger = bTaggers("akCs6PF",0.6)

#create objects locally since they dont load properly otherwise
#akCs6PFmatch = akCs6PFbTagger.match
akCs6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs6PFJets"), matched = cms.InputTag("selectedPartons"))
akCs6PFPatJetFlavourAssociationLegacy = akCs6PFbTagger.PatJetFlavourAssociationLegacy
akCs6PFPatJetPartons = akCs6PFbTagger.PatJetPartons
akCs6PFJetTracksAssociatorAtVertex = akCs6PFbTagger.JetTracksAssociatorAtVertex
akCs6PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs6PFSimpleSecondaryVertexHighEffBJetTags = akCs6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs6PFSimpleSecondaryVertexHighPurBJetTags = akCs6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs6PFCombinedSecondaryVertexBJetTags = akCs6PFbTagger.CombinedSecondaryVertexBJetTags
akCs6PFCombinedSecondaryVertexV2BJetTags = akCs6PFbTagger.CombinedSecondaryVertexV2BJetTags
akCs6PFJetBProbabilityBJetTags = akCs6PFbTagger.JetBProbabilityBJetTags
akCs6PFSoftPFMuonByPtBJetTags = akCs6PFbTagger.SoftPFMuonByPtBJetTags
akCs6PFSoftPFMuonByIP3dBJetTags = akCs6PFbTagger.SoftPFMuonByIP3dBJetTags
akCs6PFTrackCountingHighEffBJetTags = akCs6PFbTagger.TrackCountingHighEffBJetTags
akCs6PFTrackCountingHighPurBJetTags = akCs6PFbTagger.TrackCountingHighPurBJetTags
akCs6PFPatJetPartonAssociationLegacy = akCs6PFbTagger.PatJetPartonAssociationLegacy

akCs6PFImpactParameterTagInfos = akCs6PFbTagger.ImpactParameterTagInfos
akCs6PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs6PFJetProbabilityBJetTags = akCs6PFbTagger.JetProbabilityBJetTags

akCs6PFSecondaryVertexTagInfos = akCs6PFbTagger.SecondaryVertexTagInfos
akCs6PFSimpleSecondaryVertexHighEffBJetTags = akCs6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs6PFSimpleSecondaryVertexHighPurBJetTags = akCs6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs6PFCombinedSecondaryVertexBJetTags = akCs6PFbTagger.CombinedSecondaryVertexBJetTags
akCs6PFCombinedSecondaryVertexV2BJetTags = akCs6PFbTagger.CombinedSecondaryVertexV2BJetTags

akCs6PFSecondaryVertexNegativeTagInfos = akCs6PFbTagger.SecondaryVertexNegativeTagInfos
akCs6PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCs6PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs6PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCs6PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs6PFNegativeCombinedSecondaryVertexBJetTags = akCs6PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCs6PFPositiveCombinedSecondaryVertexBJetTags = akCs6PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCs6PFNegativeCombinedSecondaryVertexV2BJetTags = akCs6PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCs6PFPositiveCombinedSecondaryVertexV2BJetTags = akCs6PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCs6PFSoftPFMuonsTagInfos = akCs6PFbTagger.SoftPFMuonsTagInfos
akCs6PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs6PFSoftPFMuonBJetTags = akCs6PFbTagger.SoftPFMuonBJetTags
akCs6PFSoftPFMuonByIP3dBJetTags = akCs6PFbTagger.SoftPFMuonByIP3dBJetTags
akCs6PFSoftPFMuonByPtBJetTags = akCs6PFbTagger.SoftPFMuonByPtBJetTags
akCs6PFNegativeSoftPFMuonByPtBJetTags = akCs6PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCs6PFPositiveSoftPFMuonByPtBJetTags = akCs6PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCs6PFPatJetFlavourIdLegacy = cms.Sequence(akCs6PFPatJetPartonAssociationLegacy*akCs6PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs6PFPatJetFlavourAssociation = akCs6PFbTagger.PatJetFlavourAssociation
#akCs6PFPatJetFlavourId = cms.Sequence(akCs6PFPatJetPartons*akCs6PFPatJetFlavourAssociation)

akCs6PFJetBtaggingIP       = cms.Sequence(akCs6PFImpactParameterTagInfos *
            (akCs6PFTrackCountingHighEffBJetTags +
             akCs6PFTrackCountingHighPurBJetTags +
             akCs6PFJetProbabilityBJetTags +
             akCs6PFJetBProbabilityBJetTags 
            )
            )

akCs6PFJetBtaggingSV = cms.Sequence(akCs6PFImpactParameterTagInfos
            *
            akCs6PFSecondaryVertexTagInfos
            * (akCs6PFSimpleSecondaryVertexHighEffBJetTags+
                akCs6PFSimpleSecondaryVertexHighPurBJetTags+
                akCs6PFCombinedSecondaryVertexBJetTags+
                akCs6PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCs6PFJetBtaggingNegSV = cms.Sequence(akCs6PFImpactParameterTagInfos
            *
            akCs6PFSecondaryVertexNegativeTagInfos
            * (akCs6PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCs6PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCs6PFNegativeCombinedSecondaryVertexBJetTags+
                akCs6PFPositiveCombinedSecondaryVertexBJetTags+
                akCs6PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCs6PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCs6PFJetBtaggingMu = cms.Sequence(akCs6PFSoftPFMuonsTagInfos * (akCs6PFSoftPFMuonBJetTags
                +
                akCs6PFSoftPFMuonByIP3dBJetTags
                +
                akCs6PFSoftPFMuonByPtBJetTags
                +
                akCs6PFNegativeSoftPFMuonByPtBJetTags
                +
                akCs6PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs6PFJetBtagging = cms.Sequence(akCs6PFJetBtaggingIP
            *akCs6PFJetBtaggingSV
            *akCs6PFJetBtaggingNegSV
#            *akCs6PFJetBtaggingMu
            )

akCs6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs6PFJets"),
        genJetMatch          = cms.InputTag("akCs6PFmatch"),
        genPartonMatch       = cms.InputTag("akCs6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCs6PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs6PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs6PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs6PFJetBProbabilityBJetTags"),
            cms.InputTag("akCs6PFJetProbabilityBJetTags"),
            #cms.InputTag("akCs6PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs6PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs6PFJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akCs6PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs6PFJets"),
           	    R0  = cms.double( 0.6)
)
akCs6PFpatJetsWithBtagging.userData.userFloats.src += ['akCs6PFNjettiness:tau1','akCs6PFNjettiness:tau2','akCs6PFNjettiness:tau3']

akCs6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs6PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCs6PF"),
                                                             jetName = cms.untracked.string("akCs6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs6PFJetSequence_mc = cms.Sequence(
                                                  #akCs6PFclean
                                                  #*
                                                  akCs6PFmatch
                                                  *
                                                  akCs6PFparton
                                                  *
                                                  akCs6PFcorr
                                                  *
                                                  #akCs6PFJetID
                                                  #*
                                                  akCs6PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCs6PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCs6PFJetBtagging
                                                  *
                                                  akCs6PFNjettiness
                                                  *
                                                  akCs6PFpatJetsWithBtagging
                                                  *
                                                  akCs6PFJetAnalyzer
                                                  )

akCs6PFJetSequence_data = cms.Sequence(akCs6PFcorr
                                                    *
                                                    #akCs6PFJetID
                                                    #*
                                                    akCs6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCs6PFJetBtagging
                                                    *
                                                    akCs6PFNjettiness 
                                                    *
                                                    akCs6PFpatJetsWithBtagging
                                                    *
                                                    akCs6PFJetAnalyzer
                                                    )

akCs6PFJetSequence_jec = cms.Sequence(akCs6PFJetSequence_mc)
akCs6PFJetSequence_mb = cms.Sequence(akCs6PFJetSequence_mc)

akCs6PFJetSequence = cms.Sequence(akCs6PFJetSequence_data)
