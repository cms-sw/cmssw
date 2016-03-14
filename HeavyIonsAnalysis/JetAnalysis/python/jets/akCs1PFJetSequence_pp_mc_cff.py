

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs1PFJets"),
    matched = cms.InputTag("ak1GenJets"),
    maxDeltaR = 0.1
    )

akCs1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs1PFJets")
                                                        )

akCs1PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs1PFJets"),
    payload = "AK1PF_offline"
    )

akCs1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs1CaloJets'))

#akCs1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1GenJets'))

akCs1PFbTagger = bTaggers("akCs1PF",0.1)

#create objects locally since they dont load properly otherwise
#akCs1PFmatch = akCs1PFbTagger.match
akCs1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs1PFJets"), matched = cms.InputTag("genParticles"))
akCs1PFPatJetFlavourAssociationLegacy = akCs1PFbTagger.PatJetFlavourAssociationLegacy
akCs1PFPatJetPartons = akCs1PFbTagger.PatJetPartons
akCs1PFJetTracksAssociatorAtVertex = akCs1PFbTagger.JetTracksAssociatorAtVertex
akCs1PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs1PFSimpleSecondaryVertexHighEffBJetTags = akCs1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs1PFSimpleSecondaryVertexHighPurBJetTags = akCs1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs1PFCombinedSecondaryVertexBJetTags = akCs1PFbTagger.CombinedSecondaryVertexBJetTags
akCs1PFCombinedSecondaryVertexV2BJetTags = akCs1PFbTagger.CombinedSecondaryVertexV2BJetTags
akCs1PFJetBProbabilityBJetTags = akCs1PFbTagger.JetBProbabilityBJetTags
akCs1PFSoftPFMuonByPtBJetTags = akCs1PFbTagger.SoftPFMuonByPtBJetTags
akCs1PFSoftPFMuonByIP3dBJetTags = akCs1PFbTagger.SoftPFMuonByIP3dBJetTags
akCs1PFTrackCountingHighEffBJetTags = akCs1PFbTagger.TrackCountingHighEffBJetTags
akCs1PFTrackCountingHighPurBJetTags = akCs1PFbTagger.TrackCountingHighPurBJetTags
akCs1PFPatJetPartonAssociationLegacy = akCs1PFbTagger.PatJetPartonAssociationLegacy

akCs1PFImpactParameterTagInfos = akCs1PFbTagger.ImpactParameterTagInfos
akCs1PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs1PFJetProbabilityBJetTags = akCs1PFbTagger.JetProbabilityBJetTags

akCs1PFSecondaryVertexTagInfos = akCs1PFbTagger.SecondaryVertexTagInfos
akCs1PFSimpleSecondaryVertexHighEffBJetTags = akCs1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs1PFSimpleSecondaryVertexHighPurBJetTags = akCs1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs1PFCombinedSecondaryVertexBJetTags = akCs1PFbTagger.CombinedSecondaryVertexBJetTags
akCs1PFCombinedSecondaryVertexV2BJetTags = akCs1PFbTagger.CombinedSecondaryVertexV2BJetTags

akCs1PFSecondaryVertexNegativeTagInfos = akCs1PFbTagger.SecondaryVertexNegativeTagInfos
akCs1PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCs1PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs1PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCs1PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs1PFNegativeCombinedSecondaryVertexBJetTags = akCs1PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCs1PFPositiveCombinedSecondaryVertexBJetTags = akCs1PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCs1PFNegativeCombinedSecondaryVertexV2BJetTags = akCs1PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCs1PFPositiveCombinedSecondaryVertexV2BJetTags = akCs1PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCs1PFSoftPFMuonsTagInfos = akCs1PFbTagger.SoftPFMuonsTagInfos
akCs1PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs1PFSoftPFMuonBJetTags = akCs1PFbTagger.SoftPFMuonBJetTags
akCs1PFSoftPFMuonByIP3dBJetTags = akCs1PFbTagger.SoftPFMuonByIP3dBJetTags
akCs1PFSoftPFMuonByPtBJetTags = akCs1PFbTagger.SoftPFMuonByPtBJetTags
akCs1PFNegativeSoftPFMuonByPtBJetTags = akCs1PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCs1PFPositiveSoftPFMuonByPtBJetTags = akCs1PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCs1PFPatJetFlavourIdLegacy = cms.Sequence(akCs1PFPatJetPartonAssociationLegacy*akCs1PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs1PFPatJetFlavourAssociation = akCs1PFbTagger.PatJetFlavourAssociation
#akCs1PFPatJetFlavourId = cms.Sequence(akCs1PFPatJetPartons*akCs1PFPatJetFlavourAssociation)

akCs1PFJetBtaggingIP       = cms.Sequence(akCs1PFImpactParameterTagInfos *
            (akCs1PFTrackCountingHighEffBJetTags +
             akCs1PFTrackCountingHighPurBJetTags +
             akCs1PFJetProbabilityBJetTags +
             akCs1PFJetBProbabilityBJetTags 
            )
            )

akCs1PFJetBtaggingSV = cms.Sequence(akCs1PFImpactParameterTagInfos
            *
            akCs1PFSecondaryVertexTagInfos
            * (akCs1PFSimpleSecondaryVertexHighEffBJetTags+
                akCs1PFSimpleSecondaryVertexHighPurBJetTags+
                akCs1PFCombinedSecondaryVertexBJetTags+
                akCs1PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCs1PFJetBtaggingNegSV = cms.Sequence(akCs1PFImpactParameterTagInfos
            *
            akCs1PFSecondaryVertexNegativeTagInfos
            * (akCs1PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCs1PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCs1PFNegativeCombinedSecondaryVertexBJetTags+
                akCs1PFPositiveCombinedSecondaryVertexBJetTags+
                akCs1PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCs1PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCs1PFJetBtaggingMu = cms.Sequence(akCs1PFSoftPFMuonsTagInfos * (akCs1PFSoftPFMuonBJetTags
                +
                akCs1PFSoftPFMuonByIP3dBJetTags
                +
                akCs1PFSoftPFMuonByPtBJetTags
                +
                akCs1PFNegativeSoftPFMuonByPtBJetTags
                +
                akCs1PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs1PFJetBtagging = cms.Sequence(akCs1PFJetBtaggingIP
            *akCs1PFJetBtaggingSV
            *akCs1PFJetBtaggingNegSV
#            *akCs1PFJetBtaggingMu
            )

akCs1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs1PFJets"),
        genJetMatch          = cms.InputTag("akCs1PFmatch"),
        genPartonMatch       = cms.InputTag("akCs1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs1PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCs1PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs1PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs1PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs1PFJetBProbabilityBJetTags"),
            cms.InputTag("akCs1PFJetProbabilityBJetTags"),
            #cms.InputTag("akCs1PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs1PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs1PFJetID"),
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

akCs1PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs1PFJets"),
           	    R0  = cms.double( 0.1)
)
akCs1PFpatJetsWithBtagging.userData.userFloats.src += ['akCs1PFNjettiness:tau1','akCs1PFNjettiness:tau2','akCs1PFNjettiness:tau3']

akCs1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs1PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCs1PF"),
                                                             jetName = cms.untracked.string("akCs1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs1PFJetSequence_mc = cms.Sequence(
                                                  #akCs1PFclean
                                                  #*
                                                  akCs1PFmatch
                                                  *
                                                  akCs1PFparton
                                                  *
                                                  akCs1PFcorr
                                                  *
                                                  #akCs1PFJetID
                                                  #*
                                                  akCs1PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCs1PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs1PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCs1PFJetBtagging
                                                  *
                                                  akCs1PFNjettiness
                                                  *
                                                  akCs1PFpatJetsWithBtagging
                                                  *
                                                  akCs1PFJetAnalyzer
                                                  )

akCs1PFJetSequence_data = cms.Sequence(akCs1PFcorr
                                                    *
                                                    #akCs1PFJetID
                                                    #*
                                                    akCs1PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCs1PFJetBtagging
                                                    *
                                                    akCs1PFNjettiness 
                                                    *
                                                    akCs1PFpatJetsWithBtagging
                                                    *
                                                    akCs1PFJetAnalyzer
                                                    )

akCs1PFJetSequence_jec = cms.Sequence(akCs1PFJetSequence_mc)
akCs1PFJetSequence_mb = cms.Sequence(akCs1PFJetSequence_mc)

akCs1PFJetSequence = cms.Sequence(akCs1PFJetSequence_mc)
