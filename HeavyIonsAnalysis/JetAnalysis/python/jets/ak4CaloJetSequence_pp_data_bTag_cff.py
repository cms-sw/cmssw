

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4CaloJets"),
    matched = cms.InputTag("ak4HiGenJets")
    )

ak4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak4CaloJets")
                                                        )

ak4Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak4CaloJets"),
    payload = "AK4Calo_HI"
    )

ak4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak4CaloJets'))

ak4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

ak4CalobTagger = bTaggers("ak4Calo")

#create objects locally since they dont load properly otherwise
ak4Calomatch = ak4CalobTagger.match
ak4Caloparton = ak4CalobTagger.parton
ak4CaloPatJetFlavourAssociation = ak4CalobTagger.PatJetFlavourAssociation
ak4CaloJetTracksAssociatorAtVertex = ak4CalobTagger.JetTracksAssociatorAtVertex
ak4CaloSimpleSecondaryVertexHighEffBJetTags = ak4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak4CaloSimpleSecondaryVertexHighPurBJetTags = ak4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak4CaloCombinedSecondaryVertexBJetTags = ak4CalobTagger.CombinedSecondaryVertexBJetTags
ak4CaloCombinedSecondaryVertexMVABJetTags = ak4CalobTagger.CombinedSecondaryVertexMVABJetTags
ak4CaloJetBProbabilityBJetTags = ak4CalobTagger.JetBProbabilityBJetTags
ak4CaloSoftMuonByPtBJetTags = ak4CalobTagger.SoftMuonByPtBJetTags
ak4CaloSoftMuonByIP3dBJetTags = ak4CalobTagger.SoftMuonByIP3dBJetTags
ak4CaloTrackCountingHighEffBJetTags = ak4CalobTagger.TrackCountingHighEffBJetTags
ak4CaloTrackCountingHighPurBJetTags = ak4CalobTagger.TrackCountingHighPurBJetTags
ak4CaloPatJetPartonAssociation = ak4CalobTagger.PatJetPartonAssociation

ak4CaloImpactParameterTagInfos = ak4CalobTagger.ImpactParameterTagInfos
ak4CaloJetProbabilityBJetTags = ak4CalobTagger.JetProbabilityBJetTags
ak4CaloPositiveOnlyJetProbabilityJetTags = ak4CalobTagger.PositiveOnlyJetProbabilityJetTags
ak4CaloNegativeOnlyJetProbabilityJetTags = ak4CalobTagger.NegativeOnlyJetProbabilityJetTags
ak4CaloNegativeTrackCountingHighEffJetTags = ak4CalobTagger.NegativeTrackCountingHighEffJetTags
ak4CaloNegativeTrackCountingHighPur = ak4CalobTagger.NegativeTrackCountingHighPur
ak4CaloNegativeOnlyJetBProbabilityJetTags = ak4CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak4CaloPositiveOnlyJetBProbabilityJetTags = ak4CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak4CaloSecondaryVertexTagInfos = ak4CalobTagger.SecondaryVertexTagInfos
ak4CaloSimpleSecondaryVertexHighEffBJetTags = ak4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak4CaloSimpleSecondaryVertexHighPurBJetTags = ak4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak4CaloCombinedSecondaryVertexBJetTags = ak4CalobTagger.CombinedSecondaryVertexBJetTags
ak4CaloCombinedSecondaryVertexMVABJetTags = ak4CalobTagger.CombinedSecondaryVertexMVABJetTags

ak4CaloSecondaryVertexNegativeTagInfos = ak4CalobTagger.SecondaryVertexNegativeTagInfos
ak4CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak4CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak4CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak4CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak4CaloCombinedSecondaryVertexNegativeBJetTags = ak4CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak4CaloCombinedSecondaryVertexPositiveBJetTags = ak4CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak4CaloSoftMuonTagInfos = ak4CalobTagger.SoftMuonTagInfos
ak4CaloSoftMuonBJetTags = ak4CalobTagger.SoftMuonBJetTags
ak4CaloSoftMuonByIP3dBJetTags = ak4CalobTagger.SoftMuonByIP3dBJetTags
ak4CaloSoftMuonByPtBJetTags = ak4CalobTagger.SoftMuonByPtBJetTags
ak4CaloNegativeSoftMuonByPtBJetTags = ak4CalobTagger.NegativeSoftMuonByPtBJetTags
ak4CaloPositiveSoftMuonByPtBJetTags = ak4CalobTagger.PositiveSoftMuonByPtBJetTags

ak4CaloPatJetFlavourId = cms.Sequence(ak4CaloPatJetPartonAssociation*ak4CaloPatJetFlavourAssociation)

ak4CaloJetBtaggingIP       = cms.Sequence(ak4CaloImpactParameterTagInfos *
            (ak4CaloTrackCountingHighEffBJetTags +
             ak4CaloTrackCountingHighPurBJetTags +
             ak4CaloJetProbabilityBJetTags +
             ak4CaloJetBProbabilityBJetTags +
             ak4CaloPositiveOnlyJetProbabilityJetTags +
             ak4CaloNegativeOnlyJetProbabilityJetTags +
             ak4CaloNegativeTrackCountingHighEffJetTags +
             ak4CaloNegativeTrackCountingHighPur +
             ak4CaloNegativeOnlyJetBProbabilityJetTags +
             ak4CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak4CaloJetBtaggingSV = cms.Sequence(ak4CaloImpactParameterTagInfos
            *
            ak4CaloSecondaryVertexTagInfos
            * (ak4CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak4CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak4CaloCombinedSecondaryVertexBJetTags
                +
                ak4CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak4CaloJetBtaggingNegSV = cms.Sequence(ak4CaloImpactParameterTagInfos
            *
            ak4CaloSecondaryVertexNegativeTagInfos
            * (ak4CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak4CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak4CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak4CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak4CaloJetBtaggingMu = cms.Sequence(ak4CaloSoftMuonTagInfos * (ak4CaloSoftMuonBJetTags
                +
                ak4CaloSoftMuonByIP3dBJetTags
                +
                ak4CaloSoftMuonByPtBJetTags
                +
                ak4CaloNegativeSoftMuonByPtBJetTags
                +
                ak4CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak4CaloJetBtagging = cms.Sequence(ak4CaloJetBtaggingIP
            *ak4CaloJetBtaggingSV
            *ak4CaloJetBtaggingNegSV
            *ak4CaloJetBtaggingMu
            )

ak4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak4CaloJets"),
        genJetMatch          = cms.InputTag("ak4Calomatch"),
        genPartonMatch       = cms.InputTag("ak4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak4Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak4CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak4CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak4CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak4CaloJetProbabilityBJetTags"),
            cms.InputTag("ak4CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak4CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak4CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        embedCaloTowers = False,
        embedPFCandidates = True
        )

ak4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak4CaloJetSequence_mc = cms.Sequence(
                                                  ak4Caloclean
                                                  *
                                                  ak4Calomatch
                                                  *
                                                  ak4Caloparton
                                                  *
                                                  ak4Calocorr
                                                  *
                                                  ak4CaloJetID
                                                  *
                                                  ak4CaloPatJetFlavourId
                                                  *
                                                  ak4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak4CaloJetBtagging
                                                  *
                                                  ak4CalopatJetsWithBtagging
                                                  *
                                                  ak4CaloJetAnalyzer
                                                  )

ak4CaloJetSequence_data = cms.Sequence(ak4Calocorr
                                                    *
                                                    ak4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak4CaloJetBtagging
                                                    *
                                                    ak4CalopatJetsWithBtagging
                                                    *
                                                    ak4CaloJetAnalyzer
                                                    )

ak4CaloJetSequence_jec = ak4CaloJetSequence_mc
ak4CaloJetSequence_mix = ak4CaloJetSequence_mc

ak4CaloJetSequence = cms.Sequence(ak4CaloJetSequence_data)
