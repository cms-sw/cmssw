

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets")
    )

ak1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak1CaloJets")
                                                        )

ak1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak1CaloJets"),
    payload = "AK1Calo_HI"
    )

ak1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak1CaloJets'))

ak1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJets'))

ak1CalobTagger = bTaggers("ak1Calo")

#create objects locally since they dont load properly otherwise
ak1Calomatch = ak1CalobTagger.match
ak1Caloparton = ak1CalobTagger.parton
ak1CaloPatJetFlavourAssociation = ak1CalobTagger.PatJetFlavourAssociation
ak1CaloJetTracksAssociatorAtVertex = ak1CalobTagger.JetTracksAssociatorAtVertex
ak1CaloSimpleSecondaryVertexHighEffBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak1CaloSimpleSecondaryVertexHighPurBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak1CaloCombinedSecondaryVertexBJetTags = ak1CalobTagger.CombinedSecondaryVertexBJetTags
ak1CaloCombinedSecondaryVertexMVABJetTags = ak1CalobTagger.CombinedSecondaryVertexMVABJetTags
ak1CaloJetBProbabilityBJetTags = ak1CalobTagger.JetBProbabilityBJetTags
ak1CaloSoftMuonByPtBJetTags = ak1CalobTagger.SoftMuonByPtBJetTags
ak1CaloSoftMuonByIP3dBJetTags = ak1CalobTagger.SoftMuonByIP3dBJetTags
ak1CaloTrackCountingHighEffBJetTags = ak1CalobTagger.TrackCountingHighEffBJetTags
ak1CaloTrackCountingHighPurBJetTags = ak1CalobTagger.TrackCountingHighPurBJetTags
ak1CaloPatJetPartonAssociation = ak1CalobTagger.PatJetPartonAssociation

ak1CaloImpactParameterTagInfos = ak1CalobTagger.ImpactParameterTagInfos
ak1CaloJetProbabilityBJetTags = ak1CalobTagger.JetProbabilityBJetTags
ak1CaloPositiveOnlyJetProbabilityJetTags = ak1CalobTagger.PositiveOnlyJetProbabilityJetTags
ak1CaloNegativeOnlyJetProbabilityJetTags = ak1CalobTagger.NegativeOnlyJetProbabilityJetTags
ak1CaloNegativeTrackCountingHighEffJetTags = ak1CalobTagger.NegativeTrackCountingHighEffJetTags
ak1CaloNegativeTrackCountingHighPur = ak1CalobTagger.NegativeTrackCountingHighPur
ak1CaloNegativeOnlyJetBProbabilityJetTags = ak1CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak1CaloPositiveOnlyJetBProbabilityJetTags = ak1CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak1CaloSecondaryVertexTagInfos = ak1CalobTagger.SecondaryVertexTagInfos
ak1CaloSimpleSecondaryVertexHighEffBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak1CaloSimpleSecondaryVertexHighPurBJetTags = ak1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak1CaloCombinedSecondaryVertexBJetTags = ak1CalobTagger.CombinedSecondaryVertexBJetTags
ak1CaloCombinedSecondaryVertexMVABJetTags = ak1CalobTagger.CombinedSecondaryVertexMVABJetTags

ak1CaloSecondaryVertexNegativeTagInfos = ak1CalobTagger.SecondaryVertexNegativeTagInfos
ak1CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak1CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak1CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak1CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak1CaloCombinedSecondaryVertexNegativeBJetTags = ak1CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak1CaloCombinedSecondaryVertexPositiveBJetTags = ak1CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak1CaloSoftMuonTagInfos = ak1CalobTagger.SoftMuonTagInfos
ak1CaloSoftMuonBJetTags = ak1CalobTagger.SoftMuonBJetTags
ak1CaloSoftMuonByIP3dBJetTags = ak1CalobTagger.SoftMuonByIP3dBJetTags
ak1CaloSoftMuonByPtBJetTags = ak1CalobTagger.SoftMuonByPtBJetTags
ak1CaloNegativeSoftMuonByPtBJetTags = ak1CalobTagger.NegativeSoftMuonByPtBJetTags
ak1CaloPositiveSoftMuonByPtBJetTags = ak1CalobTagger.PositiveSoftMuonByPtBJetTags

ak1CaloPatJetFlavourId = cms.Sequence(ak1CaloPatJetPartonAssociation*ak1CaloPatJetFlavourAssociation)

ak1CaloJetBtaggingIP       = cms.Sequence(ak1CaloImpactParameterTagInfos *
            (ak1CaloTrackCountingHighEffBJetTags +
             ak1CaloTrackCountingHighPurBJetTags +
             ak1CaloJetProbabilityBJetTags +
             ak1CaloJetBProbabilityBJetTags +
             ak1CaloPositiveOnlyJetProbabilityJetTags +
             ak1CaloNegativeOnlyJetProbabilityJetTags +
             ak1CaloNegativeTrackCountingHighEffJetTags +
             ak1CaloNegativeTrackCountingHighPur +
             ak1CaloNegativeOnlyJetBProbabilityJetTags +
             ak1CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak1CaloJetBtaggingSV = cms.Sequence(ak1CaloImpactParameterTagInfos
            *
            ak1CaloSecondaryVertexTagInfos
            * (ak1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak1CaloCombinedSecondaryVertexBJetTags
                +
                ak1CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak1CaloJetBtaggingNegSV = cms.Sequence(ak1CaloImpactParameterTagInfos
            *
            ak1CaloSecondaryVertexNegativeTagInfos
            * (ak1CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak1CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak1CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak1CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak1CaloJetBtaggingMu = cms.Sequence(ak1CaloSoftMuonTagInfos * (ak1CaloSoftMuonBJetTags
                +
                ak1CaloSoftMuonByIP3dBJetTags
                +
                ak1CaloSoftMuonByPtBJetTags
                +
                ak1CaloNegativeSoftMuonByPtBJetTags
                +
                ak1CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak1CaloJetBtagging = cms.Sequence(ak1CaloJetBtaggingIP
            *ak1CaloJetBtaggingSV
            *ak1CaloJetBtaggingNegSV
            *ak1CaloJetBtaggingMu
            )

ak1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak1CaloJets"),
        genJetMatch          = cms.InputTag("ak1Calomatch"),
        genPartonMatch       = cms.InputTag("ak1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak1Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak1CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak1CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak1CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak1CaloJetProbabilityBJetTags"),
            cms.InputTag("ak1CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak1CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak1CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        embedCaloTowers = False,
        embedPFCandidates = True
        )

ak1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak1CaloJetSequence_mc = cms.Sequence(
                                                  ak1Caloclean
                                                  *
                                                  ak1Calomatch
                                                  *
                                                  ak1Caloparton
                                                  *
                                                  ak1Calocorr
                                                  *
                                                  ak1CaloJetID
                                                  *
                                                  ak1CaloPatJetFlavourId
                                                  *
                                                  ak1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak1CaloJetBtagging
                                                  *
                                                  ak1CalopatJetsWithBtagging
                                                  *
                                                  ak1CaloJetAnalyzer
                                                  )

ak1CaloJetSequence_data = cms.Sequence(ak1Calocorr
                                                    *
                                                    ak1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak1CaloJetBtagging
                                                    *
                                                    ak1CalopatJetsWithBtagging
                                                    *
                                                    ak1CaloJetAnalyzer
                                                    )

ak1CaloJetSequence_jec = ak1CaloJetSequence_mc
ak1CaloJetSequence_mix = ak1CaloJetSequence_mc

ak1CaloJetSequence = cms.Sequence(ak1CaloJetSequence_mix)
