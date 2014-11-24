

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak5CaloJets"),
    matched = cms.InputTag("ak5HiGenJets")
    )

ak5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak5CaloJets")
                                                        )

ak5Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak5CaloJets"),
    payload = "AK5Calo_HI"
    )

ak5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak5CaloJets'))

ak5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJets'))

ak5CalobTagger = bTaggers("ak5Calo")

#create objects locally since they dont load properly otherwise
ak5Calomatch = ak5CalobTagger.match
ak5Caloparton = ak5CalobTagger.parton
ak5CaloPatJetFlavourAssociation = ak5CalobTagger.PatJetFlavourAssociation
ak5CaloJetTracksAssociatorAtVertex = ak5CalobTagger.JetTracksAssociatorAtVertex
ak5CaloSimpleSecondaryVertexHighEffBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak5CaloSimpleSecondaryVertexHighPurBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak5CaloCombinedSecondaryVertexBJetTags = ak5CalobTagger.CombinedSecondaryVertexBJetTags
ak5CaloCombinedSecondaryVertexMVABJetTags = ak5CalobTagger.CombinedSecondaryVertexMVABJetTags
ak5CaloJetBProbabilityBJetTags = ak5CalobTagger.JetBProbabilityBJetTags
ak5CaloSoftMuonByPtBJetTags = ak5CalobTagger.SoftMuonByPtBJetTags
ak5CaloSoftMuonByIP3dBJetTags = ak5CalobTagger.SoftMuonByIP3dBJetTags
ak5CaloTrackCountingHighEffBJetTags = ak5CalobTagger.TrackCountingHighEffBJetTags
ak5CaloTrackCountingHighPurBJetTags = ak5CalobTagger.TrackCountingHighPurBJetTags
ak5CaloPatJetPartonAssociation = ak5CalobTagger.PatJetPartonAssociation

ak5CaloImpactParameterTagInfos = ak5CalobTagger.ImpactParameterTagInfos
ak5CaloJetProbabilityBJetTags = ak5CalobTagger.JetProbabilityBJetTags
ak5CaloPositiveOnlyJetProbabilityJetTags = ak5CalobTagger.PositiveOnlyJetProbabilityJetTags
ak5CaloNegativeOnlyJetProbabilityJetTags = ak5CalobTagger.NegativeOnlyJetProbabilityJetTags
ak5CaloNegativeTrackCountingHighEffJetTags = ak5CalobTagger.NegativeTrackCountingHighEffJetTags
ak5CaloNegativeTrackCountingHighPur = ak5CalobTagger.NegativeTrackCountingHighPur
ak5CaloNegativeOnlyJetBProbabilityJetTags = ak5CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak5CaloPositiveOnlyJetBProbabilityJetTags = ak5CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak5CaloSecondaryVertexTagInfos = ak5CalobTagger.SecondaryVertexTagInfos
ak5CaloSimpleSecondaryVertexHighEffBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak5CaloSimpleSecondaryVertexHighPurBJetTags = ak5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak5CaloCombinedSecondaryVertexBJetTags = ak5CalobTagger.CombinedSecondaryVertexBJetTags
ak5CaloCombinedSecondaryVertexMVABJetTags = ak5CalobTagger.CombinedSecondaryVertexMVABJetTags

ak5CaloSecondaryVertexNegativeTagInfos = ak5CalobTagger.SecondaryVertexNegativeTagInfos
ak5CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak5CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak5CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak5CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak5CaloCombinedSecondaryVertexNegativeBJetTags = ak5CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak5CaloCombinedSecondaryVertexPositiveBJetTags = ak5CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak5CaloSoftMuonTagInfos = ak5CalobTagger.SoftMuonTagInfos
ak5CaloSoftMuonBJetTags = ak5CalobTagger.SoftMuonBJetTags
ak5CaloSoftMuonByIP3dBJetTags = ak5CalobTagger.SoftMuonByIP3dBJetTags
ak5CaloSoftMuonByPtBJetTags = ak5CalobTagger.SoftMuonByPtBJetTags
ak5CaloNegativeSoftMuonByPtBJetTags = ak5CalobTagger.NegativeSoftMuonByPtBJetTags
ak5CaloPositiveSoftMuonByPtBJetTags = ak5CalobTagger.PositiveSoftMuonByPtBJetTags

ak5CaloPatJetFlavourId = cms.Sequence(ak5CaloPatJetPartonAssociation*ak5CaloPatJetFlavourAssociation)

ak5CaloJetBtaggingIP       = cms.Sequence(ak5CaloImpactParameterTagInfos *
            (ak5CaloTrackCountingHighEffBJetTags +
             ak5CaloTrackCountingHighPurBJetTags +
             ak5CaloJetProbabilityBJetTags +
             ak5CaloJetBProbabilityBJetTags +
             ak5CaloPositiveOnlyJetProbabilityJetTags +
             ak5CaloNegativeOnlyJetProbabilityJetTags +
             ak5CaloNegativeTrackCountingHighEffJetTags +
             ak5CaloNegativeTrackCountingHighPur +
             ak5CaloNegativeOnlyJetBProbabilityJetTags +
             ak5CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak5CaloJetBtaggingSV = cms.Sequence(ak5CaloImpactParameterTagInfos
            *
            ak5CaloSecondaryVertexTagInfos
            * (ak5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak5CaloCombinedSecondaryVertexBJetTags
                +
                ak5CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak5CaloJetBtaggingNegSV = cms.Sequence(ak5CaloImpactParameterTagInfos
            *
            ak5CaloSecondaryVertexNegativeTagInfos
            * (ak5CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak5CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak5CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak5CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak5CaloJetBtaggingMu = cms.Sequence(ak5CaloSoftMuonTagInfos * (ak5CaloSoftMuonBJetTags
                +
                ak5CaloSoftMuonByIP3dBJetTags
                +
                ak5CaloSoftMuonByPtBJetTags
                +
                ak5CaloNegativeSoftMuonByPtBJetTags
                +
                ak5CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak5CaloJetBtagging = cms.Sequence(ak5CaloJetBtaggingIP
            *ak5CaloJetBtaggingSV
            *ak5CaloJetBtaggingNegSV
            *ak5CaloJetBtaggingMu
            )

ak5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak5CaloJets"),
        genJetMatch          = cms.InputTag("ak5Calomatch"),
        genPartonMatch       = cms.InputTag("ak5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak5Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak5CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak5CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak5CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak5CaloJetProbabilityBJetTags"),
            cms.InputTag("ak5CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak5CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak5CaloJetID"),
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

ak5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak5CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak5CaloJetSequence_mc = cms.Sequence(
                                                  ak5Caloclean
                                                  *
                                                  ak5Calomatch
                                                  *
                                                  ak5Caloparton
                                                  *
                                                  ak5Calocorr
                                                  *
                                                  ak5CaloJetID
                                                  *
                                                  ak5CaloPatJetFlavourId
                                                  *
                                                  ak5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak5CaloJetBtagging
                                                  *
                                                  ak5CalopatJetsWithBtagging
                                                  *
                                                  ak5CaloJetAnalyzer
                                                  )

ak5CaloJetSequence_data = cms.Sequence(ak5Calocorr
                                                    *
                                                    ak5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak5CaloJetBtagging
                                                    *
                                                    ak5CalopatJetsWithBtagging
                                                    *
                                                    ak5CaloJetAnalyzer
                                                    )

ak5CaloJetSequence_jec = ak5CaloJetSequence_mc
ak5CaloJetSequence_mix = ak5CaloJetSequence_mc

ak5CaloJetSequence = cms.Sequence(ak5CaloJetSequence_mc)
