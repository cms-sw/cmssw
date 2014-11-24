

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak3CaloJets"),
    matched = cms.InputTag("ak3HiGenJets")
    )

ak3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak3CaloJets")
                                                        )

ak3Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak3CaloJets"),
    payload = "AK3Calo_HI"
    )

ak3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak3CaloJets'))

ak3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJets'))

ak3CalobTagger = bTaggers("ak3Calo")

#create objects locally since they dont load properly otherwise
ak3Calomatch = ak3CalobTagger.match
ak3Caloparton = ak3CalobTagger.parton
ak3CaloPatJetFlavourAssociation = ak3CalobTagger.PatJetFlavourAssociation
ak3CaloJetTracksAssociatorAtVertex = ak3CalobTagger.JetTracksAssociatorAtVertex
ak3CaloSimpleSecondaryVertexHighEffBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak3CaloSimpleSecondaryVertexHighPurBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak3CaloCombinedSecondaryVertexBJetTags = ak3CalobTagger.CombinedSecondaryVertexBJetTags
ak3CaloCombinedSecondaryVertexMVABJetTags = ak3CalobTagger.CombinedSecondaryVertexMVABJetTags
ak3CaloJetBProbabilityBJetTags = ak3CalobTagger.JetBProbabilityBJetTags
ak3CaloSoftMuonByPtBJetTags = ak3CalobTagger.SoftMuonByPtBJetTags
ak3CaloSoftMuonByIP3dBJetTags = ak3CalobTagger.SoftMuonByIP3dBJetTags
ak3CaloTrackCountingHighEffBJetTags = ak3CalobTagger.TrackCountingHighEffBJetTags
ak3CaloTrackCountingHighPurBJetTags = ak3CalobTagger.TrackCountingHighPurBJetTags
ak3CaloPatJetPartonAssociation = ak3CalobTagger.PatJetPartonAssociation

ak3CaloImpactParameterTagInfos = ak3CalobTagger.ImpactParameterTagInfos
ak3CaloJetProbabilityBJetTags = ak3CalobTagger.JetProbabilityBJetTags
ak3CaloPositiveOnlyJetProbabilityJetTags = ak3CalobTagger.PositiveOnlyJetProbabilityJetTags
ak3CaloNegativeOnlyJetProbabilityJetTags = ak3CalobTagger.NegativeOnlyJetProbabilityJetTags
ak3CaloNegativeTrackCountingHighEffJetTags = ak3CalobTagger.NegativeTrackCountingHighEffJetTags
ak3CaloNegativeTrackCountingHighPur = ak3CalobTagger.NegativeTrackCountingHighPur
ak3CaloNegativeOnlyJetBProbabilityJetTags = ak3CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak3CaloPositiveOnlyJetBProbabilityJetTags = ak3CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak3CaloSecondaryVertexTagInfos = ak3CalobTagger.SecondaryVertexTagInfos
ak3CaloSimpleSecondaryVertexHighEffBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak3CaloSimpleSecondaryVertexHighPurBJetTags = ak3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak3CaloCombinedSecondaryVertexBJetTags = ak3CalobTagger.CombinedSecondaryVertexBJetTags
ak3CaloCombinedSecondaryVertexMVABJetTags = ak3CalobTagger.CombinedSecondaryVertexMVABJetTags

ak3CaloSecondaryVertexNegativeTagInfos = ak3CalobTagger.SecondaryVertexNegativeTagInfos
ak3CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak3CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak3CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak3CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak3CaloCombinedSecondaryVertexNegativeBJetTags = ak3CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak3CaloCombinedSecondaryVertexPositiveBJetTags = ak3CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak3CaloSoftMuonTagInfos = ak3CalobTagger.SoftMuonTagInfos
ak3CaloSoftMuonBJetTags = ak3CalobTagger.SoftMuonBJetTags
ak3CaloSoftMuonByIP3dBJetTags = ak3CalobTagger.SoftMuonByIP3dBJetTags
ak3CaloSoftMuonByPtBJetTags = ak3CalobTagger.SoftMuonByPtBJetTags
ak3CaloNegativeSoftMuonByPtBJetTags = ak3CalobTagger.NegativeSoftMuonByPtBJetTags
ak3CaloPositiveSoftMuonByPtBJetTags = ak3CalobTagger.PositiveSoftMuonByPtBJetTags

ak3CaloPatJetFlavourId = cms.Sequence(ak3CaloPatJetPartonAssociation*ak3CaloPatJetFlavourAssociation)

ak3CaloJetBtaggingIP       = cms.Sequence(ak3CaloImpactParameterTagInfos *
            (ak3CaloTrackCountingHighEffBJetTags +
             ak3CaloTrackCountingHighPurBJetTags +
             ak3CaloJetProbabilityBJetTags +
             ak3CaloJetBProbabilityBJetTags +
             ak3CaloPositiveOnlyJetProbabilityJetTags +
             ak3CaloNegativeOnlyJetProbabilityJetTags +
             ak3CaloNegativeTrackCountingHighEffJetTags +
             ak3CaloNegativeTrackCountingHighPur +
             ak3CaloNegativeOnlyJetBProbabilityJetTags +
             ak3CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak3CaloJetBtaggingSV = cms.Sequence(ak3CaloImpactParameterTagInfos
            *
            ak3CaloSecondaryVertexTagInfos
            * (ak3CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak3CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak3CaloCombinedSecondaryVertexBJetTags
                +
                ak3CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak3CaloJetBtaggingNegSV = cms.Sequence(ak3CaloImpactParameterTagInfos
            *
            ak3CaloSecondaryVertexNegativeTagInfos
            * (ak3CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak3CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak3CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak3CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak3CaloJetBtaggingMu = cms.Sequence(ak3CaloSoftMuonTagInfos * (ak3CaloSoftMuonBJetTags
                +
                ak3CaloSoftMuonByIP3dBJetTags
                +
                ak3CaloSoftMuonByPtBJetTags
                +
                ak3CaloNegativeSoftMuonByPtBJetTags
                +
                ak3CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak3CaloJetBtagging = cms.Sequence(ak3CaloJetBtaggingIP
            *ak3CaloJetBtaggingSV
            *ak3CaloJetBtaggingNegSV
            *ak3CaloJetBtaggingMu
            )

ak3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak3CaloJets"),
        genJetMatch          = cms.InputTag("ak3Calomatch"),
        genPartonMatch       = cms.InputTag("ak3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak3Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak3CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak3CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak3CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak3CaloJetProbabilityBJetTags"),
            cms.InputTag("ak3CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak3CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak3CaloJetID"),
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

ak3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("ak3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak3CaloJetSequence_mc = cms.Sequence(
                                                  ak3Caloclean
                                                  *
                                                  ak3Calomatch
                                                  *
                                                  ak3Caloparton
                                                  *
                                                  ak3Calocorr
                                                  *
                                                  ak3CaloJetID
                                                  *
                                                  ak3CaloPatJetFlavourId
                                                  *
                                                  ak3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak3CaloJetBtagging
                                                  *
                                                  ak3CalopatJetsWithBtagging
                                                  *
                                                  ak3CaloJetAnalyzer
                                                  )

ak3CaloJetSequence_data = cms.Sequence(ak3Calocorr
                                                    *
                                                    ak3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak3CaloJetBtagging
                                                    *
                                                    ak3CalopatJetsWithBtagging
                                                    *
                                                    ak3CaloJetAnalyzer
                                                    )

ak3CaloJetSequence_jec = ak3CaloJetSequence_mc
ak3CaloJetSequence_mix = ak3CaloJetSequence_mc

ak3CaloJetSequence = cms.Sequence(ak3CaloJetSequence_mc)
