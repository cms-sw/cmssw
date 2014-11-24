

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak6CaloJets"),
    matched = cms.InputTag("ak6HiGenJets")
    )

ak6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak6CaloJets")
                                                        )

ak6Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak6CaloJets"),
    payload = "AK6Calo_HI"
    )

ak6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak6CaloJets'))

ak6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

ak6CalobTagger = bTaggers("ak6Calo")

#create objects locally since they dont load properly otherwise
ak6Calomatch = ak6CalobTagger.match
ak6Caloparton = ak6CalobTagger.parton
ak6CaloPatJetFlavourAssociation = ak6CalobTagger.PatJetFlavourAssociation
ak6CaloJetTracksAssociatorAtVertex = ak6CalobTagger.JetTracksAssociatorAtVertex
ak6CaloSimpleSecondaryVertexHighEffBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak6CaloSimpleSecondaryVertexHighPurBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak6CaloCombinedSecondaryVertexBJetTags = ak6CalobTagger.CombinedSecondaryVertexBJetTags
ak6CaloCombinedSecondaryVertexMVABJetTags = ak6CalobTagger.CombinedSecondaryVertexMVABJetTags
ak6CaloJetBProbabilityBJetTags = ak6CalobTagger.JetBProbabilityBJetTags
ak6CaloSoftMuonByPtBJetTags = ak6CalobTagger.SoftMuonByPtBJetTags
ak6CaloSoftMuonByIP3dBJetTags = ak6CalobTagger.SoftMuonByIP3dBJetTags
ak6CaloTrackCountingHighEffBJetTags = ak6CalobTagger.TrackCountingHighEffBJetTags
ak6CaloTrackCountingHighPurBJetTags = ak6CalobTagger.TrackCountingHighPurBJetTags
ak6CaloPatJetPartonAssociation = ak6CalobTagger.PatJetPartonAssociation

ak6CaloImpactParameterTagInfos = ak6CalobTagger.ImpactParameterTagInfos
ak6CaloJetProbabilityBJetTags = ak6CalobTagger.JetProbabilityBJetTags
ak6CaloPositiveOnlyJetProbabilityJetTags = ak6CalobTagger.PositiveOnlyJetProbabilityJetTags
ak6CaloNegativeOnlyJetProbabilityJetTags = ak6CalobTagger.NegativeOnlyJetProbabilityJetTags
ak6CaloNegativeTrackCountingHighEffJetTags = ak6CalobTagger.NegativeTrackCountingHighEffJetTags
ak6CaloNegativeTrackCountingHighPur = ak6CalobTagger.NegativeTrackCountingHighPur
ak6CaloNegativeOnlyJetBProbabilityJetTags = ak6CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak6CaloPositiveOnlyJetBProbabilityJetTags = ak6CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak6CaloSecondaryVertexTagInfos = ak6CalobTagger.SecondaryVertexTagInfos
ak6CaloSimpleSecondaryVertexHighEffBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak6CaloSimpleSecondaryVertexHighPurBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak6CaloCombinedSecondaryVertexBJetTags = ak6CalobTagger.CombinedSecondaryVertexBJetTags
ak6CaloCombinedSecondaryVertexMVABJetTags = ak6CalobTagger.CombinedSecondaryVertexMVABJetTags

ak6CaloSecondaryVertexNegativeTagInfos = ak6CalobTagger.SecondaryVertexNegativeTagInfos
ak6CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak6CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak6CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak6CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak6CaloCombinedSecondaryVertexNegativeBJetTags = ak6CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak6CaloCombinedSecondaryVertexPositiveBJetTags = ak6CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak6CaloSoftMuonTagInfos = ak6CalobTagger.SoftMuonTagInfos
ak6CaloSoftMuonBJetTags = ak6CalobTagger.SoftMuonBJetTags
ak6CaloSoftMuonByIP3dBJetTags = ak6CalobTagger.SoftMuonByIP3dBJetTags
ak6CaloSoftMuonByPtBJetTags = ak6CalobTagger.SoftMuonByPtBJetTags
ak6CaloNegativeSoftMuonByPtBJetTags = ak6CalobTagger.NegativeSoftMuonByPtBJetTags
ak6CaloPositiveSoftMuonByPtBJetTags = ak6CalobTagger.PositiveSoftMuonByPtBJetTags

ak6CaloPatJetFlavourId = cms.Sequence(ak6CaloPatJetPartonAssociation*ak6CaloPatJetFlavourAssociation)

ak6CaloJetBtaggingIP       = cms.Sequence(ak6CaloImpactParameterTagInfos *
            (ak6CaloTrackCountingHighEffBJetTags +
             ak6CaloTrackCountingHighPurBJetTags +
             ak6CaloJetProbabilityBJetTags +
             ak6CaloJetBProbabilityBJetTags +
             ak6CaloPositiveOnlyJetProbabilityJetTags +
             ak6CaloNegativeOnlyJetProbabilityJetTags +
             ak6CaloNegativeTrackCountingHighEffJetTags +
             ak6CaloNegativeTrackCountingHighPur +
             ak6CaloNegativeOnlyJetBProbabilityJetTags +
             ak6CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak6CaloJetBtaggingSV = cms.Sequence(ak6CaloImpactParameterTagInfos
            *
            ak6CaloSecondaryVertexTagInfos
            * (ak6CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak6CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak6CaloCombinedSecondaryVertexBJetTags
                +
                ak6CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak6CaloJetBtaggingNegSV = cms.Sequence(ak6CaloImpactParameterTagInfos
            *
            ak6CaloSecondaryVertexNegativeTagInfos
            * (ak6CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak6CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak6CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak6CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak6CaloJetBtaggingMu = cms.Sequence(ak6CaloSoftMuonTagInfos * (ak6CaloSoftMuonBJetTags
                +
                ak6CaloSoftMuonByIP3dBJetTags
                +
                ak6CaloSoftMuonByPtBJetTags
                +
                ak6CaloNegativeSoftMuonByPtBJetTags
                +
                ak6CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak6CaloJetBtagging = cms.Sequence(ak6CaloJetBtaggingIP
            *ak6CaloJetBtaggingSV
            *ak6CaloJetBtaggingNegSV
            *ak6CaloJetBtaggingMu
            )

ak6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak6CaloJets"),
        genJetMatch          = cms.InputTag("ak6Calomatch"),
        genPartonMatch       = cms.InputTag("ak6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak6Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak6CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak6CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak6CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak6CaloJetProbabilityBJetTags"),
            cms.InputTag("ak6CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak6CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak6CaloJetID"),
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

ak6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak6CaloJetSequence_mc = cms.Sequence(
                                                  ak6Caloclean
                                                  *
                                                  ak6Calomatch
                                                  *
                                                  ak6Caloparton
                                                  *
                                                  ak6Calocorr
                                                  *
                                                  ak6CaloJetID
                                                  *
                                                  ak6CaloPatJetFlavourId
                                                  *
                                                  ak6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak6CaloJetBtagging
                                                  *
                                                  ak6CalopatJetsWithBtagging
                                                  *
                                                  ak6CaloJetAnalyzer
                                                  )

ak6CaloJetSequence_data = cms.Sequence(ak6Calocorr
                                                    *
                                                    ak6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak6CaloJetBtagging
                                                    *
                                                    ak6CalopatJetsWithBtagging
                                                    *
                                                    ak6CaloJetAnalyzer
                                                    )

ak6CaloJetSequence_jec = ak6CaloJetSequence_mc
ak6CaloJetSequence_mix = ak6CaloJetSequence_mc

ak6CaloJetSequence = cms.Sequence(ak6CaloJetSequence_mc)
