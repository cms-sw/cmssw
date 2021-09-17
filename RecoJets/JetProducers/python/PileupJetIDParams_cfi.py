import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.PileupJetIDCutParams_cfi import *

####################################################################################################################
full_81x_chs = cms.PSet(
        impactParTkThreshold = cms.double(1.),
        cutBased = cms.bool(False),
        etaBinnedWeights = cms.bool(True),
        tmvaMethod = cms.string("JetIDMVAHighPt"),
        version = cms.int32(-1),
        nEtaBins = cms.int32(4),
        trainings = cms.VPSet(
            cms.PSet(
                jEtaMin = cms.double(0.),
                jEtaMax = cms.double(2.5),
                tmvaWeights  = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80XvarFix_Eta0to2p5_BDT.weights.xml.gz"),
                tmvaVariables = cms.vstring(
                    "nvtx",
                    "dR2Mean"     ,
                    "nParticles"     ,
                    "nCharged" ,
                    "majW" ,
                    "minW",
                    "frac01"  ,
                    "frac02"      ,
                    "frac03"   ,
                    "frac04"   ,
                    "ptD"   ,
                    "beta"   ,
                    "pull"   ,
                    "jetR"   ,
                    "jetRchg"   ,
                )
                ),
            cms.PSet(
                jEtaMin = cms.double(2.5),
                jEtaMax = cms.double(2.75),
                tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80XvarFix_Eta2p5to2p75_BDT.weights.xml.gz"),
                tmvaVariables = cms.vstring(
                    "nvtx",
                    "dR2Mean"     ,
                    "nParticles"     ,
                    "nCharged" ,
                    "majW" ,
                    "minW",
                    "frac01"  ,
                    "frac02"      ,
                    "frac03"   ,
                    "frac04"   ,
                    "ptD"   ,
                    "beta"   ,
                    "pull"   ,
                    "jetR"   ,
                    "jetRchg"   ,
                )
                ),
            cms.PSet(
                jEtaMin = cms.double(2.75),
                jEtaMax = cms.double(3.),
                tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80XvarFix_Eta2p75to3_BDT.weights.xml.gz"),
                tmvaVariables = cms.vstring(
                    "nvtx",
                    "dR2Mean"     ,
                    "nParticles"     ,
                    "nCharged" ,
                    "majW" ,
                    "minW",
                    "frac01"  ,
                    "frac02"      ,
                    "frac03"   ,
                    "frac04"   ,
                    "ptD"   ,
                    "beta"   ,
                    "pull"   ,
                    "jetR"   ,
                    "jetRchg"   ,
                )
                ),
            cms.PSet(
                jEtaMin = cms.double(3.),
                jEtaMax = cms.double(5.),
                tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80XvarFix_Eta3to5_BDT.weights.xml.gz"),
                tmvaVariables = cms.vstring(
                    "nvtx",
                    "dR2Mean"     ,
                    "nParticles"     ,
                    "majW" ,
                    "minW",
                    "frac01"  ,
                    "frac02"      ,
                    "frac03"   ,
                    "frac04"   ,
                    "ptD"   ,
                    "pull"   ,
                    "jetR"   ,
                )
                ),
        ),
        tmvaSpectators = cms.vstring(
            "jetPt"   ,
            "jetEta"   ,
        ),
        JetIdParams = full_81x_chs_wp,
        label = cms.string("full")
)

####################################################################################################################
trainingVariables_102X_Eta0To3 = [
                            "nvtx"      ,
                            "beta"      ,
                            "dR2Mean"   ,
                            "frac01"    ,
                            "frac02"    ,
                            "frac03"    ,
                            "frac04"    ,
                            "majW"      ,
                            "minW"      ,
                            "jetR"      ,
                            "jetRchg"   ,
                            "nParticles",
                            "nCharged"  ,
                            "ptD"       ,
                            "pull"      ,
                            ]
trainingVariables_102X_Eta3To5 = list(trainingVariables_102X_Eta0To3)
trainingVariables_102X_Eta3To5.remove('beta')
trainingVariables_102X_Eta3To5.remove('jetRchg')
trainingVariables_102X_Eta3To5.remove('nCharged')

####################################################################################################################
full_102x_chs = full_81x_chs.clone(
    JetIdParams = full_102x_chs_wp,
    trainings = {0: dict(tmvaWeights   = "RecoJets/JetProducers/data/pileupJetId_102X_Eta0p0To2p5_chs_BDT.weights.xml.gz",
                         tmvaVariables = trainingVariables_102X_Eta0To3),
                 1: dict(tmvaWeights   = "RecoJets/JetProducers/data/pileupJetId_102X_Eta2p5To2p75_chs_BDT.weights.xml.gz",
                         tmvaVariables = trainingVariables_102X_Eta0To3),
                 2: dict(tmvaWeights   = "RecoJets/JetProducers/data/pileupJetId_102X_Eta2p75To3p0_chs_BDT.weights.xml.gz",
                         tmvaVariables = trainingVariables_102X_Eta0To3),
                 3: dict(tmvaWeights = "RecoJets/JetProducers/data/pileupJetId_102X_Eta3p0To5p0_chs_BDT.weights.xml.gz",
                         tmvaVariables = trainingVariables_102X_Eta3To5)
    }
)

####################################################################################################################
full_94x_chs = full_102x_chs.clone(JetIdParams = full_94x_chs_wp)
for train in full_94x_chs.trainings:
    train.tmvaWeights = train.tmvaWeights.value().replace("102X", "94X")

####################################################################################################################
full_106x_UL17_chs = full_102x_chs.clone(JetIdParams = full_106x_UL17_chs_wp)
for train in full_106x_UL17_chs.trainings:
    train.tmvaWeights = train.tmvaWeights.value().replace("102X", "UL17")

####################################################################################################################
full_106x_UL18_chs = full_106x_UL17_chs.clone(JetIdParams = full_106x_UL18_chs_wp)
for train in full_106x_UL18_chs.trainings:
    train.tmvaWeights = train.tmvaWeights.value().replace("UL17", "UL18")

####################################################################################################################
full_106x_UL16_chs = full_106x_UL17_chs.clone(JetIdParams = full_106x_UL16_chs_wp)
for train in full_106x_UL16_chs.trainings:
    train.tmvaWeights = train.tmvaWeights.value().replace("UL17", "UL16")

####################################################################################################################
full_106x_UL16APV_chs = full_106x_UL17_chs.clone(JetIdParams = full_106x_UL16APV_chs_wp)
for train in full_106x_UL16APV_chs.trainings:
    train.tmvaWeights = train.tmvaWeights.value().replace("UL17", "UL16APV")

####################################################################################################################
cutbased = cms.PSet(
    impactParTkThreshold = cms.double(1.),
    cutBased = cms.bool(True),
    JetIdParams = EmptyCutBased_wp,
    label = cms.string("cutbased")
)
