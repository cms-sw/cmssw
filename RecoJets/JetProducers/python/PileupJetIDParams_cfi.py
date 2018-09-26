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
full_80x_chs = cms.PSet(
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
                tmvaWeights  = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80X_Eta0to2p5_BDT.weights.xml.gz"),
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
                tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80X_Eta2p5to2p75_BDT.weights.xml.gz"),
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
                tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80X_Eta2p75to3_BDT.weights.xml.gz"),
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
                tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_80X_Eta3to5_BDT.weights.xml.gz"),
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
        JetIdParams = full_80x_chs_wp,
        label = cms.string("full")
)

####################################################################################################################                                                                                      
full_76x_chs = cms.PSet(
    impactParTkThreshold = cms.double(1.) ,
    cutBased = cms.bool(False),
    etaBinnedWeights = cms.bool(True),
    nEtaBins = cms.int32(4),
    trainings = cms.VPSet(
        cms.PSet(
            jEtaMin = cms.double(0.),
            jEtaMax = cms.double(2.5),
            tmvaWeights  = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_76x_Eta0to2p5_BDT.weights.xml.gz"),
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
            tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_76x_Eta2p5to2p75_BDT.weights.xml.gz"),
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
            tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_76x_Eta2p75to3_BDT.weights.xml.gz"),
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
            tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/pileupJetId_76x_Eta3to5_BDT.weights.xml.gz"),
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
    tmvaMethod  = cms.string("JetIDMVAHighPt"),
    version = cms.int32(-1),
    tmvaSpectators = cms.vstring(
        "jetPt"   ,
        "jetEta"   ,
    ),
    JetIdParams = full_76x_chs_wp,
    label = cms.string("full")
 )
####################################################################################################################                                                                                      
full_74x_chs = cms.PSet(
    impactParTkThreshold = cms.double(1.) ,
    cutBased = cms.bool(False),
    etaBinnedWeights = cms.bool(True),
    nEtaBins = cms.int32(4),
    trainings = cms.VPSet(
        cms.PSet(
            jEtaMin = cms.double(0.),
            jEtaMax = cms.double(2.),
            tmvaWeights  = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_0_2_newNames.xml.gz"),
            tmvaVariables = cms.vstring(
                "dR2Mean"     ,
                "rho"       ,
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
                "betaStar"   ,
                "pull"   ,
                "jetR"   ,
                "jetRchg"   ,
            )
            ),
        cms.PSet(
            jEtaMin = cms.double(2.),
            jEtaMax = cms.double(2.5),
            tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_2_2p5_newNames.xml.gz"),
            tmvaVariables = cms.vstring(
                "dR2Mean"     ,
                "rho"       ,
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
                "betaStar"   ,
                "pull"   ,
                "jetR"   ,
                "jetRchg"   ,
            )
            ),
        cms.PSet(
            jEtaMin = cms.double(2.5),
            jEtaMax = cms.double(3.),
            tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_2p5_3_newNames.xml.gz"),
            tmvaVariables = cms.vstring(
                "dR2Mean"     ,
                "rho"       ,
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
                "betaStar"   ,
                "pull"   ,
                "jetR"   ,
                "jetRchg"   ,
            )
            ),
        cms.PSet(
            jEtaMin = cms.double(3.),
            jEtaMax = cms.double(5.),
            tmvaWeights   = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_3_5_newNames.xml.gz"),
            tmvaVariables = cms.vstring(
                "dR2Mean"     ,
                "rho"       ,
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
    version = cms.int32(-1),
    tmvaSpectators = cms.vstring(
        "jetPt"   ,
        "jetEta"   ,
        "nTrueInt"   ,
        "dRMatch"   ,
    ),
    JetIdParams = full_74x_chs_wp,
    label = cms.string("full")
 )
####################################################################################################################  
full_53x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 etaBinnedWeights = cms.bool(False),
 tmvaWeights = cms.FileInPath("CondFormats/JetMETObjects/data/TMVAClassificationCategory_JetID_53X_Dec2012.weights.xml"),
 tmvaMethod  = cms.string("JetIDMVAHighPt"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx"     ,
    "dZ"       , 
    "beta"     , 
    "betaStar" , 
    "nCharged" , 
    "nNeutrals", 
    "dR2Mean"  , 
    "ptD"      , 
    "frac01"   , 
    "frac02"   , 
    "frac03"   , 
    "frac04"   , 
    "frac05"   , 
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    "jetPhi"
    ),
 JetIdParams = full_53x_wp,
 label = cms.string("full53x")
 )
####################################################################################################################  
full_53x_chs = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 etaBinnedWeights = cms.bool(False),
 tmvaWeights = cms.FileInPath("CondFormats/JetMETObjects/data/TMVAClassificationCategory_JetID_53X_chs_Dec2012.weights.xml"),
 #tmvaWeights = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_53X_chs_Dec2012.weights.xml"),
 tmvaMethod  = cms.string("JetIDMVAHighPt"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx"     ,
    "dZ"       , 
    "beta"     , 
    "betaStar" , 
    "nCharged" , 
    "nNeutrals", 
    "dR2Mean"  , 
    "ptD"      , 
    "frac01"   , 
    "frac02"   , 
    "frac03"   , 
    "frac04"   , 
    "frac05"   , 
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    "jetPhi"
    ),
 JetIdParams = full_53x_chs_wp,
 label = cms.string("full")
 )
####################################################################################################################  
met_53x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 etaBinnedWeights = cms.bool(False),
 tmvaWeights = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_MET_53X_Dec2012.weights.xml.gz"),
 tmvaMethod  = cms.string("JetIDMVAMET"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx"     ,
    "jetPt"    ,
    "jetEta"   ,
    "jetPhi"   ,
    "dZ"       , 
    "beta"     , 
    "betaStar" , 
    "nCharged" , 
    "nNeutrals", 
    "dR2Mean"  , 
    "ptD"      , 
    "frac01"   , 
    "frac02"   , 
    "frac03"   , 
    "frac04"   , 
    "frac05"   , 
    ),
 tmvaSpectators = cms.vstring(),
 JetIdParams = met_53x_wp,
 label = cms.string("met53x")
 )
##################################################################################################################  
full_5x = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 etaBinnedWeights = cms.bool(False),
 tmvaWeights = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_MET_53X_Dec2012.weights.xml.gz"),
 tmvaMethod  = cms.string("BDT_fullPlusRMS"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    "dR2Mean",
    "nvtx",
    "nNeutrals",
    "beta",
    "betaStar",
    "dZ",
    "nCharged",
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    ),
 JetIdParams = full_5x_wp,
 label = cms.string("full")
 )

##################################################################################################################  
full_5x_chs = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 etaBinnedWeights = cms.bool(False),
 tmvaWeights = cms.FileInPath("RecoJets/JetProducers/data/TMVAClassification_5x_BDT_chsFullPlusRMS.weights.xml.gz"),
 tmvaMethod  = cms.string("BDT_chsFullPlusRMS"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    "dR2Mean",
    "nvtx",
    "nNeutrals",
    "beta",
    "betaStar",
    "dZ",
    "nCharged",
    ),
 tmvaSpectators = cms.vstring(
    "jetPt",
    "jetEta",
    ),
 JetIdParams = full_5x_chs_wp,
 label = cms.string("full")
 )

####################################################################################################################  
cutbased = cms.PSet( 
 impactParTkThreshold = cms.double(1.),
 cutBased = cms.bool(True),
 JetIdParams = PuJetIdCutBased_wp,
 label = cms.string("cutbased")
 )

####################################################################################################################  
PhilV1 = cms.PSet(
 impactParTkThreshold = cms.double(1.) ,
 cutBased = cms.bool(False),
 etaBinnedWeights = cms.bool(False),
 tmvaWeights = cms.FileInPath("RecoJets/JetProducers/data/mva_JetID_v1.weights.xml.gz"),
 tmvaMethod  = cms.string("JetID"),
 version = cms.int32(-1),
 tmvaVariables = cms.vstring(
    "nvtx",
    "jetPt",
    "jetEta",
    "jetPhi",
    "dZ",
    "d0",
    "beta",
    "betaStar",
    "nCharged",
    "nNeutrals",
    "dRMean",
    "frac01",
    "frac02",
    "frac03",
    "frac04",
    "frac05",
    ),
 tmvaSpectators = cms.vstring(),
 JetIdParams = JetIdParams,
 label = cms.string("philv1")
)

