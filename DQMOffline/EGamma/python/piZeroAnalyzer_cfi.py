import FWCore.ParameterSet.Config as cms


piZeroAnalysis = cms.EDAnalyzer("PiZeroAnalyzer",

    Name = cms.untracked.string('piZeroAnalysis'),

    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                

    triggerResultsHLT = cms.InputTag("TriggerResults","","HLT"),
    triggerResultsFU = cms.InputTag("TriggerResults","","FU"),
                                

    useTriggerFiltering = cms.bool(True),                             
    standAlone = cms.bool(False),

                                
    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2
                                

                                
# parameters for pizero finding                                
    seleXtalMinEnergy = cms.double(0.0),
    clusSeedThr = cms.double(0.5),
    clusPhiSize = cms.int32(3),
    clusEtaSize = cms.int32(3),
    ParameterLogWeighted = cms.bool(True),                          
    ParameterX0 = cms.double(0.89),
    ParameterW0 = cms.double(4.2),
    ParameterT0_barl = cms.double(5.7),
    selePtGammaOne = cms.double(0.9),
    selePtGammaTwo = cms.double(0.9),                          
    seleS4S9GammaOne = cms.double(0.85),
    seleS4S9GammaTwo = cms.double(0.85),
    selePtPi0 = cms.double(2.5),
    selePi0Iso = cms.double(0.5),
    selePi0BeltDR = cms.double(0.2),
    selePi0BeltDeta = cms.double(0.05),
    seleMinvMaxPi0 = cms.double(0.5),
    seleMinvMinPi0 = cms.double(0.0),
#                                
    OutputMEsInRootFile = cms.bool(False),
 
    OutputFileName = cms.string('DQMOfflinePiZero.root'),


)
