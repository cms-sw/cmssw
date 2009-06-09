import FWCore.ParameterSet.Config as cms

## define pat electron analyzer
analyzePatElectron = cms.EDAnalyzer("PatElectronAnalyzer",
    ## choose mode to run in:
    ## 0:plain, 1:genMatch, 2:tagAndProbe
    mode = cms.uint32(0),

    ## input collection for electrons
    electronSrc  = cms.InputTag("selectedLayer1Electrons"),
    ## input collection for generator particles
    particleSrc  = cms.InputTag("genParticles"),
                                    
    ## parameters for genMatch mode
    genMatchMode = cms.PSet(
      ## maximal allowed value of deltaR
      maxDeltaR = cms.double(0.3)
    ),

    ## parameters for tagAndProbe mode
    tagAndProbeMode = cms.PSet(
      ## maximal allowed value of deltaM
      maxDeltaM = cms.double(0.1),
      ## maximal allowed value of isolation
      ## of the tag electron
      maxTagIso = cms.double(0.3)      
    )                                 
)
