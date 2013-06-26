import FWCore.ParameterSet.Config as cms

## define pat electron analyzer
analyzePatElectron = cms.EDAnalyzer("PatElectronAnalyzer",
    ## choose mode to run in:
    ## 0 : genMatch, 1 : tagAndProbe
    mode = cms.uint32(0),

    ## maximal pt  for the electron to be considered
    minPt = cms.double(10.),
    ## maximal eta for the electron to be considered
    maxEta= cms.double(1.4),                                    
    ## electronId to be used for probe the following
    ## types are  available:
    ##  * none
    ##  * eidRobustLoose
    ##  * eidRobustTight
    ##  * eidLoose
    ##  * eidTight
    ##  * eidRobustHighEnergy
    electronID = cms.string('none'),                                    
                                    
    ## input collection for electrons
    electronSrc  = cms.InputTag("selectedLayer1Electrons"),
    ## input collection for generator particles
    particleSrc  = cms.InputTag("genParticles"),
                                    
    ## parameters for genMatch mode
    genMatchMode = cms.PSet(
      ## maximal allowed value of deltaR
      maxDeltaR = cms.double(0.1)
    ),

    ## parameters for tagAndProbe mode
    tagAndProbeMode = cms.PSet(
      ## maximal allowed value of deltaM
      maxDeltaM = cms.double(10.),
      ## maximal allowed value of isolation
      ## of the tag electron
      maxTagIso = cms.double(1.0)      
    )                                 
)
