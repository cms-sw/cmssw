import FWCore.ParameterSet.Config as cms

correctionType = "Prompt2015"
# THE FOLLOWING VALUES ARE THOSE FOR 2015 PROMPT RECO 
# the array follows this parametrization:
#  (0.0    <= aeta && aeta < 1.0    && bad )
#  (0.0    <= aeta && aeta < 1.0    && gold)
#  (1.0    <= aeta && aeta < 1.4442 && bad )
#  (1.0    <= aeta && aeta < 1.4442 && gold)
#  (1.566  <= aeta && aeta < 2.0    && bad )
#  (1.566  <= aeta && aeta < 2.0    && gold)
#  (2.0    <= aeta && aeta < 2.5    && bad )
#  (2.0    <= aeta && aeta < 2.5    && gold)
#use the 1.566-2.0 also for the 1.4442-1.566 area, in the lack of a new correction
#  (1.4442 <= aeta && aeta < 1.566  && bad )
#  (1.4442 <= aeta && aeta < 1.566  && gold)
smearingList = {"Prompt2015":cms.vdouble(0.013654,0.014142,0.020859,0.017120,0.028083,0.027289,0.031793,0.030831,0.028083, 0.027289)}
scaleList    = {"Prompt2015":cms.vdouble(0.99544,0.99882,0.99662,1.0065,0.98633,0.99536,0.97859,0.98567,0.98633, 0.99536)}

calibratedElectrons = cms.EDProducer("CalibratedElectronProducerRun2",

                                     # input collections
                                     electrons = cms.InputTag('gedGsfElectrons'),
                                     gbrForestName = cms.string("gedelectron_p4combination_25ns"),

                                     # data or MC corrections
                                     # if isMC is false, data corrections are applied
                                     isMC = cms.bool(False),
    
                                     # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                     isSynchronization = cms.bool(False),
                                     
                                     smearings = smearingList[correctionType],
                                     scales = scaleList[correctionType]
                                     )


calibratedPatElectrons = cms.EDProducer("CalibratedPatElectronProducerRun2",
                                        
                                        # input collections
                                        electrons = cms.InputTag('slimmedElectrons'),
                                        gbrForestName = cms.string("gedelectron_p4combination_25ns"),
                                        
                                        # data or MC corrections
                                        # if isMC is false, data corrections are applied
                                        isMC = cms.bool(False),
                                        
                                        # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                        isSynchronization = cms.bool(False),

                                        smearings = smearingList[correctionType],
                                        scales = scaleList[correctionType]
                                        )


