import FWCore.ParameterSet.Config as cms

gsfElectrons = cms.EDProducer('CorrectedElectronsProducer',
                              electronCollection = cms.InputTag("gsfElectrons"),
                              scPositionCorrectionEBPlus  = cms.vdouble(0., 0., 0.),
                              scPositionCorrectionEBMinus = cms.vdouble(0., 0., 0.),
                              scPositionCorrectionEEPlus  = cms.vdouble(0.52, -0.81, 0.81),
                              scPositionCorrectionEEMinus = cms.vdouble(-0.02, -0.81, -0.94)
                              )
