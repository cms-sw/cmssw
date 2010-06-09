import FWCore.ParameterSet.Config as cms

gsfElectrons = cms.EDProducer('CorrectedElectronsProducer',
                              electronCollection = cms.InputTag("gsfElectrons"),
                              scPositionCorrectionEBPlus  = cms.vdouble(0., 0., 0.),
                              scPositionCorrectionEBMinus = cms.vdouble(0., 0., 0.),
                              scPositionCorrectionEEPlus  = cms.vdouble(0., 0., 0.),
                              scPositionCorrectionEEMinus = cms.vdouble(0., 0., 0.)
                              )
