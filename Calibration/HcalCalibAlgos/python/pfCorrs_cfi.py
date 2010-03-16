import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
#from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

hcalPFcorrs = cms.EDAnalyzer("HcalCorrPFCalculation",
   TrackAssociatorParameterBlock,

#    ConeRadiusCm = cms.untracked.double(30.),

      associationConeSize = cms.double(52.),
        calibrationConeSize = cms.double(35.),
        trackIsolationCone = cms.double(40.),
        EcalCone = cms.double(10.),

#     energyECALmip = cms.double(1.0),
#    RespcorrAdd = cms.untracked.bool(True),
#    PFcorrAdd = cms.untracked.bool(True),
)

