import FWCore.ParameterSet.Config as cms

src = cms.ESSource("EcalTrivialConditionRetriever",
    producedEcalPedestals = cms.untracked.bool(False),
    intercalibConstantsFile = cms.untracked.string('/afs/cern.ch/user/g/govoni/public/SM16testCalib4CMSSW.txt'),
    producedEcalWeights = cms.untracked.bool(False),
    adcToGeVEBConstant = cms.untracked.double(0.035),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(False),
    producedEcalADCToGeVConstant = cms.untracked.bool(True)
)


