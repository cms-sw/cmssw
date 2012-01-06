from RecoMET.METFilters.EcalDeadCellTPfilter_cfi import *

ecalDeadCellTPfilter = EcalDeadCellTPfilter.clone()
ecalDeadCellTPfilter.tpDigiCollection = cms.InputTag("ecalTPSkim")
ecalDeadCellTPfilter.etValToBeFlagged = cms.double(63.75)
ecalDeadCellTPfilter.ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB")
ecalDeadCellTPfilter.eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE")
