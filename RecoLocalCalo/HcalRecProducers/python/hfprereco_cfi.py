import FWCore.ParameterSet.Config as cms

hfprereco = cms.EDProducer("HFPreReconstructor",
    digiLabel = cms.InputTag("hcalDigis"),
    dropZSmarkedPassed = cms.bool(True),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(hfprereco, digiLabel = cms.InputTag("hcalDigis","HFQIE10DigiCollection"))
