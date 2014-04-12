import FWCore.ParameterSet.Config as cms

dtT0AbsoluteReferenceCorrection = cms.EDAnalyzer("DTT0Correction",
    correctionAlgo = cms.string('DTT0AbsoluteReferenceCorrection'),
    correctionAlgoConfig = cms.PSet(
        # Format "wheel station sector" (e.g. "-1 3 10")
        calibChamber = cms.string('All'),
        # T0 reference (TDC counts) 
        reference = cms.double(640.) 
    )
)
