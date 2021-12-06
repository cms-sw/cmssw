import FWCore.ParameterSet.Config as cms

from CalibPPS.AlignmentGlobal.ppsAlignmentWorker_cfi import ppsAlignmentWorker

MEtoEDMConvertPPSAlignment = cms.EDProducer('MEtoEDMConverter',
    Name=cms.untracked.string('MEtoEDMConverter'),
    Verbosity=cms.untracked.int32(0),
    Frequency=cms.untracked.int32(50),
    MEPathToSave=cms.untracked.string('AlCaReco/PPSAlignment'),
    deleteAfterCopy=cms.untracked.bool(True)
)

taskALCARECOPPSAlignment = cms.Task(
    ppsAlignmentWorker,
    MEtoEDMConvertPPSAlignment
)
