import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *

# IMPORTANT: This variable has to end in "Table"! 
uHTRTable= cms.EDProducer("HcalUHTRTableProducer",
  InputLabel = cms.untracked.InputTag("rawDataCollector"),
  FEDs = cms.untracked.vint32(),
)

uHTRTableTask = cms.Task(uHTRTable)
uHTRTableSeq = cms.Sequence(uHTRTable)