import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *

# IMPORTANT: This variable has to end in "Table"! 
uMNioTable= cms.EDProducer("HcalUMNioTableProducer",
  tagUMNio = cms.untracked.InputTag("hcalDigis"),
)

uMNioTableTask = cms.Task(uMNioTable)
uMNioTableSeq = cms.Sequence(uMNioTable)