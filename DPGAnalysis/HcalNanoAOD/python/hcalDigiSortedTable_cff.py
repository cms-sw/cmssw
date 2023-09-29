import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *

# IMPORTANT: This variable has to end in "Table"! 
#     Otherwise, the NanoAODOutputModule will ignore it by default 
#     (e.g., the `keep nanoaodFlatTable_*Table_*_*"` bit in 
#     process.NanoAODEDMEventContent.outputCommands, 
#     see cmssw source code for details on this arcane syntax)
hcalDigiSortedTable= cms.EDProducer("HcalDigiSortedTableProducer",
  tagQIE11    = cms.untracked.InputTag("hcalDigis"),
  tagQIE10    = cms.untracked.InputTag("hcalDigis"),
  tagHO       = cms.untracked.InputTag("hcalDigis"),
  #taguMNio   = cms.untracked.InputTag("hcalDigis"),
  #StoreLaser = cms.untracked.bool(False),
  #chargeSkim = cms.untracked.double(0)
)

hcalDigiSortedTableTask = cms.Task(hcalDigiSortedTable)
hcalDigiSortedTableSeq = cms.Sequence(hcalDigiSortedTable)