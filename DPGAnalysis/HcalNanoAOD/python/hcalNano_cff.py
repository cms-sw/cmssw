from PhysicsTools.NanoAOD.common_cff import Var,CandVars
from DPGAnalysis.HcalNanoAOD.hcalRecHitTable_cff import *
from DPGAnalysis.HcalNanoAOD.hcalDigiSortedTable_cff import *
from DPGAnalysis.HcalNanoAOD.hcalDetIdTable_cff import *

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

hcalNanoTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hcalDigiSortedTableTask, 
    hcalRecHitTableTask, 
)

hcalNanoDigiTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hcalDigiSortedTableTask, 
)

hcalNanoRecHitTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hcalRecHitTableTask, 
)

# Tasks for HCAL AlCa workflows
hcalNanoPhiSymTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hbheRecHitTable,
    hfRecHitTable,
)

hcalNanoIsoTrkTask = cms.Task(
    nanoMetadata, 
    hcalDetIdTableTask, 
    hbheRecHitTable,
)
