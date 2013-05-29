## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

from PhysicsTools.PatAlgos.tools.coreTools import *
removeCleaning(process)

## let it run
process.p = cms.Path(
    process.patDefaultSequence
    )

process.maxEvents.input     = 1000
process.options.wantSummary = True 
