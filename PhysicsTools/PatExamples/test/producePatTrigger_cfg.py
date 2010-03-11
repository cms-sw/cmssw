# Start with pre-defined skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# Define the path
process.p = cms.Path(
    process.patDefaultSequence
)

process.maxEvents.input     = 1000 # Reduce number of events for testing.
process.out.fileName        = 'edmPatTrigger.root'
process.options.wantSummary = False # to suppress the long output at the end of the job
