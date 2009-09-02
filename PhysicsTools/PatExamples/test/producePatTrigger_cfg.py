# Start with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# Load the standard PAT config
process.load( "PhysicsTools.PatAlgos.patSequences_cff" )

# Define the path
process.p = cms.Path(
    process.patDefaultSequence
)

process.maxEvents.input     = 1000 # Reduce number of events for testing.
process.out.fileName        = 'edmPatTrigger.root'
process.options.wantSummary = False # to suppress the long output at the end of the job
