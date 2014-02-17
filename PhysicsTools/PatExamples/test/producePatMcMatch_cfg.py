# Start with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# Load the standard PAT config
process.load( "PhysicsTools.PatAlgos.patSequences_cff" )

# Load the exercise config
process.load( "PhysicsTools.PatExamples.mcMatch_cfi" ) # The file to modify!

# Modify the default config according to needed exercise settings
# You can comment these lines in order to run the default rather than
# your OWN MC matching from PhysicsTools/PatExamples/python/mcMatching_cfi
# CAUTION: Uncommented, this does NOT run out-of-the-box!
# Own muon match
process.makeAllLayer1Muons.remove( process.muonMatch )
process.makeAllLayer1Muons += process.myMuonMatch
process.makeAllLayer1Muons.remove( process.allLayer1Muons )
process.makeAllLayer1Muons += process.allLayer1Muons
process.allLayer1Muons.genParticleMatch = "myMuonMatch"
process.allLayer1Muons.embedGenMatch = True
# Own jet match to MC jets
process.makeAllLayer1Jets.remove( process.jetGenJetMatch )
process.makeAllLayer1Jets += process.myJetGenJetMatch
process.makeAllLayer1Jets.remove( process.allLayer1Jets )
process.makeAllLayer1Jets += process.allLayer1Jets
process.allLayer1Jets.genJetMatch = "myJetGenJetMatch"

# Define the path
process.p = cms.Path(
    process.patDefaultSequence
)

process.maxEvents.input     = 1000 # Reduce number of events for testing.
process.out.fileName        = 'edmPatMcMatch.root'
process.out.outputCommands += [ 'keep recoGenParticles_genParticles_*_*' ] # Keeps the MC objects for references
process.options.wantSummary = False # to suppress the long output at the end of the job
