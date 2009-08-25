import FWCore.ParameterSet.Config as cms

statements = set()

# the A stream has the HLT default output, with FEDs
import HLTrigger.Configuration.hltOutputA_cff
statements.update( statement for statement in HLTrigger.Configuration.hltOutputA_cff.block_hltOutputA.outputCommands if statement.find('drop') != 0 )

statements = list(statements)
statements.sort()

block_hltDefaultOutputWithFEDs = cms.PSet(
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*', )
)
block_hltDefaultOutputWithFEDs.outputCommands.extend( statements )
