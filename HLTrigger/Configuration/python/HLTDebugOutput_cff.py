import FWCore.ParameterSet.Config as cms

statements = set()

# the DQM, HLTDQM and HLTMON streams have the HLT debug outputs used online; hltOutput8E29 hltOutput1E31 hltOutputHIon have the optional extra offline event content
import HLTrigger.Configuration.hltOutputMON_cff
statements.update( statement for statement in HLTrigger.Configuration.hltOutputMON_cff.block_hltOutputDQM.outputCommands     if statement.find('drop') != 0 )
statements.update( statement for statement in HLTrigger.Configuration.hltOutputMON_cff.block_hltOutputHLTDQM.outputCommands  if statement.find('drop') != 0 )
statements.update( statement for statement in HLTrigger.Configuration.hltOutputMON_cff.block_hltOutputHLTMON.outputCommands  if statement.find('drop') != 0 )
statements.update( statement for statement in HLTrigger.Configuration.hltOutputMON_cff.block_hltOutput8E29.outputCommands    if statement.find('drop') != 0 )
statements.update( statement for statement in HLTrigger.Configuration.hltOutputMON_cff.block_hltOutput1E31.outputCommands    if statement.find('drop') != 0 )
statements.update( statement for statement in HLTrigger.Configuration.hltOutputMON_cff.block_hltOutputHIon.outputCommands    if statement.find('drop') != 0 )

statements = list(statements)
statements.sort()

block_hltDebugOutput = cms.PSet(
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*', )
)
block_hltDebugOutput.outputCommands.extend( statements )
