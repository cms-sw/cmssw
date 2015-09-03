#! /usr/bin/env python

import sys
import subprocess
import FWCore.ParameterSet.Config as cms

config = sys.argv[1]

def extractBlock(config, blocks, target):
  #print 'configuration: %s' % config
  #print 'blocks:        %s' % ', '.join(blocks)
  #print 'target:        %s' % target
  #print
  commands = ','.join( block + '::outputCommands' for block in blocks )
  proc = subprocess.Popen(
    "hltConfigFromDB --configName %s --noedsources --nopaths --noes --nopsets --noservices --cff --blocks %s --format python | sed -e'/^streams/,/^)/d' -e'/^datasets/,/^)/d' > %s" % (config, commands, target),
    shell  = True,
    stdin  = None,
    stdout = None,
    stderr = None,
  )
  proc.wait()

def extractBlocks(config):
  outputA    = ( 'hltOutputA', 'hltOutputPhysicsEGammaCommissioning' )
  outputALCA = ( 'hltOutputALCAPHISYM', 'hltOutputALCAP0', 'hltOutputALCALUMIPIXELS' , 'hltOutputRPCMON' )
  outputMON  = ( 'hltOutputA', 'hltOutputPhysicsEGammaCommissioning', 'hltOutputDQM', 'hltOutputHLTMonitor', 'hltOutputLookArea', 'hltOutputReleaseValidation' )
  extractBlock(config, outputA,    'hltOutputA_cff.py')
  extractBlock(config, outputALCA, 'hltOutputALCA_cff.py')
  extractBlock(config, outputMON,  'hltOutputMON_cff.py')

def makePSet(statements):
  statements = list(statements)
  statements.sort()
  block = cms.PSet(
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*', )
  )
  block.outputCommands.extend( statements )
  return block


def buildPSet(blocks):
  statements = set()
  for block in blocks:
    statements.update( statement for statement in block if statement.find('drop') != 0 )
  return makePSet(statements)


def buildPSetWithoutRAWs(blocks):
  statements = set()
  for block in blocks:
    statements.update( statement for statement in block if statement.find('drop') != 0 and statement.find('keep FEDRawDataCollection') != 0)
  return makePSet(statements)


# customisation of AOD event content, requested by David Dagenhart
def dropL1GlobalTriggerObjectMapRecord(block):
  """drop the old L1GlobalTriggerObjectMapRecord data format from the block (meant for the AOD data tier)"""
  try:
    # look for the hltL1GtObjectMap keep statement
    position = block.outputCommands.index('keep *_hltL1GtObjectMap_*_*')
  except ValueError:
    pass
  else:
    # add just after it a drop statement for the old data format
    block.outputCommands.insert(position  + 1, 'drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap_*_*')


# extract the HLT layer event content
extractBlocks( config )
import hltOutputA_cff
import hltOutputALCA_cff
import hltOutputMON_cff

# hltDebugOutput

if not hasattr(hltOutputMON_cff,'block_hltOutputA'):
  hltOutputMON_cff.block_hltOutputA = hltOutputMON_cff.block_hltOutputPhysicsEGammaCommissioning
if not hasattr(hltOutputMON_cff,'block_hltOutputDQM'):
  hltOutputMON_cff.block_hltOutputDQM = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
if not hasattr(hltOutputMON_cff,'block_hltOutputHLTMonitor'):
  hltOutputMON_cff.block_hltOutputHLTMonitor = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*' ))
if not hasattr(hltOutputMON_cff,'block_hltOutputLookArea'):
  hltOutputMON_cff.block_hltOutputLookArea = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
if not hasattr(hltOutputMON_cff,'block_hltOutputReleaseValidation'):
  hltOutputMON_cff.block_hltOutputReleaseValidation = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))

hltDebugOutputBlocks = (
  # the DQM, HLTDQM and HLTMON streams have the HLT debug outputs used online
  hltOutputMON_cff.block_hltOutputA.outputCommands,
  hltOutputMON_cff.block_hltOutputDQM.outputCommands,
  hltOutputMON_cff.block_hltOutputHLTMonitor.outputCommands,
  hltOutputMON_cff.block_hltOutputLookArea.outputCommands,
  hltOutputMON_cff.block_hltOutputReleaseValidation.outputCommands,
)
hltDebugOutputContent = buildPSet(hltDebugOutputBlocks)


# hltDebugWithAlCaOutput
if not hasattr(hltOutputALCA_cff,'block_hltOutputALCAPHISYM'):
  hltOutputALCA_cff.block_hltOutputALCAPHISYM = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
if not hasattr(hltOutputALCA_cff,'block_hltOutputALCAP0'):
  hltOutputALCA_cff.block_hltOutputALCAP0 = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
if not hasattr(hltOutputALCA_cff,'block_hltOutputALCALUMIPIXELS'):
  hltOutputALCA_cff.block_hltOutputALCALUMIPIXELS = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
if not hasattr(hltOutputALCA_cff,'block_hltOutputRPCMON'):
  hltOutputALCA_cff.block_hltOutputRPCMON = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
hltDebugWithAlCaOutputBlocks = (
  # the DQM, HLTDQM and HLTMON streams have the HLT debug outputs used online
  hltOutputMON_cff.block_hltOutputA.outputCommands,
  hltOutputMON_cff.block_hltOutputDQM.outputCommands,
  hltOutputMON_cff.block_hltOutputHLTMonitor.outputCommands,
  hltOutputMON_cff.block_hltOutputLookArea.outputCommands,
  hltOutputMON_cff.block_hltOutputReleaseValidation.outputCommands,
  # the ALCA streams have the AlCa outputs
  hltOutputALCA_cff.block_hltOutputALCAPHISYM.outputCommands,
  hltOutputALCA_cff.block_hltOutputALCAP0.outputCommands,
  hltOutputALCA_cff.block_hltOutputALCALUMIPIXELS.outputCommands,
  hltOutputALCA_cff.block_hltOutputRPCMON.outputCommands,
)
hltDebugWithAlCaOutputContent = buildPSet(hltDebugWithAlCaOutputBlocks)


# hltDefaultOutput
if not hasattr(hltOutputA_cff,'block_hltOutputA'):
  hltOutputA_cff.block_hltOutputA = hltOutputA_cff.block_hltOutputPhysicsEGammaCommissioning
hltDefaultOutputBlocks = (
  # the A stream has the HLT default output, with FEDs - strip out the FEDRawDataCollection keep statements for hltDefaultOutput
  hltOutputA_cff.block_hltOutputA.outputCommands,
)
hltDefaultOutputContent         = buildPSetWithoutRAWs(hltDefaultOutputBlocks)
hltDefaultOutputWithFEDsContent = buildPSet(hltDefaultOutputBlocks)


# define the CMSSW default event content configurations

# RAW event content
HLTriggerRAW = cms.PSet(
    outputCommands = cms.vstring()
)
HLTriggerRAW.outputCommands.extend(hltDefaultOutputWithFEDsContent.outputCommands)

# RECO event content
HLTriggerRECO = cms.PSet(
    outputCommands = cms.vstring()
)
HLTriggerRECO.outputCommands.extend(hltDefaultOutputContent.outputCommands)

# AOD event content
HLTriggerAOD = cms.PSet(
    outputCommands = cms.vstring()
)
HLTriggerAOD.outputCommands.extend(hltDefaultOutputContent.outputCommands)
dropL1GlobalTriggerObjectMapRecord(HLTriggerAOD)

# HLTDEBUG RAW event content
HLTDebugRAW = cms.PSet(
    outputCommands = cms.vstring()
)
HLTDebugRAW.outputCommands.extend(hltDebugWithAlCaOutputContent.outputCommands)

# HLTDEBUG FEVT event content
HLTDebugFEVT = cms.PSet(
    outputCommands = cms.vstring()
)
HLTDebugFEVT.outputCommands.extend(hltDebugWithAlCaOutputContent.outputCommands)


# dump the expanded event content configurations to a python configuration fragment
dump = open('HLTrigger_EventContent_cff.py', 'w')
dump.write('''import FWCore.ParameterSet.Config as cms

# EventContent for HLT related products.

# This file exports the following five EventContent blocks:
#   HLTriggerRAW  HLTriggerRECO  HLTriggerAOD (without DEBUG products)
#   HLTDebugRAW   HLTDebugFEVT                (with    DEBUG products)
#
# as these are used in Configuration/EventContent
#
''')
dump.write('HLTriggerRAW  = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTriggerRAW.outputCommands))
dump.write('HLTriggerRECO = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTriggerRECO.outputCommands))
dump.write('HLTriggerAOD  = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTriggerAOD.outputCommands))
dump.write('HLTDebugRAW   = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTDebugRAW.outputCommands))
dump.write('HLTDebugFEVT  = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTDebugFEVT.outputCommands))
dump.close()
