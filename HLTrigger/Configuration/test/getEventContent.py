#!/usr/bin/env python3
"""getEventContent.py: print EventContent cff fragment of a ConfDB configuration
"""
import argparse
import subprocess
import os
import re

import FWCore.ParameterSet.Config as cms
import HLTrigger.Configuration.Tools.pipe as pipe
import HLTrigger.Configuration.Tools.options as options

def getHLTProcessBlocks(config, blocks):
  """return cms.Process containing the OutputModules of the HLT configuration
  """
  # cmd-line args to select HLT configuration
  if config.menu.run:
    configline = f'--runNumber {config.menu.run}'
  else:
    configline = f'--{config.menu.database} --{config.menu.version} --configName {config.menu.name}'

  # cmd to download HLT configuration
  cmdline = f'hltConfigFromDB {configline}'
  if config.proxy:
    cmdline += f' --dbproxy --dbproxyhost {config.proxy_host} --dbproxyport {config.proxy_port}'

  cmdline += ' --noedsources --noes --nopsets --noservices --nopaths --format python'
  cmdline += ' --blocks '+','.join({foo+'::outputCommands' for foo in blocks})

  # load HLT configuration
  try:
    foo = {}
    exec(pipe.pipe(cmdline).decode(), foo)
  except:
    raise Exception(f'query did not return a valid python file:\n query="{cmdline}"')

  ret = {}
  for block in blocks:
    key = 'block_'+block
    ret[key] = foo[key] if key in foo else None
    if ret[key] != None and not isinstance(ret[key], cms.PSet):
      raise Exception(f'query did not return valid HLT blocks:\n query="{cmdline}"')

  return ret

def getHLTProcessBlockGroups(config, blockGroupDict):
  ret = {}
  blockDict = getHLTProcessBlocks(config, {bar for foo in blockGroupDict.values() for bar in foo})
  for groupName in blockGroupDict:
    ret[groupName] = cms.PSet()
    for blockKey in blockGroupDict[groupName]:
      blockName = 'block_'+blockKey
      if blockDict[blockName] != None:
        setattr(ret[groupName], blockName, blockDict[blockName])
  return ret

def makePSet(statements):
  statements = sorted(statements)
  block = cms.PSet(
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*', )
  )
  block.outputCommands.extend( statements )
  return block

def makePSetNoDrop(statements):
  statements = sorted(statements)
  block = cms.PSet(
    outputCommands = cms.untracked.vstring()
  )
  block.outputCommands.extend( statements )
  return block

def buildPSet(blocks):
  statements = set()
  for block in blocks:
    statements.update( statement for statement in block if statement.find('drop') != 0 )
  return makePSet(statements)

def buildPSetNoDrop(blocks):
  statements = set()
  for block in blocks:
    statements.update( statement for statement in block if statement.find('drop') != 0 )
  return makePSetNoDrop(statements)

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

def printHLTriggerEventContentCff(process):

  blockGroups = getHLTProcessBlockGroups(config, {
    'hltOutputA_cff': [
      'hltOutputA',
      'hltOutputPhysicsCommissioning',
    ],
    'hltOutputALCA_cff': [
      'hltOutputALCAPHISYM',
      'hltOutputALCAP0',
      'hltOutputALCAPPSExpress',
      'hltOutputALCAPPSPrompt',
      'hltOutputALCALumiPixelsCountsExpress',
      'hltOutputALCALumiPixelsCountsPrompt',
      'hltOutputRPCMON',
    ],
    'hltOutputMON_cff': [
      'hltOutputA',
      'hltOutputPhysicsCommissioning',
      'hltOutputDQM',
      'hltOutputHLTMonitor',
      'hltOutputReleaseValidation',
    ],
    'hltOutputScouting_cff': [
      'hltOutputScoutingPF',
    ],
  })

  hltOutputA_cff = blockGroups['hltOutputA_cff']
  hltOutputALCA_cff = blockGroups['hltOutputALCA_cff']
  hltOutputMON_cff = blockGroups['hltOutputMON_cff']
  hltOutputScouting_cff = blockGroups['hltOutputScouting_cff']

  # hltDebugOutput

  if not hasattr(hltOutputMON_cff,'block_hltOutputA'):
    hltOutputMON_cff.block_hltOutputA = hltOutputMON_cff.block_hltOutputPhysicsCommissioning
  if not hasattr(hltOutputMON_cff,'block_hltOutputDQM'):
    hltOutputMON_cff.block_hltOutputDQM = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputMON_cff,'block_hltOutputHLTMonitor'):
    hltOutputMON_cff.block_hltOutputHLTMonitor = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*' ))
  if not hasattr(hltOutputMON_cff,'block_hltOutputReleaseValidation'):
    hltOutputMON_cff.block_hltOutputReleaseValidation = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))

  hltDebugOutputBlocks = (
    # the DQM and HLTMON streams have the HLT debug outputs used online
    hltOutputMON_cff.block_hltOutputA.outputCommands,
    hltOutputMON_cff.block_hltOutputDQM.outputCommands,
    hltOutputMON_cff.block_hltOutputHLTMonitor.outputCommands,
    hltOutputMON_cff.block_hltOutputReleaseValidation.outputCommands,
  )
  hltDebugOutputContent = buildPSet(hltDebugOutputBlocks)

  # hltDebugWithAlCaOutput

  if not hasattr(hltOutputALCA_cff,'block_hltOutputALCAPHISYM'):
    hltOutputALCA_cff.block_hltOutputALCAPHISYM = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputALCA_cff,'block_hltOutputALCAP0'):
    hltOutputALCA_cff.block_hltOutputALCAP0 = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputALCA_cff,'block_hltOutputALCAPPSExpress'):
    hltOutputALCA_cff.block_hltOutputALCAPPSExpress = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputALCA_cff,'block_hltOutputALCAPPSPrompt'):
    hltOutputALCA_cff.block_hltOutputALCAPPSPrompt = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputALCA_cff,'block_hltOutputALCALumiPixelsCountsExpress'):
    hltOutputALCA_cff.block_hltOutputALCALumiPixelsCountsExpress = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputALCA_cff,'block_hltOutputALCALumiPixelsCountsPrompt'):
    hltOutputALCA_cff.block_hltOutputALCALumiPixelsCountsPrompt = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  if not hasattr(hltOutputALCA_cff,'block_hltOutputRPCMON'):
    hltOutputALCA_cff.block_hltOutputRPCMON = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))
  hltDebugWithAlCaOutputBlocks = (
    # the DQM and HLTMON streams have the HLT debug outputs used online
    hltOutputMON_cff.block_hltOutputA.outputCommands,
    hltOutputMON_cff.block_hltOutputDQM.outputCommands,
    hltOutputMON_cff.block_hltOutputHLTMonitor.outputCommands,
    hltOutputMON_cff.block_hltOutputReleaseValidation.outputCommands,
    # the ALCA streams have the AlCa outputs
    hltOutputALCA_cff.block_hltOutputALCAPHISYM.outputCommands,
    hltOutputALCA_cff.block_hltOutputALCAP0.outputCommands,
    hltOutputALCA_cff.block_hltOutputALCAPPSExpress.outputCommands,
    hltOutputALCA_cff.block_hltOutputALCAPPSPrompt.outputCommands,
    hltOutputALCA_cff.block_hltOutputALCALumiPixelsCountsExpress.outputCommands,
    hltOutputALCA_cff.block_hltOutputALCALumiPixelsCountsPrompt.outputCommands,
    hltOutputALCA_cff.block_hltOutputRPCMON.outputCommands,
  )
  hltDebugWithAlCaOutputContent = buildPSet(hltDebugWithAlCaOutputBlocks)

  # hltScoutingOutput

  if not hasattr(hltOutputScouting_cff,'block_hltOutputScoutingPF'):
    hltOutputScouting_cff.block_hltOutputScoutingPF = cms.PSet(outputCommands = cms.untracked.vstring( 'drop *' ))

  hltScoutingOutputBlocks = (
    # the Scouting streams have the Scouting outputs
    hltOutputScouting_cff.block_hltOutputScoutingPF.outputCommands,
  )
  hltScoutingOutputContent = buildPSetNoDrop(hltScoutingOutputBlocks)

  # hltDefaultOutput
  if not hasattr(hltOutputA_cff,'block_hltOutputA'):
    hltOutputA_cff.block_hltOutputA = hltOutputA_cff.block_hltOutputPhysicsCommissioning
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
  HLTriggerRAW.outputCommands.extend(hltScoutingOutputContent.outputCommands)

  # RECO event content
  HLTriggerRECO = cms.PSet(
      outputCommands = cms.vstring()
  )
  HLTriggerRECO.outputCommands.extend(hltDefaultOutputContent.outputCommands)
  HLTriggerRECO.outputCommands.extend(hltScoutingOutputContent.outputCommands)

  # AOD event content
  HLTriggerAOD = cms.PSet(
      outputCommands = cms.vstring()
  )
  HLTriggerAOD.outputCommands.extend(hltDefaultOutputContent.outputCommands)
  HLTriggerAOD.outputCommands.extend(hltScoutingOutputContent.outputCommands)
  dropL1GlobalTriggerObjectMapRecord(HLTriggerAOD)

  # HLTDEBUG RAW event content
  HLTDebugRAW = cms.PSet(
      outputCommands = cms.vstring()
  )
  HLTDebugRAW.outputCommands.extend(hltDebugWithAlCaOutputContent.outputCommands)
  HLTDebugRAW.outputCommands.extend(hltScoutingOutputContent.outputCommands)

  # HLTDEBUG FEVT event content
  HLTDebugFEVT = cms.PSet(
      outputCommands = cms.vstring()
  )
  HLTDebugFEVT.outputCommands.extend(hltDebugWithAlCaOutputContent.outputCommands)
  HLTDebugFEVT.outputCommands.extend(hltScoutingOutputContent.outputCommands)

  # Scouting event content
  HLTScouting = cms.PSet(
      outputCommands = cms.vstring()
  )
  HLTScouting.outputCommands.extend(hltScoutingOutputContent.outputCommands)

  # print the expanded event content configurations to stdout
  print('''import FWCore.ParameterSet.Config as cms

# EventContent for HLT related products.

# This file exports the following EventContent blocks:
#   HLTriggerRAW  HLTriggerRECO  HLTriggerAOD (without DEBUG products)
#   HLTDebugRAW   HLTDebugFEVT                (with    DEBUG products)
#   HLTScouting                               (with Scouting products)
#
# as these are used in Configuration/EventContent
#''')
  print('HLTriggerRAW  = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTriggerRAW.outputCommands))
  print('HLTriggerRECO = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTriggerRECO.outputCommands))
  print('HLTriggerAOD  = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTriggerAOD.outputCommands))
  print('HLTDebugRAW   = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTDebugRAW.outputCommands))
  print('HLTDebugFEVT  = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTDebugFEVT.outputCommands))
  print('HLTScouting   = cms.PSet(\n    outputCommands = cms.vstring( *(\n%s\n    ) )\n)\n'  % ',\n'.join( '        \'%s\'' % keep for keep in HLTScouting.outputCommands))

###
### main
###
if __name__ == '__main__':

  # defaults of cmd-line arguments
  defaults = options.HLTProcessOptions()

  parser = argparse.ArgumentParser(
    prog = './'+os.path.basename(__file__),
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = __doc__
  )

  # required argument
  parser.add_argument('menu',
                      action  = 'store',
                      type    = options.ConnectionHLTMenu,
                      metavar = 'MENU',
                      help    = 'HLT menu to dump from the database. Supported formats are:\n  - /path/to/configuration[/Vn]\n  - [[{v1|v2|v3}/]{run3|run2|online|adg}:]/path/to/configuration[/Vn]\n  - run:runnumber\nThe possible converters are "v1", "v2, and "v3" (default).\nThe possible databases are "run3" (default, used for offline development), "run2" (used for accessing run2 offline development menus), "online" (used to extract online menus within Point 5) and "adg" (used to extract the online menus outside Point 5).\nIf no menu version is specified, the latest one is automatically used.\nIf "run:" is used instead, the HLT menu used for the given run number is looked up and used.\nNote other converters and databases exist as options but they are only for expert/special use.' )

  # options
  parser.add_argument('--dbproxy',
                      dest    = 'proxy',
                      action  = 'store_true',
                      default = defaults.proxy,
                      help    = 'Use a socks proxy to connect outside CERN network (default: False)' )
  parser.add_argument('--dbproxyport',
                      dest    = 'proxy_port',
                      action  = 'store',
                      metavar = 'PROXYPORT',
                      default = defaults.proxy_port,
                       help    = 'Port of the socks proxy (default: 8080)' )
  parser.add_argument('--dbproxyhost',
                      dest    = 'proxy_host',
                      action  = 'store',
                      metavar = 'PROXYHOST',
                      default = defaults.proxy_host,
                      help    = 'Host of the socks proxy (default: "localhost")' )

  # parse command line arguments and options
  config = parser.parse_args()

  printHLTriggerEventContentCff(config)
