import FWCore.ParameterSet.Config as cms

def customiseGenPUProtonsEventContent(process):

    commonOutputCommands = cms.untracked.vstring('keep *_genPUProtons*_*_*', 'keep *_*_genPUProtons*_*')
    process.RAWSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWSIMHLTEventContent.outputCommands.extend( commonOutputCommands)
    process.GENRAWEventContent.outputCommands.extend( commonOutputCommands)
    process.PREMIXEventContent.outputCommands.extend( commonOutputCommands)
    process.PREMIXRAWEventContent.outputCommands.extend( commonOutputCommands)
    process.REPACKRAWSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.RECOSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.AODSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWRECOSIMHLTEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWRECODEBUGHLTEventContent.outputCommands.extend( commonOutputCommands)
    process.FEVTSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWDEBUGEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWDEBUGHLTEventContent.outputCommands.extend( commonOutputCommands)
    process.FEVTDEBUGEventContent.outputCommands.extend( commonOutputCommands)
    process.FEVTDEBUGHLTEventContent.outputCommands.extend( commonOutputCommands)
    process.REPACKRAWSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.MINIAODSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWMINIAODSIMEventContent.outputCommands.extend( commonOutputCommands)
    process.RAWAODSIMEventContent.outputCommands.extend( commonOutputCommands)

    return process
