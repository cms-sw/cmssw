#!/usr/bin/env python3
"""
_Repack_

Module that generates standard repack configurations

"""
import copy
import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
import Configuration.Skimming.RAWSkims_cff as RawSkims
from Configuration.AlCa.GlobalTag import GlobalTag

def repackProcess(**args):
    """
    _repackProcess_

    Creates and returns a repack process

    supported options:

    - outputs      : defines output modules

    """
    from Configuration.EventContent.EventContent_cff import RAWEventContent
    from Configuration.EventContent.EventContent_cff import HLTSCOUTEventContent
    from Configuration.EventContent.EventContent_cff import L1SCOUTEventContent
    process = cms.Process("REPACK")
    process.load("FWCore.MessageLogger.MessageLogger_cfi")
    
    process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))

    process.configurationMetadata = cms.untracked.PSet(
        name=cms.untracked.string("repack-config"),
        version=cms.untracked.string("none"),
        annotation=cms.untracked.string("auto generated configuration")
    )

    process.options = cms.untracked.PSet(
        Rethrow=cms.untracked.vstring("ProductNotFound", "TooManyProducts", "TooFewProducts"),
        wantSummary=cms.untracked.bool(False)
    )

    process.source = cms.Source(
        "NewEventStreamFileReader",
        fileNames=cms.untracked.vstring()
    )

    defaultDataTier = "RAW"

    # Should we default to something if dataTier arg isn't provided?
    dataTier = args.get('dataTier', defaultDataTier)
    eventContent = RAWEventContent
    if dataTier == "HLTSCOUT":
        eventContent = HLTSCOUTEventContent
    elif dataTier == "L1SCOUT":
        eventContent = L1SCOUTEventContent

    outputs = args.get('outputs', [])

    if len(outputs) > 0:
        process.outputPath = cms.EndPath()
        
    globalTag = args.get('globalTag', None)   
    if globalTag:
        process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
        process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')
    
    for output in outputs:

        selectEventsBase = output.get('selectEvents', None)
        rawSkim = output.get('rawSkim', None)

        if rawSkim:

            selectEventsBase = [item.replace(":HLT", "") for item in selectEventsBase]

            process.baseSelection = hlt.hltHighLevel.clone(
                TriggerResultsTag = "TriggerResults::HLT",
                HLTPaths = cms.vstring(selectEventsBase)
            )
            skim = getattr(RawSkims, rawSkim)
            setattr(process, rawSkim, skim)
            path = cms.Path(skim + process.baseSelection)
            selectEvents = f"{rawSkim}Path"
            setattr(process, selectEvents, path)

        else:
            selectEvents = selectEventsBase

        moduleLabel = output['moduleLabel']
        maxSize = output.get('maxSize', None)

        outputModule = cms.OutputModule(
            "PoolOutputModule",
            compressionAlgorithm=copy.copy(eventContent.compressionAlgorithm),
            compressionLevel=copy.copy(eventContent.compressionLevel),
            fileName=cms.untracked.string("%s.root" % moduleLabel)
        )

        outputModule.dataset = cms.untracked.PSet(dataTier=cms.untracked.string(dataTier))

        if maxSize is not None:
            outputModule.maxSize = cms.untracked.int32(maxSize)

        if selectEvents is not None:
            outputModule.SelectEvents = cms.untracked.PSet(
                SelectEvents=cms.vstring(selectEvents)
            )

        setattr(process, moduleLabel, outputModule)

        process.outputPath += outputModule

    return process
