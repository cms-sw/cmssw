#!/usr/bin/env python
"""
_Merge_

Module that generates standard merge job configurations for use in any
standard processing

"""


from FWCore.ParameterSet.Config import Process, EndPath
from FWCore.ParameterSet.Modules import OutputModule, Source, Service
from Configuration.EventContent.EventContent_cff import NANOAODEventContent
import FWCore.ParameterSet.Types as CfgTypes


def mergeProcess(*inputFiles, **options):
    """
    _mergeProcess_

    Creates and returns a merge process that will merge the provided
    filenames

    supported options:

    - process_name : name of the process, defaults to Merge
    - outputmod_label : label of the output module, defaults to Merged
    - newDQMIO : specifies if the new DQM format should be used to merge the files
    - output_file : sets the output file name
    - output_lfn : sets the output LFN

    """
    #  //
    # // process supported options
    #//
    processName = options.get("process_name", "Merge")
    outputModLabel = options.get("outputmod_label", "Merged")
    outputFilename = options.get("output_file", "Merged.root")
    outputLFN = options.get("output_lfn", None)
    dropDQM = options.get("drop_dqm", False)
    newDQMIO = options.get("newDQMIO", False)
    mergeNANO = options.get("mergeNANO", False)
    #  //
    # // build process
    #//
    process = Process(processName)

    #  //
    # // input source
    #//
    if newDQMIO:
        process.source = Source("DQMRootSource")
        process.add_(Service("DQMStore"))
    else:
        process.source = Source("PoolSource")
        if dropDQM:
            process.source.inputCommands = CfgTypes.untracked.vstring('keep *','drop *_EDMtoMEConverter_*_*')
    process.source.fileNames = CfgTypes.untracked(CfgTypes.vstring())
    for entry in inputFiles:
        process.source.fileNames.append(str(entry))
 
    #  //
    # // output module
    #//
    if newDQMIO:
        outMod = OutputModule("DQMRootOutputModule")
    elif mergeNANO:
        outMod = OutputModule("NanoAODOutputModule",NANOAODEventContent.clone())
    else:
        outMod = OutputModule("PoolOutputModule")

    outMod.fileName = CfgTypes.untracked.string(outputFilename)
    if outputLFN != None:
        outMod.logicalFileName = CfgTypes.untracked.string(outputLFN)
    setattr(process, outputModLabel, outMod)

    process.outputPath = EndPath(outMod)

    return process
