#!/usr/bin/env python3
"""
_Merge_

Module that generates standard merge job configurations for use in any
standard processing

"""


from FWCore.ParameterSet.Config import Process, EndPath
from FWCore.ParameterSet.Modules import OutputModule, Source, Service
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
    - mergeNANO : to merge NanoAOD
    - bypassVersionCheck : to bypass version check in case merging happened in lower version of CMSSW (i.e. UL HLT case). This will be FALSE by default.

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
    bypassVersionCheck = options.get("bypassVersionCheck", False)
    isL1Scouting = options.get("isL1Scouting", False)
    #  //
    # // build process
    #//
    process = Process(processName)

    #  //
    # // input source
    #//
    if newDQMIO:
        process.source = Source("DQMRootSource", reScope = CfgTypes.untracked.string(""))
        process.add_(Service("DQMStore"))
    else:
        process.source = Source("PoolSource")
        if bypassVersionCheck:
            process.source.bypassVersionCheck = CfgTypes.untracked.bool(True)
        if dropDQM:
            process.source.inputCommands = CfgTypes.untracked.vstring('keep *','drop *_EDMtoMEConverter_*_*')
        if not mergeNANO:
            process.source.noRunLumiSort = CfgTypes.untracked.bool(True)
    process.source.fileNames = CfgTypes.untracked(CfgTypes.vstring())
    for entry in inputFiles:
        process.source.fileNames.append(str(entry))

    #  //
    # // output module
    #//
    if newDQMIO:
        outMod = OutputModule("DQMRootOutputModule")
    elif mergeNANO:
        import Configuration.EventContent.EventContent_cff
        if isL1Scouting:
            # For Run-3 L1-Scouting data, the plugin "OrbitNanoAODOutputModule"
            # is used in the "merge" step, instead of "NanoAODOutputModule".
            # "OrbitNanoAODOutputModule" converts orbit-based NanoAOD tables (EDM)
            # to event/BX-based "flat" NanoAOD branches.
            outMod = OutputModule("OrbitNanoAODOutputModule",
                Configuration.EventContent.EventContent_cff.L1SCOUTNANOAODEventContent.clone(),
                # skip BXs in which all the L1-Scouting tables are empty
                skipEmptyBXs = CfgTypes.bool(True)
            )
        else:
            outMod = OutputModule("NanoAODOutputModule", Configuration.EventContent.EventContent_cff.NANOAODEventContent.clone())
    else:
        outMod = OutputModule("PoolOutputModule")
        outMod.mergeJob = CfgTypes.untracked.bool(True)
        outMod.eventAuxiliaryBasketSize = CfgTypes.untracked.int32(2*1024*1024)

    outMod.fileName = CfgTypes.untracked.string(outputFilename)
    if outputLFN != None:
        outMod.logicalFileName = CfgTypes.untracked.string(outputLFN)
    setattr(process, outputModLabel, outMod)

    process.outputPath = EndPath(outMod)

    return process
