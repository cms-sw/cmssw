#!/usr/bin/env python3
"""
_Repack_

Module that generates standard repack configurations

"""
import copy
import FWCore.ParameterSet.Config as cms


def repackProcess(**args):
    """
    _repackProcess_

    Creates and returns a repack process

    supported options:

    - outputs      : defines output modules

    """
    from Configuration.EventContent.EventContent_cff import RAWEventContent
    process = cms.Process("REPACK")
    process.load("FWCore.MessageLogger.MessageLogger_cfi")

    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

    process.configurationMetadata = cms.untracked.PSet(
        name = cms.untracked.string("repack-config"),
        version = cms.untracked.string("none"),
        annotation = cms.untracked.string("auto generated configuration")
        )

    process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring("ProductNotFound","TooManyProducts","TooFewProducts"),
        wantSummary = cms.untracked.bool(False)
        )

    process.source = cms.Source(
        "NewEventStreamFileReader",
        fileNames = cms.untracked.vstring()
        )

    outputs = args.get('outputs', [])

    if len(outputs) > 0:
        process.outputPath = cms.EndPath()

    for output in outputs:

        moduleLabel = output['moduleLabel']
        selectEvents = output.get('selectEvents', None)
        maxSize = output.get('maxSize', None)

        outputModule = cms.OutputModule(
            "PoolOutputModule",
            compressionAlgorithm=copy.copy(RAWEventContent.compressionAlgorithm),
            compressionLevel=copy.copy(RAWEventContent.compressionLevel),
            fileName = cms.untracked.string("%s.root" % moduleLabel)
            )

        outputModule.dataset = cms.untracked.PSet(dataTier = cms.untracked.string("RAW"))

        if maxSize != None:
            outputModule.maxSize = cms.untracked.int32(maxSize)

        if selectEvents != None:
            outputModule.SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring(selectEvents)
                )

        setattr(process, moduleLabel, outputModule)

        process.outputPath += outputModule

    return process