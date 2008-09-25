#!/usr/bin/env python

"""
_HarvestingConfig_

Module containing utilities for generating DQM Harvesting histograms.

Insert this in the CMSSW release somewhere general like
Configuration/GlobalRuns/python

The WM Tools will use this at runtime to get a process object
for a harvesting job.
This module acts as a standard API to the WM Tools, the
makeDQMHarvestingConfig is a public interface and should not
be changed without talking to DMWM, anything under the hood can
be changed.

DQM Harvesting jobs will know the following information:

- dataset Name: (aka workflow in harvesting jobs) This is the dataset name
 being harvested, and can be used to trigger different configurations for
 primary datasets and data tiers etc.

- run Number: The run number being harvested.

- global Tag: The conditions tag used to process the run, as recorded
  in DBS or the Tier 0

- input files: A list of LFNs for the (dataset, run) being harvested that
  will need to be added to the source.

- options: Future proofing agains API changes, not presently used
           Note that adding new options needs to be communicated to
           DMWM to insert the appropriate information

"""
import os
import FWCore.ParameterSet.Config as cms

def makeAlcaHarvestingConfig(datasetName, runNumber,
                             globalTag, *inputFiles, **options):
    """
    _makeAlcaHervestingConfig_

    Create an ALCA Harvesting configuration

    """

    process = cms.Process("EDMtoMEConvert")

    process.load("DQMServices.Components.EDMtoMEConverter_cff")

    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"

    #  //
    # // Set the GlobalTag
    #//
    process.GlobalTag.globaltag = globalTag
    process.prefer("GlobalTag")

    process.load("Configuration.StandardSequences.Geometry_cff")

    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

    process.options = cms.untracked.PSet(
        fileMode = cms.untracked.string('FULLMERGE')
        )

    process.source = cms.Source(
        "PoolSource",
        processingMode = cms.untracked.string("RunsLumisAndEvents"),
        fileNames = cms.untracked.vstring()
        )

    #  //
    # // Add input files
    #//
    for fileName in inputFiles:
        process.source.fileNames.append(fileName)

    process.maxEvents.input = -1

    process.source.processingMode = "RunsAndLumis"
    #  //
    # // Add configuration details for production system and
    #//  bookkeeping
    process.configurationMetadata = cms.untracked(cms.PSet())
    process.configurationMetadata.name = cms.untracked(
        cms.string("Configuration/GlobalRuns/python/HarvestingConfig.py"))
    process.configurationMetadata.version = cms.untracked(
        cms.string(os.environ['CMSSW_VERSION']))
    process.configurationMetadata.annotation = cms.untracked(
        cms.string("DQM Alca Harvesting Configuration"))

    process.DQMStore.referenceFileName = ''
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = datasetName

    # Merge different runs in one output ROOT file
    #process.DQMStore.collateHistograms = True
    #process.EDMtoMEConverter.convertOnEndLumi = False
    #process.EDMtoMEConverter.convertOnEndRun = True
    #process.dqmSaver.saveByRun = -1
    #process.dqmSaver.saveAtJobEnd = True

    # Produce one ROOT file per run
    process.DQMStore.collateHistograms = False
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = False

    process.p1 = cms.Path(
        process.EDMtoMEConverter*process.dqmSaver)

    msg = "ALCA Harvesting Not tested Yet"
    raise NotImplementedError, msg

    return process

def makeStandardHarvestingConfig(datasetName, runNumber,
                                 globalTag, *inputFiles, **options):
    """
    _makeStandardHarvestingConfig_

    make a harvesting configuration for the standard/default
    harvesting scenario

    """

    process = cms.Process("EDMtoMEConvert")

    process.load("DQMServices.Components.EDMtoMEConverter_cff")

    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"

    #  //
    # // Set the GlobalTag
    #//
    process.GlobalTag.globaltag = globalTag
    process.prefer("GlobalTag")

    process.load("Configuration.StandardSequences.Geometry_cff")
    process.load("DQMOffline.Configuration.DQMOffline_SecondStep_cff")

    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

    process.options = cms.untracked.PSet(
        fileMode = cms.untracked.string('FULLMERGE')
        )

    process.source = cms.Source(
        "PoolSource",
        processingMode = cms.untracked.string("RunsLumisAndEvents"),
        fileNames = cms.untracked.vstring()
        )

    #  //
    # // Add input files
    #//
    for fileName in inputFiles:
        process.source.fileNames.append(fileName)

    process.maxEvents.input = -1

    process.source.processingMode = "RunsAndLumis"
    #  //
    # // Add configuration details for production system and
    #//  bookkeeping
    process.configurationMetadata = cms.untracked(cms.PSet())
    process.configurationMetadata.name = cms.untracked(
        cms.string("Configuration/GlobalRuns/python/HarvestingConfig.py"))
    process.configurationMetadata.version = cms.untracked(
        cms.string(os.environ['CMSSW_VERSION']))
    process.configurationMetadata.annotation = cms.untracked(
        cms.string("DQM Standard Harvesting Configuration"))

    process.DQMStore.referenceFileName = ''
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = datasetName

    process.DQMStore.collateHistograms = False
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = False

    process.p1 = cms.Path(
        process.EDMtoMEConverter*process.DQMOffline_SecondStep*process.dqmSaver)

    return process

def makeCosmicHarvestingConfig(datasetName, runNumber,
                               globalTag, *inputFiles, **options):
    """
    _makeCosmicHarvestingConfig_

    Produce a cosmic run harvesting configuration

    """

    process = cms.Process("EDMtoMEConvert")

    process.load("DQMServices.Components.EDMtoMEConverter_cff")

    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"

    #  //
    # // Set the GlobalTag
    #//
    process.GlobalTag.globaltag = globalTag
    process.prefer("GlobalTag")

    process.load("Configuration.StandardSequences.Geometry_cff")
    process.load("DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff")

    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

    process.options = cms.untracked.PSet(
        fileMode = cms.untracked.string('FULLMERGE')
        )

    process.source = cms.Source(
        "PoolSource",
        processingMode = cms.untracked.string("RunsLumisAndEvents"),
        fileNames = cms.untracked.vstring()
        )

    #  //
    # // Add input files
    #//
    for fileName in inputFiles:
        process.source.fileNames.append(fileName)

    process.maxEvents.input = -1

    process.source.processingMode = "RunsAndLumis"
    #  //
    # // Add configuration details for production system and
    #//  bookkeeping
    process.configurationMetadata = cms.untracked(cms.PSet())
    process.configurationMetadata.name = cms.untracked(
        cms.string("Configuration/GlobalRuns/python/HarvestingConfig.py"))
    process.configurationMetadata.version = cms.untracked(
        cms.string(os.environ['CMSSW_VERSION']))
    process.configurationMetadata.annotation = cms.untracked(
        cms.string("DQM Cosmic Harvesting Configuration"))

    process.DQMStore.referenceFileName = ''
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = datasetName

    process.DQMStore.collateHistograms = False
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = False

    process.p1 = cms.Path(
        process.EDMtoMEConverter*process.DQMOfflineCosmics_SecondStep*process.dqmSaver)

    return process

def makeDQMHarvestingConfig(datasetName, runNumber, globalTag, *inputFiles,
                            **options):
    """
    _makeDQMHarvestingConfig_

    API method to create a harvesting configuration

    """
    #  //
    # // Use the dataset name to distribute to various configurations
    #//
    if datasetName.endswith("ALCARECO"):
        #  //
        # // Alca harvesting
        #//
        return makeAlcaHarvestingConfig(datasetName, runNumber,
                                        globalTag, *inputFiles, **options)


    if datasetName.startswith("/Cosmic"):
        #  //
        # // Cosmic harvesting
        #//
        return makeCosmicHarvestingConfig(datasetName, runNumber,
                                          globalTag, *inputFiles, **options)

    #  //
    # // Safe default needs to be added here
    #//
    return makeStandardHarvestingConfig(datasetName, runNumber,
                                        globalTag, *inputFiles, **options)

