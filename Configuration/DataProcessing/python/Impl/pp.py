#!/usr/bin/env python
"""
_pp_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Scenario import Scenario
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,addMonitoring
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class pp(Scenario):
    """
    _pp_

    Implement configuration building for data processing for proton
    collision data taking

    """


    def promptReco(self, globalTag, writeTiers = ['RECO'], **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """

        skims = ['TkAlMinBias',
                 'TkAlMuonIsolated',
                 'MuAlCalIsolatedMu',
                 'MuAlOverlaps',
                 'HcalCalIsoTrk',
                 'HcalCalDijets',
                 'SiStripCalMinBias',
                 'EcalCalElectron',
                 'DtCalib',
                 'TkAlJpsiMuMu',
                 'TkAlUpsilonMuMu',
                 'TkAlZMuMu']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = 'RAW2DIGI,L1Reco,RECO'+step+',DQM,ENDJOB'
        options.eventcontent = ','.join(writeTiers)
        options.datatier = ','.join(writeTiers)
        options.conditions = globalTag
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        #add the former top level patches here
        customisePrompt(process)
        addMonitoring(process)
        
        return process


    def expressProcessing(self, globalTag, writeTiers = [], **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """

        skims = ['SiStripCalZeroBias',
                 'TkAlMinBias',
                 'DtCalib',
                 'MuAlCalIsolatedMu',
                 'SiStripPCLHistos']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = 'RAW2DIGI,L1Reco,RECO'+step+',DQM,ENDJOB'
        options.eventcontent = ','.join(writeTiers)
        options.datatier = ','.join(writeTiers)
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("NewEventStreamFileReader",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        #add the former top level patches here
        customiseExpress(process)
        addMonitoring(process)
                
        return process


    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        AlcaReco processing & skims for proton collisions

        """

        step = ""
        if 'PromptCalibProd' in skims:
            step = "ALCA:PromptCalibProd" 
            skims.remove('PromptCalibProd')
        
        if len( skims ) > 0:
            if step != "":
                step += ","
            step += "ALCAOUTPUT:"+('+'.join(skims))
                
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = step
        options.conditions = args['globaltag'] if 'globaltag' in args else 'None'
        options.triggerResultsProcess = 'RECO'
        
        process = cms.Process('ALCA')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source(
           "PoolSource",
           fileNames = cms.untracked.vstring()
        )

        cb.prepare() 

        # FIXME: dirty hack..any way around this?
        # Tier0 needs the dataset used for ALCAHARVEST step to be a different data-tier
        if 'PromptCalibProd' in step:
            process.ALCARECOStreamPromptCalibProd.dataset.dataTier = cms.untracked.string('ALCAPROMPT')

        return process


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        Proton collisions data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "HARVESTING:dqmHarvesting"
        options.name = "EDMtoMEConvert"
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
 
        process = cms.Process("HARVESTING")
        if args.get('newDQMIO', False):
            process.source = cms.Source("DQMRootSource")
        else:
            process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        process.dqmSaver.saveByLumiSection = 1
        if args.has_key('referenceFile') and args.get('referenceFile', ''):
            process.DQMStore.referenceFileName = \
                                cms.untracked.string(args['referenceFile'])

        return process


    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "ALCAHARVEST:BeamSpotByRun+BeamSpotByLumi+SiStripQuality"
        options.name = "ALCAHARVEST"
        options.conditions = globalTag
 
        process = cms.Process("ALCAHARVEST")
        process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        
        return process

