#!/usr/bin/env python
"""
_pp_

Scenario supporting pre-production

"""

import os
import sys

from Configuration.DataProcessing.Scenario import Scenario
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
    


class preprodmc(Scenario):
    """
    _preprodmc_

    Implement configuration building for RelVal MC production 

    """


    def promptReco(self, globalTag, writeTiers = ['RECO']):
        """
        _promptReco_

        Prompt reco for pre-production

        """
        
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = 'RAW2DIGI,RECO,VALIDATION:validation_preprod,DQM:DQMOfflinePOG'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag

        
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        for tier in writeTiers: 
          addOutputModule(process, tier, process.RECOEventContent)        
 
        return process

    def expressProcessing(self, globalTag,  writeTiers = [],
                          datasets = [], alcaDataset = None):
        """
        _expressProcessing_

        Express processing for pre-production

        """

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = \
          """RAW2DIGI,RECO,ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics,ENDJOB"""
        options.isMC = False
        options.isData = True
        options.eventcontent = None
        options.relval = None
        options.beamspot = None
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        
        process = cms.Process('EXPRESS')
        cb = ConfigBuilder(options, process = process)

        process.source = cms.Source(
           "NewEventStreamFileReader",
           fileNames = cms.untracked.vstring()
        )
        
        cb.prepare()

        #  //
        # // Install the OutputModules for everything but ALCA
        #//
        self.addExpressOutputModules(process, writeTiers, datasets)
        
        #  //
        # // TODO: Install Alca output
        #//
        
        return process
    

    def alcaReco(self, *skims):
        """
        _alcaReco_

        AlcaReco processing & skims for pre-production

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = 'ALCA:MuAlStandAloneCosmics+DQM,ENDJOB'
        options.isMC = False
        options.isData = True
        options.conditions = "FrontierConditions_GlobalTag,CRAFT_V16P::All" 
        options.beamspot = None
        options.eventcontent = None
        options.relval = None
        
        process = cms.Process('ALCA')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source(
           "PoolSource",
           fileNames = cms.untracked.vstring()
        )

        cb.prepare() 

        #  //
        # // Verify and Edit the list of skims to be written out
        #//  by this job
        availableStreams = process.outputModules_().keys()

        #  //
        # // First up: Verify skims are available by output module name
        #//
        for skim in skims:
            if skim not in availableStreams:
                msg = "Skim named: %s not available " % skim
                msg += "in Alca Reco Config:\n"
                msg += "Known Skims: %s\n" % availableStreams
                raise RuntimeError, msg

        #  //
        # // Prune any undesired skims
        #//
        for availSkim in availableStreams:
            if availSkim not in skims:
                self.dropOutputModule(process, availSkim)

        return process
                

        

        


    def dqmHarvesting(self, datasetName, runNumber,  globalTag, **options):
        """
        _dqmHarvesting_

        DQM Harvesting for pre-production

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "HARVESTING:validationpreprodHarvesting+dqmHarvestingPOG"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
 
        process = cms.Process("HARVESTING")
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
