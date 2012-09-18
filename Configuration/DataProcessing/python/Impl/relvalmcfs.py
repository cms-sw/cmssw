#!/usr/bin/env python
"""
_relvalmcfs_

Scenario supporting RelVal MC FastSim production

"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
import FWCore.ParameterSet.Config as cms


class relvalmcfs(Scenario):
    """
    _relvalmcfs_

    Implement configuration building for RelVal MC FastSim production 

    """


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        DQM Harvesting for RelVal MC production

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "HARVESTING:validationHarvestingFS"
        options.isMC = True
        options.isData = False
        options.beamspot = None
        options.name = "EDMtoMEConvert"
        options.conditions = globalTag
 
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
        if args.has_key('referenceFile') and args.get('referenceFile', ''):
            process.DQMStore.referenceFileName = \
                                cms.untracked.string(args['referenceFile'])
        
        return process
