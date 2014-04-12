#!/usr/bin/env python
"""
_preprodmc_

Scenario supporting pre-production

"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
import FWCore.ParameterSet.Config as cms


class preprodmc(Scenario):
    """
    _preprodmc_

    Implement configuration building for RelVal MC production 

    """

    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        DQM Harvesting for pre-production

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "HARVESTING:validationpreprodHarvesting+dqmHarvestingPOG"
        options.isMC = True
        options.isData = False
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
 
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
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        if args.has_key('referenceFile') and args.get('referenceFile', ''):
            process.DQMStore.referenceFileName = \
                                cms.untracked.string(args['referenceFile'])
        
        return process
