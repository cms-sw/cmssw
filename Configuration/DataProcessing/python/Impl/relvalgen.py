#!/usr/bin/env python3
"""
_relvalgen_

Scenario supporting RelVal GEN production

"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
import FWCore.ParameterSet.Config as cms

class relvalgen(Scenario):
    def __init__(self):
        Scenario.__init__(self)
    """
    _relvalgen_

    Implement configuration building for RelVal GEN production 

    """


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        DQM Harvesting for RelVal GEN production

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "HARVESTING:genHarvesting"
        options.isMC = True
        options.isData = False
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
        options.harvesting = "AtJobEnd"
 
        process = cms.Process("HARVESTING", self.eras)
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
