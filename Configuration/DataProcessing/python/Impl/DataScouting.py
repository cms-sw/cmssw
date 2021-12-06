#!/usr/bin/env python3
"""
_DataScouting_

Scenario supporting proton collisions data scouting
Inheriting to reco.
Really against OO principles, but pragmatism should prevale, I guess.
"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,addMonitoring,dictIO,dqmIOSource,harvestingMode,dqmSeq,gtNameAndConnect
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class DataScouting(Scenario):
    def __init__(self):
        Scenario.__init__(self)
    """
    _DataScouting_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Collision data, data scouting (dst stream).
        This method provides the scheleton process for the dataScouting.
        dpiparo 17-7-2012
        I follow the structure of the package.
        """
        options = Options()
        options.scenario = 'pp'
        options.__dict__.update(defaultOptions.__dict__)
        options.step = 'DQM:DQM/DataScouting/dataScouting_cff.dataScoutingDQMSequence,ENDJOB'
        dictIO(options,args)        
        options.conditions = gtNameAndConnect(globalTag, args)
                
        process = cms.Process('DataScouting', self.eras)
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()
        
        return process        

    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        Proton collisions data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = 'pp'
        options.step = "HARVESTING"+dqmSeq(args,':DQMOffline')
        options.name = "EDMtoMEConvert"
        options.conditions = gtNameAndConnect(globalTag, args)
 
        process = cms.Process("HARVESTING", self.eras)
        process.source = dqmIOSource(args)
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        harvestingMode(process,datasetName,args,rANDl=False)
        return process
