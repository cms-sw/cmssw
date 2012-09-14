#!/usr/bin/env python
"""
_DataScouting_

Scenario supporting proton collisions data scouting
Inheriting to reco.
Really against OO principles, but pragmatism should prevale, I guess.
"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco

from Configuration.DataProcessing.Utils import stepALCAPRODUCER,addMonitoring,dictIO,dqmIOSource,harvestingMode,dqmSeq
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class DataScouting(Reco):
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
        options.scenario = self.cbSc
        options.__dict__.update(defaultOptions.__dict__)
        options.step = 'DQM:DQM/DataScouting/dataScouting_cff.dataScoutingDQMSequence,ENDJOB'
        dictIO(options,args)        
        options.conditions = globalTag
                
        process = cms.Process('DataScouting')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()
        
        return process        

    def __getEmptyProcess(self):
      return cms.Process('Empty')
      

    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        In this scheme this method does not make any sense, but I have to 
        override the Reco one.

        """       
        return self.__getEmptyProcess()


    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        Same as above

        """
        return self.__getEmptyProcess()

    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        Proton collisions data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = self.cbSc
        options.step = "HARVESTING"+dqmSeq(args,':DQMOffline')
        options.name = "EDMtoMEConvert"
        options.conditions = globalTag
 
        process = cms.Process("HARVESTING")
        process.source = dqmIOSource(args)
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        harvestingMode(process,datasetName,args,rANDl=False)
        return process


    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Again the same thing.

        """
        return self.__getEmptyProcess()
        