#!/usr/bin/env python
"""
_Scenario_

Standard cmsRun Process building interface used for data processing
for a particular data scenario.
A scenario is a macro-data-taking setting such as cosmic running,
beam halo running, or particular validation tests.

This class defines the interfaces used by the Tier 0 and Tier 1
processing to wrap calls to ConfigBuilder in order to retrieve all the
configurations for the various types of job

"""

import FWCore.ParameterSet.Config as cms

class Scenario(object):
    """
    _Scenario_

    """
    def __init__(self):
        pass


    def dropOutputModule(self, processRef, moduleName):
        """
        _dropOutputModule_

        Util to prune an unwanted output module

        """
        del process._Process__outputmodules[moduleName]
        return



    def promptReco(self, globalTag, writeTiers = ['RECO']):
        """
        _installPromptReco_

        given a skeleton process object and references
        to the output modules for the products it produces,
        install the standard reco sequences and event content for this
        scenario

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for promptReco"
        raise NotImplementedError, msg



    def alcaReco(self, *skims):
        """
        _alcaReco_

        Given a skeleton process install the alcareco sequences and
        skims.
        For each skim name in the list of skims, install the appropriate
        output module with the name of the skim

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for alcaReco"
        raise NotImplementedError, msg


    def dqmHarvesting(self, datasetName, runNumber,  globalTag,
                      **options):
        """
        _dqmHarvesting_

        build a DQM Harvesting configuration

        Arguments:
        
        datasetName - aka workflow name for DQMServer, this is the name of the
        dataset containing the harvested run
        runNumber - The run being harvested
        globalTag - The global tag being used
        inputFiles - The list of LFNs being harvested

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for dqmHarvesting"
        raise NotImplementedError, msg

    def addExpressOutputModules(self, process, tiers, datasets):
        """
        _addExpressOutputModules_

        Util method to unpack and install the set of data tier
        output modules corresponding to the list of tiers and datasets
        provided

        """
        for tier in tiers:
            for dataset in datasets:
                moduleName = "write%s%s" % (tier, dataset)
                contentName = "%sEventContent" % tier
                contentAttr = getattr(process, contentName)
                setattr(process, moduleName, 

                        cms.OutputModule(
                    "PoolOutputModule", 
                    fileName = cms.untracked.string('%s.root' % moduleName), 
                    dataset = cms.untracked.PSet( 
                    dataTier = cms.untracked.string(tier), 
                    ),
                    eventContent = contentAttr
                    )
                        
                        )
        return

    def expressProcessing(self, globalTag, writeTiers = [],
                          datasets = [], alcaDataset = None ):
        """
        _expressProcessing_

        Build an express processing configuration for this scenario.

        Express processing runs conversion, reco and alca reco on each
        streamer file in the express stream and writes out RAW, RECO and
        a combined ALCA file that gets mergepacked in a later step

        writeTiers is list of tiers to write out, not including ALCA
        
        datasets is the list of datasets to split into for each tier
        written out. Should always be one dataset

        alcaDataset - if set, this means the combined Alca file is written
        out with no dataset splitting, it gets assigned straight to the datase
        provided

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for expressProcessing"
        raise NotImplementedError, msg        


    def expressMergepacking(self, *outputModules):
        """
        _expressMergepacking_

        Build/customise a mergepacking configuration

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for expressMergepacking"
        raise NotImplementedError, msg        

    
    def skimming(self, *skims):
        """
        _skimming_

        Given a process install the sequences for Tier 1 skimming
        and the appropriate output modules

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for skimming"
        raise NotImplementedError, msg        

