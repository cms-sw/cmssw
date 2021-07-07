#!/usr/bin/env python3
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
from Configuration.DataProcessing.Merge import mergeProcess
from Configuration.DataProcessing.Repack import repackProcess

#central import, will be used by all daughter classes anyways
from Configuration.Applications.ConfigBuilder import ConfigBuilder,Options,defaultOptions


class Scenario(object):
    """
    _Scenario_

    """
    def __init__(self):
        self.eras=cms.Modifier()


    def promptReco(self, globalTag, **options):
        """
        _installPromptReco_

        given a skeleton process object and references
        to the output modules for the products it produces,
        install the standard reco sequences and event content for this
        scenario

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for promptReco"
        raise NotImplementedError(msg)


    def expressProcessing(self, globalTag, **options):
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
        raise NotImplementedError(msg)



    def visualizationProcessing(self, globalTag, **options):
        """
        _expressProcessing_

        Build a configuration for the visualization processing for this scenario.

        Visualization processing runs unpacking, and reco on 
        streamer files and it is equipped to run on the online cluster
        and writes RECO or FEVT files,
        
        writeTiers is list of tiers to write out.
        

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for visualizationProcessing"
        raise NotImplementedError(msg)




    def alcaSkim(self, skims, **options):
        """
        _alcaSkim_

        Given a skeleton process install the skim splitting for given skims

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for alcaSkim"
        raise NotImplementedError(msg)


    def alcaReco(self, *skims, **options):
        """
        _alcaSkim_

        Given a skeleton process install the skim production for given skims

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for alcaReco"
        raise NotImplementedError(msg)


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **options):
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
        raise NotImplementedError(msg)


    def alcaHarvesting(self, globalTag, datasetName, **options):
        """
        _alcaHarvesting_

        build an AlCa Harvesting configuration

        Arguments:
        
        globalTag - The global tag being used
        inputFiles - The list of LFNs being harvested

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for alcaHarvesting"
        raise NotImplementedError(msg)


    def skimming(self, skims, globalTag, **options):
        """
        _skimming_

        Given a process install the sequences for Tier 1 skimming
        and the appropriate output modules

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for skimming"
        raise NotImplementedError(msg)        


    def merge(self, *inputFiles, **options):
        """
        _merge_

        builds a merge configuration

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        return mergeProcess(*inputFiles, **options)


    def repack(self, **options):
        """
        _repack_

        builds a repack configuration

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        return repackProcess(**options)


    #
    # helper methods
    #

    def dropOutputModule(self, processRef, moduleName):
        """
        _dropOutputModule_

        Util to prune an unwanted output module

        """
        del process._Process__outputmodules[moduleName]
        return
