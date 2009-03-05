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


class Scenario(object):
    """
    _Scenario_

    """
    def __init__(self):
        pass





    def promptReco(self, process, recoOutputModule, aodOutputModule = None,
                   alcaOutputModule = None):
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



    def alcaReco(self, process, *skims):
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
                      *inputFiles, **options):
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



    def expressProcessing(self, process):
        """
        _expressProcessing_

        Build an express processing configuration for this scenario.

        Express processing runs conversion, reco and alca reco on each
        streamer file in the express stream and writes out RAW, RECO and
        a combined ALCA file that gets mergepacked in a later step

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for expressProcessing"
        raise NotImplementedError, msg        


    def expressMergepacking(self, process, *outputModules):
        """
        _expressMergepacking_

        Build/customise a mergepacking configuration

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for expressMergepacking"
        raise NotImplementedError, msg        

    
    def skimming(self, process, *skims):
        """
        _skimming_

        Given a process install the sequences for Tier 1 skimming
        and the appropriate output modules

        """
        msg = "Scenario Implementation %s\n" % self.__class__.__name__
        msg += "Does not contain an implementation for skimming"
        raise NotImplementedError, msg        

