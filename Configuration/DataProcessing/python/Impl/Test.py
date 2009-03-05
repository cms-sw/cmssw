#!/usr/bin/env python
"""
_Test_

Test Scenario implementation for unittests/development purposes

Not for use with data taking

"""


from Configuration.DataProcessing.Scenario import Scenario
import FWCore.ParameterSet.Config as cms

class Test(Scenario):
    """
    _Test_

    Test Scenario

    """

    
    def promptReco(process, recoOutputModule, aodOutputModule = None,
                   alcaOutputModule = None):
        """
        _installPromptReco_

        given a skeleton process object and references
        to the output modules for the products it produces,
        install the standard reco sequences and event content for this
        scenario

        """
        return cms.Process("RECO")



    def alcaReco(process, *skims):
        """
        _alcaReco_

        Given a skeleton process install the alcareco sequences and
        skims.
        For each skim name in the list of skims, install the appropriate
        output module with the name of the skim

        """
        return cms.Process("ALCARECO")
        
        
    def dqmHarvesting(datasetName, runNumber,  globalTag,
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
        return cms.Process("DQM")



    def expressProcessing(process):
        """
        _expressProcessing_

        Build an express processing configuration for this scenario.

        Express processing runs conversion, reco and alca reco on each
        streamer file in the express stream and writes out RAW, RECO and
        a combined ALCA file that gets mergepacked in a later step

        """
        return cms.Process("Express")


    def expressMergepacking(process, *outputModules):
        """
        _expressMergepacking_

        Build/customise a mergepacking configuration

        """
        return cms.Process("Express")

    
    def skimming(process, *skims):
        """
        _skimming_

        Given a process install the sequences for Tier 1 skimming
        and the appropriate output modules

        """
        return cms.Process("Express")
        

