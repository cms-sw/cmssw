#!/usr/bin/env python3
"""
_HeavyIonsEra_Run2_HI_

Test for Collision Scenario implementation

"""


import unittest
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.GetScenario import getScenario



def writePSetFile(name, process):
    """
    _writePSetFile_

    Util to dump the process to a file

    """
    handle = open(name, 'w')
    handle.write(process.dumpPython())
    handle.close()


class HeavyIonsEra_Run2_HIScenarioTest(unittest.TestCase):
    """
    unittest for HeavyIonsEra_Run2_HI collisions scenario

    """

    def testA(self):
        """get the scenario"""
        try:
            scenario = getScenario("HeavyIonsEra_Run2_HI")
        except Exception as ex:
            msg = "Failed to get HeavyIonsEra_Run2_HI scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testPromptReco(self):
        """test promptReco method"""
        scenario = getScenario("HeavyIonsEra_Run2_HI")
        try:
            process = scenario.promptReco("GLOBALTAG::ALL")
            writePSetFile("testPromptReco.py", process)
        except Exception as ex:
            msg = "Failed to create Prompt Reco configuration\n"
            msg += "for HeavyIonsEra_Run2_HI Scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testExpressProcessing(self):
        """ test expressProcessing method"""
        scenario = getScenario("HeavyIonsEra_Run2_HI")
        try:
            process = scenario.expressProcessing("GLOBALTAG::ALL")
            writePSetFile("testExpressProcessing.py", process)
        except Exception as ex:
            msg = "Failed to create Express Processing configuration\n"
            msg += "for HeavyIonsEra_Run2_HI Scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testAlcaSkim(self):
        """ test alcaSkim method"""
        scenario = getScenario("HeavyIonsEra_Run2_HI")
        try:
            process = scenario.alcaSkim(["MuAlCalIsolatedMu"])
            writePSetFile("testAlcaReco.py", process)
        except Exception as ex:
           msg = "Failed to create Alca Skimming configuration\n"
           msg += "for HeavyIonsEra_Run2_HI Scenario\n"
           msg += str(ex)
           self.fail(msg)


    def testDQMHarvesting(self):
        """test dqmHarvesting  method"""
        scenario = getScenario("HeavyIonsEra_Run2_HI")
        try:
            process = scenario.dqmHarvesting("dataset", 123456,
                                             "GLOBALTAG::ALL")
            writePSetFile("testDQMHarvesting.py", process)
        except Exception as ex:
            msg = "Failed to create DQM Harvesting configuration "
            msg += "for HeavyIonsEra_Run2_HI scenario:\n"
            msg += str(ex)
            self.fail(msg)


if __name__ == '__main__':
    unittest.main()
