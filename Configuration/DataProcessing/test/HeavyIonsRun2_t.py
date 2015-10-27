#!/usr/bin/env python
"""
_HeavyIonsRun2_

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


class HeavyIonsRun2ScenarioTest(unittest.TestCase):
    """
    unittest for HeavyIonsRun2 collisions scenario

    """

    def testA(self):
        """get the scenario"""
        try:
            scenario = getScenario("HeavyIonsRun2")
        except Exception, ex:
            msg = "Failed to get HeavyIonsRun2 scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testPromptReco(self):
        """test promptReco method"""
        scenario = getScenario("HeavyIonsRun2")
        try:
            process = scenario.promptReco("GLOBALTAG::ALL")
            writePSetFile("testPromptReco.py", process)
        except Exception, ex:
            msg = "Failed to create Prompt Reco configuration\n"
            msg += "for HeavyIonsRun2 Scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testExpressProcessing(self):
        """ test expressProcessing method"""
        scenario = getScenario("HeavyIonsRun2")
        try:
            process = scenario.expressProcessing("GLOBALTAG::ALL")
            writePSetFile("testExpressProcessing.py", process)
        except Exception, ex:
            msg = "Failed to create Express Processing configuration\n"
            msg += "for HeavyIonsRun2 Scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testAlcaSkim(self):
        """ test alcaSkim method"""
        scenario = getScenario("HeavyIonsRun2")
        try:
            process = scenario.alcaSkim(["MuAlCalIsolatedMu"])
            writePSetFile("testAlcaReco.py", process)
        except Exception, ex:
           msg = "Failed to create Alca Skimming configuration\n"
           msg += "for HeavyIonsRun2 Scenario\n"
           msg += str(ex)
           self.fail(msg)


    def testDQMHarvesting(self):
        """test dqmHarvesting  method"""
        scenario = getScenario("HeavyIonsRun2")
        try:
            process = scenario.dqmHarvesting("dataset", 123456,
                                             "GLOBALTAG::ALL")
            writePSetFile("testDQMHarvesting.py", process)
        except Exception, ex:
            msg = "Failed to create DQM Harvesting configuration "
            msg += "for HeavyIonsRun2 scenario:\n"
            msg += str(ex)
            self.fail(msg)


if __name__ == '__main__':
    unittest.main()
