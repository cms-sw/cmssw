#!/usr/bin/env python3
"""
_cosmicsEra_Run2_2018_

Test for CosmicsRun2 Scenario implementation

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


class cosmicsEra_Run2_2018ScenarioTest(unittest.TestCase):
    """
    unittest for cosmicsEra_Run2_2018 scenario

    """

    def testA(self):
        """get the scenario"""
        try:
            scenario = getScenario("cosmicsEra_Run2_2018")
        except Exception as ex:
            msg = "Failed to get cosmicsEra_Run2_2018 scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testPromptReco(self):
        """test promptReco method"""
        scenario = getScenario("cosmicsEra_Run2_2018")
        try:
            process = scenario.promptReco("GLOBALTAG::ALL")
            writePSetFile("testPromptReco.py", process)
        except Exception as ex:
            msg = "Failed to create Prompt Reco configuration\n"
            msg += "for cosmicsEra_Run2_2018 Scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testExpressProcessing(self):
        """ test expressProcessing method"""
        scenario = getScenario("cosmicsEra_Run2_2018")
        try:
            process = scenario.expressProcessing("GLOBALTAG::ALL")
            writePSetFile("testExpressProcessing.py", process)
        except Exception as ex:
            msg = "Failed to create Express Processing configuration\n"
            msg += "for cosmicsEra_Run2_2018 Scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testAlcaSkim(self):
        """ test alcaSkim method"""
        scenario = getScenario("cosmicsEra_Run2_2018")
        try:
            process = scenario.alcaSkim(["MuAlCalIsolatedMu"])
            writePSetFile("testAlcaReco.py", process)
        except Exception as ex:
           msg = "Failed to create Alca Skimming configuration\n"
           msg += "for cosmicsEra_Run2_2018 Scenario\n"
           msg += str(ex)
           self.fail(msg)


    def testDQMHarvesting(self):
        """test dqmHarvesting  method"""
        scenario = getScenario("cosmicsEra_Run2_2018")
        try:
            process = scenario.dqmHarvesting("dataset", 123456,
                                             "GLOBALTAG::ALL")
            writePSetFile("testDQMHarvesting.py", process)
        except Exception as ex:
            msg = "Failed to create DQM Harvesting configuration "
            msg += "for cosmicsEra_Run2_2018 scenario:\n"
            msg += str(ex)
            self.fail(msg)


if __name__ == '__main__':
    unittest.main()
