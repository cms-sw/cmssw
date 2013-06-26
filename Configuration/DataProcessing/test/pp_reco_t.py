#!/usr/bin/env python
"""
_pp_

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


class ppScenarioTest(unittest.TestCase):
    """
    unittest for pp collisions scenario

    """

    def testA(self):
        """get the scenario"""
        try:
            scenario = getScenario("pp")
        except Exception, ex:
            msg = "Failed to get pp scenario\n"
            msg += str(ex)
            self.fail(msg)


    def testPromptReco(self):
        """test promptReco method"""
        scenario = getScenario("pp")
        try:
            process = scenario.promptReco("FT_R_42_V10A::All",writeTiers = ['RECO', 'AOD', 'ALCARECO', 'DQM'])
            writePSetFile("testPromptReco.py", process)
        except Exception, ex:
            msg = "Failed to create Prompt Reco configuration\n"
            msg += "for pp Scenario\n"
            msg += str(ex)
            self.fail(msg)




if __name__ == '__main__':
    unittest.main()
