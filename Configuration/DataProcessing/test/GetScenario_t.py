#!/usr/bin/env python3
"""
_GetScenario_

Unittest for GetScenario module

"""

import unittest
from Configuration.DataProcessing.GetScenario import getScenario

class GetScenarioTest(unittest.TestCase):
    """GetScenario module test"""

    def testA(self):
        """test retrieving the Test scenario"""
        try:
            scenario = getScenario("Test")

        except Exception as ex:
            msg = "Failed to get Test scenario:\n"
            msg += str(ex)
            self.fail(msg)


    def testB(self):
        """test retrieving non existent Scenario"""
        
        self.assertRaises(RuntimeError,
                          getScenario, "ThisScenarioDoesNotExist")


        

if __name__ == '__main__':
    unittest.main()

