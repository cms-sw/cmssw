#!/usr/bin/env python3
# encoding: utf-8
"""
Monitoring_t.py

Created by Dave Evans on 2011-05-19.
Copyright (c) 2011 Fermilab. All rights reserved.
"""

import unittest
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.Monitoring import addMonitoring
from Configuration.DataProcessing.Merge import mergeProcess


class untitled(unittest.TestCase):
    """
    Unittest for Monitoring module
    """
        
        
    def testA(self):
        """ test addMonitoring call"""
        
        # test with a nice simple merge process config
        process = mergeProcess(
                        ["/store/dummyinput.root"],
                        process_name = "unittest",
                        output_file = "dummy.root",
                        output_lfn = "/store/dummy.root")
        
        try:
            addMonitoring(process)
        except Exception as ex:
            msg = "Failed to call addMonitoring on a cms.Process:\n"
            msg += str(ex)
            self.fail(msg)
            
        servicelist = process.services.keys()
        self.failUnless("SimpleMemoryCheck" in servicelist, "SimpleMemoryCheck not in list of services")
        self.failUnless("Timing" in servicelist, "Timing not in list of services")
        

    
if __name__ == '__main__':
    unittest.main()