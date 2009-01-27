#!/usr/bin/env python

"""
MVAConfigBuilder.py

Author: Evan K. Friis, UC Davis friis@physics.ucdavis.edu

Usage: Any objects with file type [name].mvac will be turned into [name].xml with the correct format

A configuration script to automate changes in available MVA input variables (for the PhysicsTools/MVAComputer framework)
across numerous MVA configurations.

This is total kludge and will be deprecated when 
        a. I figure out how you use <xi:include... success full (xml sucks)
        b. Proposed changes to the MVA objects allow classification steering in a single xml file
"""

import os
import glob
import string

filesToConvert = glob.glob("./*.mvac")

for aFile in filesToConvert:
   xmlFileName = string.replace(aFile, "mvac", "xml")
   if os.path.exists(xmlFileName):
      os.system("cp %s %s.bak" % (xmlFileName, xmlFileName))
      os.remove(xmlFileName)

   print "Building %s..." % xmlFileName
   os.system("cat Preamble.xml.fragment >  %s" %         xmlFileName)  
   os.system("cat Inputs.xml.fragment   >> %s" %         xmlFileName)  
   os.system("cat Helpers.xml.fragment  >> %s" %         xmlFileName)  
   os.system("cat %s                    >> %s" % (aFile, xmlFileName)) 
   os.system("cat Finale.xml.fragment   >> %s" %         xmlFileName)  


