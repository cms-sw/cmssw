#!/usr/bin/env python

"""
BuildIsoMvacs.py

Author: Evan K. Friis, UC Davis friis@physics.ucdavis.edu

Usage: Prepare copies of all MVA configuration files (*.mvac) with the isolation inputs removed.
       New copies will be titled *Iso.xml
"""
import os
import glob
import string

filesToConvert = glob.glob("./*.mvac")

for aFile in filesToConvert:
   xmlFileName = string.replace(aFile, ".mvac", "Iso.xml")
   if os.path.exists(xmlFileName):
      os.system("cp %s %s.bak" % (xmlFileName, xmlFileName))
      os.remove(xmlFileName)

   print "Building %s..." % xmlFileName
   os.system("cat Preamble.xml.fragment >  %s" %         xmlFileName)  
   os.system("cat Inputs.xml.fragment   >> %s" %         xmlFileName)  
   os.system("cat Helpers.xml.fragment  >> %s" %         xmlFileName)  
   os.system("cat %s | grep -v Outlier  >> %s" % (aFile, xmlFileName)) 
   os.system("cat Finale.xml.fragment   >> %s" %         xmlFileName)  
