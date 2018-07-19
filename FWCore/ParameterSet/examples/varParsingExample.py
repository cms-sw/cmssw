#! /usr/bin/env python

from __future__ import print_function
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.inputFiles = 'bob','peter'
#options.setNoDefaultClear ('inputFiles')
#options.setNoCommaSplit ('inputFiles')
options.parseArguments()

print(options.inputFiles)
