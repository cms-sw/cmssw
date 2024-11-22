#! /usr/bin/env python3

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.inputFiles = 'bob','peter'
#options.setNoDefaultClear ('inputFiles')
#options.setNoCommaSplit ('inputFiles')
options.parseArguments()

print(options.inputFiles)
