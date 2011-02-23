#! /usr/bin/env python

# A Pyrelval Wrapper

import sys
import os
import Configuration.PyReleaseValidation
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder, defaultOptions
from Configuration.PyReleaseValidation.cmsDriverOptions import options, python_config_filename

# after cleanup of all config parameters pass it to the ConfigBuilder
configBuilder = ConfigBuilder(options, with_output = True, with_input = True)
configBuilder.prepare()

# fetch the results and write it to file
if options.python_filename: python_config_filename = options.python_filename
config = file(python_config_filename,"w")
config.write(configBuilder.pythonCfgCode)
config.close()

# handle different dump options
if options.dump_python:
    result = {}
    execfile(python_config_filename, result)
    process = result["process"]
    expanded = process.dumpPython()
    expandedFile = file(python_config_filename,"w")
    expandedFile.write(expanded)
    expandedFile.close()
    print "Expanded config file", python_config_filename, "created"
    sys.exit(0)           
  
if options.no_exec_flag:
    print "Config file "+python_config_filename+ " created"
    sys.exit(0)
else:
    commandString = options.prefix+" cmsRun"
    print "Starting "+commandString+' '+python_config_filename
    commands = commandString.lstrip().split()
    os.execvpe(commands[0],commands+[python_config_filename],os.environ)
    sys.exit()
    


    
