#! /usr/bin/env python

# A Pyrelval Wrapper

def run():
        import sys
        import os
        import Configuration.Applications
        from Configuration.Applications.ConfigBuilder import ConfigBuilder
        from Configuration.Applications.cmsDriverOptions import OptionsFromCommandLine
        options = OptionsFromCommandLine()
        
        # after cleanup of all config parameters pass it to the ConfigBuilder
        configBuilder = ConfigBuilder(options, with_output = True, with_input = True)
        configBuilder.prepare()
        # fetch the results and write it to file
        config = file(options.python_filename,"w")
        config.write(configBuilder.pythonCfgCode)
        config.close()

        # handle different dump options
        if options.dump_python:
            result = {}
            execfile(options.python_filename, result)
            process = result["process"]
            expanded = process.dumpPython()
            expandedFile = file(options.python_filename,"w")
            expandedFile.write(expanded)
            expandedFile.close()
            print "Expanded config file", options.python_filename, "created"
            sys.exit(0)           
  
        if options.no_exec_flag:
            print "Config file "+options.python_filename+ " created"
            sys.exit(0)
        else:
            commandString = options.prefix+" cmsRun "+options.suffix
            print "Starting "+commandString+' '+options.python_filename
            commands = commandString.lstrip().split()
            os.execvpe(commands[0],commands+[options.python_filename],os.environ)
            sys.exit()

run()


    
