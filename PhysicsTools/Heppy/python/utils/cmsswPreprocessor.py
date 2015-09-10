import os 
import sys                                                              
import re 
import imp
from PhysicsTools.HeppyCore.framework.config import CFG
class CmsswPreprocessor :
	def __init__(self,configFile,command="cmsRun",options={}) :
		self.configFile=configFile
		self.command=command
                self.options=options
	
	def run(self,component,wd,firstEvent,nEvents):
		print wd,firstEvent,nEvents
		if nEvents is None:
			nEvents = -1

                cmsswConfig = imp.load_source("cmsRunProcess",self.configFile)
		inputfiles= []
		for fn in component.files :
			if not re.match("file:.*",fn) and not re.match("root:.*",fn) :
				fn="file:"+fn
			inputfiles.append(fn)

                # Four cases: 
                # - no options, cmsswConfig with initialize function
                #     run initialize with default parameters
                # - filled options, cmsswConfig with initialize function
                #     pass on options to initialize
                # - no options, classic cmsswConfig
                #     legacy mode
                # - filled options, classic cmsswConfig
                #     legacy mode but warn that options are not passed on

                if hasattr(cmsswConfig, "initialize"):
                        if len(self.options) == 0:                                
                                cmsswConfig.process = cmsswConfig.initialize()                                 
                        else:                                
                                cmsswConfig.process = cmsswConfig.initialize(**self.options)                                 
                else:
                        if len(self.options) == 0:                             
                                pass
                        else:
                                print "WARNING: cmsswPreprocessor received options but can't pass on to cmsswConfig"
                
		cmsswConfig.process.source.fileNames = inputfiles
		cmsswConfig.process.maxEvents.input=nEvents
		#fixme: implement skipEvent / firstevent

		outfilename=wd+"/cmsswPreProcessing.root"
		for outName in cmsswConfig.process.endpath.moduleNames() :
			out = getattr(cmsswConfig.process,outName)
			out.fileName = outfilename
		if not hasattr(component,"options"):
			component.options = CFG(name="postCmsrunOptions")
                #use original as primary and new as secondary 
                #component.options.inputFiles= component.files
		#component.options.secondaryInputFiles=[outfilename]

                #use new as primary and original as secondary
                component.options.secondaryInputFiles= component.files
		component.options.inputFiles=[outfilename]
                component.files=[outfilename]

		configfile=wd+"/cmsRun_config.py"
		f = open(configfile, 'w')
		f.write(cmsswConfig.process.dumpPython())
		f.close()
		runstring="%s %s >& %s/cmsRun.log" % (self.command,configfile,wd)
		print "Running pre-processor: %s " %runstring
                ret=os.system(runstring)
                if ret != 0:
                     print "CMSRUN failed"
                     exit(ret)
		return component
