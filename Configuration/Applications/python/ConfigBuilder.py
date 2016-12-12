#! /usr/bin/env python

__version__ = "$Revision: 1.19 $"
__source__ = "$Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v $"

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Modules import _Module
import sys
import re
import collections
from subprocess import Popen,PIPE
import FWCore.ParameterSet.DictTypes as DictTypes
class Options:
        pass

# the canonical defaults
defaultOptions = Options()
defaultOptions.datamix = 'DataOnSim'
defaultOptions.isMC=False
defaultOptions.isData=True
defaultOptions.step=''
defaultOptions.pileup='NoPileUp'
defaultOptions.pileup_input = None
defaultOptions.pileup_dasoption = ''
defaultOptions.geometry = 'SimDB'
defaultOptions.geometryExtendedOptions = ['ExtendedGFlash','Extended','NoCastor']
defaultOptions.magField = ''
defaultOptions.conditions = None
defaultOptions.scenarioOptions=['pp','cosmics','nocoll','HeavyIons']
defaultOptions.harvesting= 'AtRunEnd'
defaultOptions.gflash = False
defaultOptions.number = -1
defaultOptions.number_out = None
defaultOptions.arguments = ""
defaultOptions.name = "NO NAME GIVEN"
defaultOptions.evt_type = ""
defaultOptions.filein = ""
defaultOptions.dasquery=""
defaultOptions.dasoption=""
defaultOptions.secondfilein = ""
defaultOptions.customisation_file = []
defaultOptions.customisation_file_unsch = []
defaultOptions.customise_commands = ""
defaultOptions.inline_custom=False
defaultOptions.particleTable = 'pythiapdt'
defaultOptions.particleTableList = ['pythiapdt','pdt']
defaultOptions.dirin = ''
defaultOptions.dirout = ''
defaultOptions.filetype = 'EDM'
defaultOptions.fileout = 'output.root'
defaultOptions.filtername = ''
defaultOptions.lazy_download = False
defaultOptions.custom_conditions = ''
defaultOptions.hltProcess = ''
defaultOptions.eventcontent = None
defaultOptions.datatier = None
defaultOptions.inlineEventContent = True
defaultOptions.inlineObjets =''
defaultOptions.hideGen=False
from Configuration.StandardSequences.VtxSmeared import VtxSmearedDefaultKey,VtxSmearedHIDefaultKey
defaultOptions.beamspot=None
defaultOptions.outputDefinition =''
defaultOptions.inputCommands = None
defaultOptions.outputCommands = None
defaultOptions.inputEventContent = ''
defaultOptions.dropDescendant = False
defaultOptions.relval = None
defaultOptions.slhc = None
defaultOptions.profile = None
defaultOptions.isRepacked = False
defaultOptions.restoreRNDSeeds = False
defaultOptions.donotDropOnInput = ''
defaultOptions.python_filename =''
defaultOptions.io=None
defaultOptions.lumiToProcess=None
defaultOptions.fast=False
defaultOptions.runsAndWeightsForMC = None
defaultOptions.runsScenarioForMC = None
defaultOptions.runUnscheduled = False
defaultOptions.timeoutOutput = False
defaultOptions.nThreads = '1'

# some helper routines
def dumpPython(process,name):
        theObject = getattr(process,name)
        if isinstance(theObject,cms.Path) or isinstance(theObject,cms.EndPath) or isinstance(theObject,cms.Sequence):
                return "process."+name+" = " + theObject.dumpPython("process")
        elif isinstance(theObject,_Module) or isinstance(theObject,cms.ESProducer):
                return "process."+name+" = " + theObject.dumpPython()+"\n"
        else:
                return "process."+name+" = " + theObject.dumpPython()+"\n"
def filesFromList(fileName,s=None):
	import os
	import FWCore.ParameterSet.Config as cms
	prim=[]
	sec=[]
	for line in open(fileName,'r'):
		if line.count(".root")>=2:
			#two files solution...
			entries=line.replace("\n","").split()
			if not entries[0] in prim:
				prim.append(entries[0])
			if not entries[1] in sec:
				sec.append(entries[1])
		elif (line.find(".root")!=-1):
			entry=line.replace("\n","")
			if not entry in prim:
				prim.append(entry)
	if s:
		if not hasattr(s,"fileNames"):
			s.fileNames=cms.untracked.vstring(prim)
		else:
			s.fileNames.extend(prim)
		if len(sec)!=0:
			if not hasattr(s,"secondaryFileNames"):
				s.secondaryFileNames=cms.untracked.vstring(sec)
			else:
				s.secondaryFileNames.extend(sec)
	print "found files: ",prim
	if len(prim)==0:
		raise Exception("There are not files in input from the file list")
	if len(sec)!=0:
		print "found parent files:",sec
	return (prim,sec)
	
def filesFromDASQuery(query,option="",s=None):
	import os,time
	import FWCore.ParameterSet.Config as cms
	prim=[]
	sec=[]
	print "the query is",query
	eC=5
	count=0
	while eC!=0 and count<3:
		if count!=0:
			print 'Sleeping, then retrying DAS'
			time.sleep(100)
		p = Popen('das_client.py %s --query "%s"'%(option,query), stdout=PIPE,shell=True)
                pipe=p.stdout.read()
		tupleP = os.waitpid(p.pid, 0)
		eC=tupleP[1]
		count=count+1
	if eC==0:
		print "DAS succeeded after",count,"attempts",eC
	else:
		print "DAS failed 3 times- I give up"
	for line in pipe.split('\n'):
		if line.count(".root")>=2:
			#two files solution...
			entries=line.replace("\n","").split()
			if not entries[0] in prim:
				prim.append(entries[0])
			if not entries[1] in sec:
				sec.append(entries[1])
		elif (line.find(".root")!=-1):
			entry=line.replace("\n","")
			if not entry in prim:
				prim.append(entry)
	if s:
		if not hasattr(s,"fileNames"):
			s.fileNames=cms.untracked.vstring(prim)
		else:
			s.fileNames.extend(prim)
		if len(sec)!=0:
			if not hasattr(s,"secondaryFileNames"):
				s.secondaryFileNames=cms.untracked.vstring(sec)
			else:
				s.secondaryFileNames.extend(sec)
	print "found files: ",prim
	if len(sec)!=0:
		print "found parent files:",sec
	return (prim,sec)

def MassReplaceInputTag(aProcess,oldT="rawDataCollector",newT="rawDataRepacker"):
	from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
	for s in aProcess.paths_().keys():
		massSearchReplaceAnyInputTag(getattr(aProcess,s),oldT,newT)

def anyOf(listOfKeys,dict,opt=None):
	for k in listOfKeys:
		if k in dict:
			toReturn=dict[k]
			dict.pop(k)
			return toReturn
	if opt!=None:
		return opt
	else:
		raise Exception("any of "+','.join(listOfKeys)+" are mandatory entries of --output options")

class ConfigBuilder(object):
    """The main building routines """

    def __init__(self, options, process = None, with_output = False, with_input = False ):
        """options taken from old cmsDriver and optparse """
	    
        options.outfile_name = options.dirout+options.fileout

        self._options = options

        if self._options.isData and options.isMC:
                raise Exception("ERROR: You may specify only --data or --mc, not both")
        #if not self._options.conditions:
        #        raise Exception("ERROR: No conditions given!\nPlease specify conditions. E.g. via --conditions=IDEAL_30X::All")

	# check that MEtoEDMConverter (running in ENDJOB) and DQMIO don't run in the same job
	if 'ENDJOB' in self._options.step:
		if  (hasattr(self._options,"outputDefinition") and \
		    self._options.outputDefinition != '' and \
		    any(anyOf(['t','tier','dataTier'],outdic) == 'DQMIO' for outdic in eval(self._options.outputDefinition))) or \
		    (hasattr(self._options,"datatier") and \
		    self._options.datatier and \
		    'DQMIO' in self._options.datatier):
			print "removing ENDJOB from steps since not compatible with DQMIO dataTier" 
			self._options.step=self._options.step.replace(',ENDJOB','')


		
        # what steps are provided by this class?
        stepList = [re.sub(r'^prepare_', '', methodName) for methodName in ConfigBuilder.__dict__ if methodName.startswith('prepare_')]
        self.stepMap={}
	self.stepKeys=[]
        for step in self._options.step.split(","):
                if step=='': continue
                stepParts = step.split(":")
                stepName = stepParts[0]
                if stepName not in stepList and not stepName.startswith('re'):
                        raise ValueError("Step "+stepName+" unknown")
                if len(stepParts)==1:
                        self.stepMap[stepName]=""
                elif len(stepParts)==2:
                        self.stepMap[stepName]=stepParts[1].split('+')
                elif len(stepParts)==3:
                        self.stepMap[stepName]=(stepParts[2].split('+'),stepParts[1])
                else:
                        raise ValueError("Step definition "+step+" invalid")
		self.stepKeys.append(stepName)
		
        #print "map of steps is:",self.stepMap

        self.with_output = with_output
        if hasattr(self._options,"no_output_flag") and self._options.no_output_flag:
                self.with_output = False
        self.with_input = with_input
        if process == None:
            self.process = cms.Process(self._options.name)
        else:
            self.process = process
        self.imports = []
        self.importsUnsch = []
        self.define_Configs()
        self.schedule = list()

        # we are doing three things here:
        # creating a process to catch errors
        # building the code to re-create the process

        self.additionalCommands = []
        # TODO: maybe a list of to be dumped objects would help as well
        self.blacklist_paths = []
        self.addedObjects = []
        self.additionalOutputs = {}

        self.productionFilterSequence = None
	self.nextScheduleIsConditional=False
	self.conditionalPaths=[]
	self.excludedPaths=[]

    def profileOptions(self):
	    """
	    addIgProfService
	    Function to add the igprof profile service so that you can dump in the middle
	    of the run.
	    """
	    profileOpts = self._options.profile.split(':')
	    profilerStart = 1
	    profilerInterval = 100
	    profilerFormat = None
	    profilerJobFormat = None
	    
	    if len(profileOpts):
		    #type, given as first argument is unused here
		    profileOpts.pop(0) 
	    if len(profileOpts):   
		    startEvent = profileOpts.pop(0)
		    if not startEvent.isdigit():
			    raise Exception("%s is not a number" % startEvent)
		    profilerStart = int(startEvent)
	    if len(profileOpts):
		    eventInterval = profileOpts.pop(0)
		    if not eventInterval.isdigit():
			    raise Exception("%s is not a number" % eventInterval)
		    profilerInterval = int(eventInterval)
	    if len(profileOpts):
		    profilerFormat = profileOpts.pop(0)


	    if not profilerFormat:
		    profilerFormat = "%s___%s___%s___%s___%s___%s___%%I.gz" % (self._options.evt_type.replace("_cfi", ""),
									       self._options.step,
									       self._options.pileup,
									       self._options.conditions,
									       self._options.datatier,
									       self._options.profileTypeLabel)
	    if not profilerJobFormat and profilerFormat.endswith(".gz"):
		    profilerJobFormat = profilerFormat.replace(".gz", "_EndOfJob.gz")
	    elif not profilerJobFormat:
		    profilerJobFormat = profilerFormat + "_EndOfJob.gz"

	    return (profilerStart,profilerInterval,profilerFormat,profilerJobFormat)

    def load(self,includeFile):
        includeFile = includeFile.replace('/','.')
        self.process.load(includeFile)
        return sys.modules[includeFile]

    def loadAndRemember(self, includeFile,unsch=0):
        """helper routine to load am memorize imports"""
        # we could make the imports a on-the-fly data method of the process instance itself
        # not sure if the latter is a good idea
        includeFile = includeFile.replace('/','.')
	if unsch==0:
		self.imports.append(includeFile)
		self.process.load(includeFile)
		return sys.modules[includeFile]
	else:
		self.importsUnsch.append(includeFile)
		return 0#sys.modules[includeFile]

    def executeAndRemember(self, command):
        """helper routine to remember replace statements"""
        self.additionalCommands.append(command)
        if not command.strip().startswith("#"):
            # substitute: process.foo = process.bar -> self.process.foo = self.process.bar
            import re
            exec(re.sub(r"([^a-zA-Z_0-9]|^)(process)([^a-zA-Z_0-9])",r"\1self.process\3",command))
            #exec(command.replace("process.","self.process."))

    def addCommon(self):
            if 'HARVESTING' in self.stepMap.keys() or 'ALCAHARVEST' in self.stepMap.keys():
                    self.process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound'),fileMode = cms.untracked.string('FULLMERGE'))
            else:
                    self.process.options = cms.untracked.PSet( )

	    if self._options.runUnscheduled:
		    self.process.options.allowUnscheduled=cms.untracked.bool(True)
		    
            self.addedObjects.append(("","options"))

            if self._options.lazy_download:
                    self.process.AdaptorConfig = cms.Service("AdaptorConfig",
                                                             stats = cms.untracked.bool(True),
                                                             enable = cms.untracked.bool(True),
                                                             cacheHint = cms.untracked.string("lazy-download"),
                                                             readHint = cms.untracked.string("read-ahead-buffered")
                                                             )
                    self.addedObjects.append(("Setup lazy download","AdaptorConfig"))

            #self.process.cmsDriverCommand = cms.untracked.PSet( command=cms.untracked.string('cmsDriver.py '+self._options.arguments) )
            #self.addedObjects.append(("what cmsDriver command was used","cmsDriverCommand"))

	    if self._options.profile:
		    (start, interval, eventFormat, jobFormat)=self.profileOptions()
		    self.process.IgProfService = cms.Service("IgProfService",
							     reportFirstEvent            = cms.untracked.int32(start),
							     reportEventInterval         = cms.untracked.int32(interval),
							     reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > %s"%(eventFormat)),
							     reportToFileAtPostEndJob    = cms.untracked.string("| gzip -c > %s"%(jobFormat)))
		    self.addedObjects.append(("Setup IGProf Service for profiling","IgProfService"))
							     
    def addMaxEvents(self):
        """Here we decide how many evts will be processed"""
        self.process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(int(self._options.number)))
	if self._options.number_out:
		self.process.maxEvents.output = cms.untracked.int32(int(self._options.number_out))
        self.addedObjects.append(("","maxEvents"))

    def addSource(self):
        """Here the source is built. Priority: file, generator"""
        self.addedObjects.append(("Input source","source"))

	def filesFromOption(self):
		for entry in self._options.filein.split(','):
			print "entry",entry
			if entry.startswith("filelist:"):
				filesFromList(entry[9:],self.process.source)
			elif entry.startswith("dbs:") or entry.startswith("das:"):
				filesFromDASQuery('file dataset = %s'%(entry[4:]),self._options.dasoption,self.process.source)
			else:
				self.process.source.fileNames.append(self._options.dirin+entry)
		if self._options.secondfilein:
			if not hasattr(self.process.source,"secondaryFileNames"):
				raise Exception("--secondfilein not compatible with "+self._options.filetype+"input type")
			for entry in self._options.secondfilein.split(','):
				print "entry",entry
				if entry.startswith("filelist:"):
					self.process.source.secondaryFileNames.extend((filesFromList(entry[9:]))[0])
				elif entry.startswith("dbs:") or entry.startswith("das:"):
					self.process.source.secondaryFileNames.extend((filesFromDASQuery('file dataset = %s'%(entry[4:]),self._options.dasoption))[0])
				else:
					self.process.source.secondaryFileNames.append(self._options.dirin+entry)

        if self._options.filein or self._options.dasquery:
	   if self._options.filetype == "EDM":
		   self.process.source=cms.Source("PoolSource",
						  fileNames = cms.untracked.vstring(),
						  secondaryFileNames= cms.untracked.vstring())
		   filesFromOption(self)
	   elif self._options.filetype == "DAT":
		   self.process.source=cms.Source("NewEventStreamFileReader",fileNames = cms.untracked.vstring())
		   filesFromOption(self)
           elif self._options.filetype == "LHE":
		   self.process.source=cms.Source("LHESource", fileNames = cms.untracked.vstring())
		   if self._options.filein.startswith("lhe:"):
			   #list the article directory automatically
			   args=self._options.filein.split(':')
			   article=args[1]
			   print 'LHE input from article ',article
			   location='/store/lhe/'
			   import os
			   textOfFiles=os.popen('cmsLHEtoEOSManager.py -l '+article)
			   for line in textOfFiles:
				   for fileName in [x for x in line.split() if '.lhe' in x]:
					   self.process.source.fileNames.append(location+article+'/'+fileName)
			   if len(args)>2:
				   self.process.source.skipEvents = cms.untracked.uint32(int(args[2]))
		   else:
			   filesFromOption(self)

		   
	   elif self._options.filetype == "DQM":
		   self.process.source=cms.Source("DQMRootSource",
						  fileNames = cms.untracked.vstring())
		   filesFromOption(self)

	   elif self._options.filetype == "DQMDAQ":
		   # FIXME: how to configure it if there are no input files specified?
		   self.process.source=cms.Source("DQMStreamerReader")
		   
			   
           if ('HARVESTING' in self.stepMap.keys() or 'ALCAHARVEST' in self.stepMap.keys()) and (not self._options.filetype == "DQM"):
               self.process.source.processingMode = cms.untracked.string("RunsAndLumis")

	if self._options.dasquery!='':
               self.process.source=cms.Source("PoolSource", fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
	       filesFromDASQuery(self._options.dasquery,self._options.dasoption,self.process.source)

	##drop LHEXMLStringProduct on input to save memory if appropriate
	if 'GEN' in self.stepMap.keys():
        	if self._options.inputCommands:
        		self._options.inputCommands+=',drop LHEXMLStringProduct_*_*_*,'
		else:
			self._options.inputCommands='keep *, drop LHEXMLStringProduct_*_*_*,'    

	if self.process.source and self._options.inputCommands:
		if not hasattr(self.process.source,'inputCommands'): self.process.source.inputCommands=cms.untracked.vstring()
		for command in self._options.inputCommands.split(','):
			# remove whitespace around the keep/drop statements
			command = command.strip()
			if command=='': continue
			self.process.source.inputCommands.append(command)
		if not self._options.dropDescendant:
			self.process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool(False)

	if self._options.lumiToProcess:
		import FWCore.PythonUtilities.LumiList as LumiList
		self.process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange( LumiList.LumiList(self._options.lumiToProcess).getCMSSWString().split(',') )

        if 'GEN' in self.stepMap.keys() or 'LHE' in self.stepMap or (not self._options.filein and hasattr(self._options, "evt_type")):
            if self.process.source is None:
                self.process.source=cms.Source("EmptySource")

	# modify source in case of run-dependent MC
	self.runsAndWeights=None
	if self._options.runsAndWeightsForMC or self._options.runsScenarioForMC :
		if not self._options.isMC :
			raise Exception("options --runsAndWeightsForMC and --runsScenarioForMC are only valid for MC")
		if self._options.runsAndWeightsForMC:
			self.runsAndWeights = eval(self._options.runsAndWeightsForMC)
		else:
			from Configuration.StandardSequences.RunsAndWeights import RunsAndWeights
			if type(RunsAndWeights[self._options.runsScenarioForMC])==str:
				__import__(RunsAndWeights[self._options.runsScenarioForMC])
				self.runsAndWeights = sys.modules[RunsAndWeights[self._options.runsScenarioForMC]].runProbabilityDistribution
			else:
				self.runsAndWeights = RunsAndWeights[self._options.runsScenarioForMC]
				
	if self.runsAndWeights:
		import SimGeneral.Configuration.ThrowAndSetRandomRun as ThrowAndSetRandomRun
		ThrowAndSetRandomRun.throwAndSetRandomRun(self.process.source,self.runsAndWeights)
		self.additionalCommands.append('import SimGeneral.Configuration.ThrowAndSetRandomRun as ThrowAndSetRandomRun')
		self.additionalCommands.append('ThrowAndSetRandomRun.throwAndSetRandomRun(process.source,%s)'%(self.runsAndWeights))

        return

    def addOutput(self):
        """ Add output module to the process """
	result=""
	if self._options.outputDefinition:
		if self._options.datatier:
			print "--datatier & --eventcontent options ignored"
							
		#new output convention with a list of dict
		outList = eval(self._options.outputDefinition)
		for (id,outDefDict) in enumerate(outList):
			outDefDictStr=outDefDict.__str__()
			if not isinstance(outDefDict,dict):
				raise Exception("--output needs to be passed a list of dict"+self._options.outputDefinition+" is invalid")
		        #requires option: tier
			theTier=anyOf(['t','tier','dataTier'],outDefDict)
		        #optional option: eventcontent, filtername, selectEvents, moduleLabel, filename
			## event content
			theStreamType=anyOf(['e','ec','eventContent','streamType'],outDefDict,theTier)
			theFilterName=anyOf(['f','ftN','filterName'],outDefDict,'')
			theSelectEvent=anyOf(['s','sE','selectEvents'],outDefDict,'')
			theModuleLabel=anyOf(['l','mL','moduleLabel'],outDefDict,'')
			theExtraOutputCommands=anyOf(['o','oC','outputCommands'],outDefDict,'')
			# module label has a particular role
			if not theModuleLabel:
				tryNames=[theStreamType.replace(theTier.replace('-',''),'')+theTier.replace('-','')+'output',
					  theStreamType.replace(theTier.replace('-',''),'')+theTier.replace('-','')+theFilterName+'output',
					  theStreamType.replace(theTier.replace('-',''),'')+theTier.replace('-','')+theFilterName+theSelectEvent.split(',')[0].replace(':','for').replace(' ','')+'output'
					  ]
				for name in tryNames:
					if not hasattr(self.process,name):
						theModuleLabel=name
						break
			if not theModuleLabel:
				raise Exception("cannot find a module label for specification: "+outDefDictStr)
			if id==0:
				defaultFileName=self._options.outfile_name
			else:
				defaultFileName=self._options.outfile_name.replace('.root','_in'+theTier+'.root')
				
			theFileName=self._options.dirout+anyOf(['fn','fileName'],outDefDict,defaultFileName)
			if not theFileName.endswith('.root'):
				theFileName+='.root'
				
			if len(outDefDict.keys()):
				raise Exception("unused keys from --output options: "+','.join(outDefDict.keys()))
			if theStreamType=='DQMIO': theStreamType='DQM'
			if theStreamType=='ALL':
				theEventContent = cms.PSet(outputCommands = cms.untracked.vstring('keep *'))
			else:
				theEventContent = getattr(self.process, theStreamType+"EventContent")


			addAlCaSelects=False
			if theStreamType=='ALCARECO' and not theFilterName:
				theFilterName='StreamALCACombined'
				addAlCaSelects=True

			CppType='PoolOutputModule'
			if self._options.timeoutOutput:
				CppType='TimeoutPoolOutputModule'
			if theStreamType=='DQM' and theTier=='DQMIO': CppType='DQMRootOutputModule'
			output = cms.OutputModule(CppType,			
						  theEventContent.clone(),
						  fileName = cms.untracked.string(theFileName),
						  dataset = cms.untracked.PSet(
				                     dataTier = cms.untracked.string(theTier),
						     filterName = cms.untracked.string(theFilterName))
						  )
			if not theSelectEvent and hasattr(self.process,'generation_step') and theStreamType!='LHE':
				output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step'))
			if not theSelectEvent and hasattr(self.process,'filtering_step'):
				output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filtering_step'))				
			if theSelectEvent:
				output.SelectEvents =cms.untracked.PSet(SelectEvents = cms.vstring(theSelectEvent))

			if addAlCaSelects:
				if not hasattr(output,'SelectEvents'):
					output.SelectEvents=cms.untracked.PSet(SelectEvents=cms.vstring())
				for alca in self.AlCaPaths:
					output.SelectEvents.SelectEvents.extend(getattr(self.process,'OutALCARECO'+alca).SelectEvents.SelectEvents)

			
			if hasattr(self.process,theModuleLabel):
				raise Exception("the current process already has a module "+theModuleLabel+" defined")
			#print "creating output module ",theModuleLabel
			setattr(self.process,theModuleLabel,output)
			outputModule=getattr(self.process,theModuleLabel)
			setattr(self.process,theModuleLabel+'_step',cms.EndPath(outputModule))
			path=getattr(self.process,theModuleLabel+'_step')
			self.schedule.append(path)

			if not self._options.inlineEventContent and hasattr(self.process,theStreamType+"EventContent"):
				def doNotInlineEventContent(instance,label = "cms.untracked.vstring(process."+theStreamType+"EventContent.outputCommands)"):
					return label
				outputModule.outputCommands.__dict__["dumpPython"] = doNotInlineEventContent
			if theExtraOutputCommands:
				if not isinstance(theExtraOutputCommands,list):
					raise Exception("extra ouput command in --option must be a list of strings")
				if hasattr(self.process,theStreamType+"EventContent"):
					self.executeAndRemember('process.%s.outputCommands.extend(%s)'%(theModuleLabel,theExtraOutputCommands))
				else:
					outputModule.outputCommands.extend(theExtraOutputCommands)

			result+="\nprocess."+theModuleLabel+" = "+outputModule.dumpPython()

		##ends the --output options model
		return result

	streamTypes=self._options.eventcontent.split(',')
	tiers=self._options.datatier.split(',')
	if not self._options.outputDefinition and len(streamTypes)!=len(tiers):
		raise Exception("number of event content arguments does not match number of datatier arguments")

        # if the only step is alca we don't need to put in an output
        if self._options.step.split(',')[0].split(':')[0] == 'ALCA':
            return "\n"

        for i,(streamType,tier) in enumerate(zip(streamTypes,tiers)):
		if streamType=='': continue
		if streamType=='DQMIO': streamType='DQM'
                theEventContent = getattr(self.process, streamType+"EventContent")
                if i==0:
                        theFileName=self._options.outfile_name
                        theFilterName=self._options.filtername
                else:
                        theFileName=self._options.outfile_name.replace('.root','_in'+streamType+'.root')
                        theFilterName=self._options.filtername
		CppType='PoolOutputModule'
		if self._options.timeoutOutput:
			CppType='TimeoutPoolOutputModule'
		if streamType=='DQM' and tier=='DQMIO': CppType='DQMRootOutputModule'
                output = cms.OutputModule(CppType,
                                          theEventContent,
                                          fileName = cms.untracked.string(theFileName),
                                          dataset = cms.untracked.PSet(dataTier = cms.untracked.string(tier),
                                                                       filterName = cms.untracked.string(theFilterName)
                                                                       )
                                          )
                if hasattr(self.process,"generation_step") and streamType!='LHE':
                        output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step'))
		if hasattr(self.process,"filtering_step"):
                        output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filtering_step'))

                if streamType=='ALCARECO':
                        output.dataset.filterName = cms.untracked.string('StreamALCACombined')

                if "MINIAOD" in streamType:
                    output.dropMetaData = cms.untracked.string('ALL')
                    output.fastCloning= cms.untracked.bool(False)
                    output.overrideInputFileSplitLevels = cms.untracked.bool(True)                      

                outputModuleName=streamType+'output'
                setattr(self.process,outputModuleName,output)
                outputModule=getattr(self.process,outputModuleName)
                setattr(self.process,outputModuleName+'_step',cms.EndPath(outputModule))
                path=getattr(self.process,outputModuleName+'_step')
                self.schedule.append(path)

		if self._options.outputCommands and streamType!='DQM':
			for evct in self._options.outputCommands.split(','):
				if not evct: continue
				self.executeAndRemember("process.%s.outputCommands.append('%s')"%(outputModuleName,evct.strip()))
				
                if not self._options.inlineEventContent:
                        def doNotInlineEventContent(instance,label = "process."+streamType+"EventContent.outputCommands"):
                                return label
                        outputModule.outputCommands.__dict__["dumpPython"] = doNotInlineEventContent

                result+="\nprocess."+outputModuleName+" = "+outputModule.dumpPython()

        return result

    def addStandardSequences(self):
        """
        Add selected standard sequences to the process
        """
        # load the pile up file
	if self._options.pileup:
		pileupSpec=self._options.pileup.split(',')[0]

		# Does the requested pile-up scenario exist?
		from Configuration.StandardSequences.Mixing import Mixing,defineMixing
		if not pileupSpec in Mixing and '.' not in pileupSpec and 'file:' not in pileupSpec:
			message = pileupSpec+' is not a know mixing scenario:\n available are: '+'\n'.join(Mixing.keys())
			raise Exception(message)

		# Put mixing parameters in a dictionary
		if '.' in pileupSpec:
			mixingDict={'file':pileupSpec}
		elif pileupSpec.startswith('file:'):
			mixingDict={'file':pileupSpec[5:]}
		else:
			import copy
			mixingDict=copy.copy(Mixing[pileupSpec])
		if len(self._options.pileup.split(','))>1:
			mixingDict.update(eval(self._options.pileup[self._options.pileup.find(',')+1:]))

		# Load the pu cfg file corresponding to the requested pu scenario
		if 'file:' in pileupSpec:
			#the file is local
			self.process.load(mixingDict['file'])
			print "inlining mixing module configuration"
			self._options.inlineObjets+=',mix'
		else:
			self.loadAndRemember(mixingDict['file'])

		mixingDict.pop('file')
		if not "DATAMIX" in self.stepMap.keys(): # when DATAMIX is present, pileup_input refers to pre-mixed GEN-RAW
			if self._options.pileup_input:
				if self._options.pileup_input.startswith('dbs:') or self._options.pileup_input.startswith('das:'):
					mixingDict['F']=filesFromDASQuery('file dataset = %s'%(self._options.pileup_input[4:],),self._options.pileup_dasoption)[0]
				else:
					mixingDict['F']=self._options.pileup_input.split(',')
			specialization=defineMixing(mixingDict)
			for command in specialization:
				self.executeAndRemember(command)
			if len(mixingDict)!=0:
				raise Exception('unused mixing specification: '+mixingDict.keys().__str__())


        # load the geometry file
        try:
		if len(self.stepMap):
			self.loadAndRemember(self.GeometryCFF)
			if ('SIM' in self.stepMap or 'reSIM' in self.stepMap) and not self._options.fast:
				self.loadAndRemember(self.SimGeometryCFF)
				if self.geometryDBLabel:
					self.executeAndRemember('process.XMLFromDBSource.label = cms.string("%s")'%(self.geometryDBLabel))
        except ImportError:
                print "Geometry option",self._options.geometry,"unknown."
                raise

	if len(self.stepMap):
		self.loadAndRemember(self.magFieldCFF)

	for stepName in self.stepKeys:
		stepSpec = self.stepMap[stepName]
		print "Step:", stepName,"Spec:",stepSpec
		if stepName.startswith('re'):
			##add the corresponding input content
			if stepName[2:] not in self._options.donotDropOnInput:
				self._options.inputEventContent='%s,%s'%(stepName.upper(),self._options.inputEventContent)
			stepName=stepName[2:]
		if stepSpec=="":
			getattr(self,"prepare_"+stepName)(sequence = getattr(self,stepName+"DefaultSeq"))
		elif type(stepSpec)==list:
			getattr(self,"prepare_"+stepName)(sequence = '+'.join(stepSpec))
		elif type(stepSpec)==tuple:
			getattr(self,"prepare_"+stepName)(sequence = ','.join([stepSpec[1],'+'.join(stepSpec[0])]))
		else:
			raise ValueError("Invalid step definition")
		
	if self._options.restoreRNDSeeds!=False:
		#it is either True, or a process name
		if self._options.restoreRNDSeeds==True:
			self.executeAndRemember('process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")')
		else:
			self.executeAndRemember('process.RandomNumberGeneratorService.restoreStateTag=cms.untracked.InputTag("randomEngineStateProducer","","%s")'%(self._options.restoreRNDSeeds))
		if self._options.inputEventContent or self._options.inputCommands:
			if self._options.inputCommands:
				self._options.inputCommands+='keep *_randomEngineStateProducer_*_*,'
			else:
				self._options.inputCommands='keep *_randomEngineStateProducer_*_*,'
					

    def completeInputCommand(self):
	    if self._options.inputEventContent:
		    import copy
		    def dropSecondDropStar(iec):
			    #drop occurence of 'drop *' in the list
			    count=0
			    for item in iec:
				    if item=='drop *':
					    if count!=0:
						    iec.remove(item)
					    count+=1


		    ## allow comma separated input eventcontent
		    if not hasattr(self.process.source,'inputCommands'): self.process.source.inputCommands=cms.untracked.vstring()
		    for evct in self._options.inputEventContent.split(','):
			    if evct=='': continue
			    theEventContent = getattr(self.process, evct+"EventContent")
			    if hasattr(theEventContent,'outputCommands'):
				    self.process.source.inputCommands.extend(copy.copy(theEventContent.outputCommands))
			    if hasattr(theEventContent,'inputCommands'):
				    self.process.source.inputCommands.extend(copy.copy(theEventContent.inputCommands))

		    dropSecondDropStar(self.process.source.inputCommands)
		    
		    if not self._options.dropDescendant:
			    self.process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool(False)

			    
	    return 

    def addConditions(self):
        """Add conditions to the process"""
	if not self._options.conditions: return
	
	if 'FrontierConditions_GlobalTag' in self._options.conditions:
		print 'using FrontierConditions_GlobalTag in --conditions is not necessary anymore and will be deprecated soon. please update your command line'
		self._options.conditions = self._options.conditions.replace("FrontierConditions_GlobalTag,",'')
						
        self.loadAndRemember(self.ConditionsDefaultCFF)
        from Configuration.AlCa.GlobalTag import GlobalTag
        self.process.GlobalTag = GlobalTag(self.process.GlobalTag, self._options.conditions, self._options.custom_conditions)
        self.additionalCommands.append('from Configuration.AlCa.GlobalTag import GlobalTag')
        self.additionalCommands.append('process.GlobalTag = GlobalTag(process.GlobalTag, %s, %s)' % (repr(self._options.conditions), repr(self._options.custom_conditions)))

	if self._options.slhc:
		self.loadAndRemember("SLHCUpgradeSimulations/Geometry/fakeConditions_%s_cff"%(self._options.slhc,))
		

    def addCustomise(self,unsch=0):
        """Include the customise code """

	custOpt=[]
	if unsch==0:
		for c in self._options.customisation_file:
			custOpt.extend(c.split(","))
	else:
		for c in self._options.customisation_file_unsch:
			custOpt.extend(c.split(","))

	custMap=DictTypes.SortedKeysDict()
	for opt in custOpt:
		if opt=='': continue
		if opt.count('.')>1:
			raise Exception("more than . in the specification:"+opt)
		fileName=opt.split('.')[0]
		if opt.count('.')==0:	rest='customise'
		else:
			rest=opt.split('.')[1]
			if rest=='py': rest='customise' #catch the case of --customise file.py
			
		if fileName in custMap:
			custMap[fileName].extend(rest.split('+'))
		else:
			custMap[fileName]=rest.split('+')
		
        if len(custMap)==0:
                final_snippet='\n'
        else:
                final_snippet='\n# customisation of the process.\n'

	allFcn=[]
	for opt in custMap:
		allFcn.extend(custMap[opt])
	for fcn in allFcn:
		if allFcn.count(fcn)!=1:
			raise Exception("cannot specify twice "+fcn+" as a customisation method") 

	for f in custMap:
		# let python search for that package and do syntax checking at the same time
                packageName = f.replace(".py","").replace("/",".")
                __import__(packageName)
                package = sys.modules[packageName]

                # now ask the package for its definition and pick .py instead of .pyc
                customiseFile = re.sub(r'\.pyc$', '.py', package.__file__)
		
                final_snippet+='\n# Automatic addition of the customisation function from '+packageName+'\n'
		if self._options.inline_custom:
			for line in file(customiseFile,'r'):
				if "import FWCore.ParameterSet.Config" in line:
					continue
				final_snippet += line
		else:
			final_snippet += 'from %s import %s \n'%(packageName,','.join(custMap[f]))
		for fcn in custMap[f]:
			print "customising the process with",fcn,"from",f
			if not hasattr(package,fcn):
				#bound to fail at run time
				raise Exception("config "+f+" has no function "+fcn)
			#execute the command
			self.process=getattr(package,fcn)(self.process)
			#and print it in the configuration
			final_snippet += "\n#call to customisation function "+fcn+" imported from "+packageName
			final_snippet += "\nprocess = %s(process)\n"%(fcn,)

	if len(custMap)!=0:
		final_snippet += '\n# End of customisation functions\n'

	### now for a useful command
	if unsch==1 or not self._options.runUnscheduled:
		if self._options.customise_commands:
			import string
			final_snippet +='\n# Customisation from command line'
			for com in self._options.customise_commands.split('\\n'):
				com=string.lstrip(com)
				self.executeAndRemember(com)
				final_snippet +='\n'+com

        return final_snippet

    #----------------------------------------------------------------------------
    # here the methods to define the python includes for each step or
    # conditions
    #----------------------------------------------------------------------------
    def define_Configs(self):
        if len(self.stepMap):
		self.loadAndRemember('Configuration/StandardSequences/Services_cff')
        if self._options.particleTable not in defaultOptions.particleTableList:
            print 'Invalid particle table provided. Options are:'
            print defaultOptions.particleTable
            sys.exit(-1)
        else:
	    if len(self.stepMap):
		    self.loadAndRemember('SimGeneral.HepPDTESSource.'+self._options.particleTable+'_cfi')

        self.loadAndRemember('FWCore/MessageService/MessageLogger_cfi')

        self.ALCADefaultCFF="Configuration/StandardSequences/AlCaRecoStreams_cff"
        self.GENDefaultCFF="Configuration/StandardSequences/Generator_cff"
        self.SIMDefaultCFF="Configuration/StandardSequences/Sim_cff"
        self.DIGIDefaultCFF="Configuration/StandardSequences/Digi_cff"
        self.DIGI2RAWDefaultCFF="Configuration/StandardSequences/DigiToRaw_cff"
        self.L1EMDefaultCFF='Configuration/StandardSequences/SimL1Emulator_cff'
        self.L1MENUDefaultCFF="Configuration/StandardSequences/L1TriggerDefaultMenu_cff"
        self.HLTDefaultCFF="Configuration/StandardSequences/HLTtable_cff"
        self.RAW2DIGIDefaultCFF="Configuration/StandardSequences/RawToDigi_Data_cff"
        self.L1RecoDefaultCFF="Configuration/StandardSequences/L1Reco_cff"
        self.L1TrackTriggerDefaultCFF="Configuration/StandardSequences/L1TrackTrigger_cff"
        self.RECODefaultCFF="Configuration/StandardSequences/Reconstruction_Data_cff"
        self.PATDefaultCFF="Configuration/StandardSequences/PAT_cff"
	self.EIDefaultCFF=None
        self.SKIMDefaultCFF="Configuration/StandardSequences/Skims_cff"
        self.POSTRECODefaultCFF="Configuration/StandardSequences/PostRecoGenerator_cff"
        self.VALIDATIONDefaultCFF="Configuration/StandardSequences/Validation_cff"
        self.L1HwValDefaultCFF = "Configuration/StandardSequences/L1HwVal_cff"
        self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOffline_cff"
        self.HARVESTINGDefaultCFF="Configuration/StandardSequences/Harvesting_cff"
        self.ALCAHARVESTDefaultCFF="Configuration/StandardSequences/AlCaHarvesting_cff"
        self.ENDJOBDefaultCFF="Configuration/StandardSequences/EndOfProcess_cff"
        self.ConditionsDefaultCFF = "Configuration/StandardSequences/FrontierConditions_GlobalTag_cff"
        self.CFWRITERDefaultCFF = "Configuration/StandardSequences/CrossingFrameWriter_cff"
        self.REPACKDefaultCFF="Configuration/StandardSequences/DigiToRaw_Repack_cff"

        if "DATAMIX" in self.stepMap.keys():
            self.DATAMIXDefaultCFF="Configuration/StandardSequences/DataMixer"+self._options.datamix+"_cff"
	    if self._options.datamix == 'PreMix':
		    self.DIGIDefaultCFF="Configuration/StandardSequences/DigiDMPreMix_cff"
	    else:
	            self.DIGIDefaultCFF="Configuration/StandardSequences/DigiDM_cff"
            self.DIGI2RAWDefaultCFF="Configuration/StandardSequences/DigiToRawDM_cff"
            self.L1EMDefaultCFF='Configuration/StandardSequences/SimL1EmulatorDM_cff'

	if "DIGIPREMIX" in self.stepMap.keys():
            self.DIGIDefaultCFF="Configuration/StandardSequences/Digi_PreMix_cff"

        self.ALCADefaultSeq=None
	self.LHEDefaultSeq='externalLHEProducer'
        self.GENDefaultSeq='pgen'
        self.SIMDefaultSeq='psim'
        self.DIGIDefaultSeq='pdigi'
	self.DIGIPREMIXDefaultSeq='pdigi'
	self.DIGIPREMIX_S2DefaultSeq='pdigi'
        self.DATAMIXDefaultSeq=None
        self.DIGI2RAWDefaultSeq='DigiToRaw'
        self.HLTDefaultSeq='GRun'
        self.L1DefaultSeq=None
        self.L1REPACKDefaultSeq='GT'
        self.HARVESTINGDefaultSeq=None
        self.ALCAHARVESTDefaultSeq=None
        self.CFWRITERDefaultSeq=None
        self.RAW2DIGIDefaultSeq='RawToDigi'
        self.L1RecoDefaultSeq='L1Reco'
        self.L1TrackTriggerDefaultSeq='L1TrackTrigger'
        if self._options.fast or ('RAW2DIGI' in self.stepMap and 'RECO' in self.stepMap):
                self.RECODefaultSeq='reconstruction'
        else:
                self.RECODefaultSeq='reconstruction_fromRECO'

	self.EIDefaultSeq='top'
        self.POSTRECODefaultSeq=None
        self.L1HwValDefaultSeq='L1HwVal'
        self.DQMDefaultSeq='DQMOffline'
        self.VALIDATIONDefaultSeq=''
        self.ENDJOBDefaultSeq='endOfProcess'
        self.REPACKDefaultSeq='DigiToRawRepack'
	self.PATDefaultSeq='miniAOD'

        self.EVTCONTDefaultCFF="Configuration/EventContent/EventContent_cff"

	if not self._options.beamspot:
		self._options.beamspot=VtxSmearedDefaultKey
		
        # if its MC then change the raw2digi
        if self._options.isMC==True:
                self.RAW2DIGIDefaultCFF="Configuration/StandardSequences/RawToDigi_cff"
		self.RECODefaultCFF="Configuration/StandardSequences/Reconstruction_cff"
		self.PATDefaultCFF="Configuration/StandardSequences/PATMC_cff"
                self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineMC_cff"
                self.ALCADefaultCFF="Configuration/StandardSequences/AlCaRecoStreamsMC_cff"
	else:
		self._options.beamspot = None
	
	#patch for gen, due to backward incompatibility
	if 'reGEN' in self.stepMap:
		self.GENDefaultSeq='fixGenInfo'

        if self._options.scenario=='cosmics':
	    self._options.pileup='Cosmics'	
            self.DIGIDefaultCFF="Configuration/StandardSequences/DigiCosmics_cff"
            self.RECODefaultCFF="Configuration/StandardSequences/ReconstructionCosmics_cff"
	    self.SKIMDefaultCFF="Configuration/StandardSequences/SkimsCosmics_cff"
            self.EVTCONTDefaultCFF="Configuration/EventContent/EventContentCosmics_cff"
            self.VALIDATIONDefaultCFF="Configuration/StandardSequences/ValidationCosmics_cff"
            self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineCosmics_cff"
            if self._options.isMC==True:
                self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineCosmicsMC_cff"
            self.HARVESTINGDefaultCFF="Configuration/StandardSequences/HarvestingCosmics_cff"
            self.RECODefaultSeq='reconstructionCosmics'
            self.DQMDefaultSeq='DQMOfflineCosmics'

        if self._options.scenario=='HeavyIons':
	    if not self._options.beamspot:
		    self._options.beamspot=VtxSmearedHIDefaultKey
            self.HLTDefaultSeq = 'HIon'
            self.VALIDATIONDefaultCFF="Configuration/StandardSequences/ValidationHeavyIons_cff"
            self.VALIDATIONDefaultSeq=''
            self.EVTCONTDefaultCFF="Configuration/EventContent/EventContentHeavyIons_cff"
            self.RECODefaultCFF="Configuration/StandardSequences/ReconstructionHeavyIons_cff"
            self.RECODefaultSeq='reconstructionHeavyIons'
            self.ALCADefaultCFF = "Configuration/StandardSequences/AlCaRecoStreamsHeavyIons_cff"
            self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineHeavyIons_cff"
            self.DQMDefaultSeq='DQMOfflineHeavyIons'
            self.SKIMDefaultCFF="Configuration/StandardSequences/SkimsHeavyIons_cff"
            self.HARVESTINGDefaultCFF="Configuration/StandardSequences/HarvestingHeavyIons_cff"
            if self._options.isMC==True:
                    self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineHeavyIonsMC_cff"


        self.RAW2RECODefaultSeq=','.join([self.RAW2DIGIDefaultSeq,self.RECODefaultSeq])

	self.USERDefaultSeq='user'
	self.USERDefaultCFF=None

        # the magnetic field
	if self._options.isData:
		if self._options.magField==defaultOptions.magField:
			print "magnetic field option forced to: AutoFromDBCurrent"
		self._options.magField='AutoFromDBCurrent'
        self.magFieldCFF = 'Configuration/StandardSequences/MagneticField_'+self._options.magField.replace('.','')+'_cff'
        self.magFieldCFF = self.magFieldCFF.replace("__",'_')

        # the geometry
	self.GeometryCFF='Configuration/StandardSequences/GeometryRecoDB_cff'
	self.geometryDBLabel=None
	simGeometry=''
        if self._options.fast:
                if 'start' in self._options.conditions.lower():
                        self.GeometryCFF='FastSimulation/Configuration/Geometries_START_cff'
                else:
                        self.GeometryCFF='FastSimulation/Configuration/Geometries_MC_cff'
        else:
		def inGeometryKeys(opt):
			from Configuration.StandardSequences.GeometryConf import GeometryConf
			if opt in GeometryConf:
				return GeometryConf[opt]
			else:
				return opt

		geoms=self._options.geometry.split(',')
		if len(geoms)==1: geoms=inGeometryKeys(geoms[0]).split(',')
		if len(geoms)==2:
			#may specify the reco geometry
			if '/' in geoms[1] or '_cff' in geoms[1]:
				self.GeometryCFF=geoms[1]
			else:
				self.GeometryCFF='Configuration/Geometry/Geometry'+geoms[1]+'_cff'

		if (geoms[0].startswith('DB:')):
			self.SimGeometryCFF='Configuration/StandardSequences/GeometrySimDB_cff'
			self.geometryDBLabel=geoms[0][3:]
			print "with DB:"
		else:
			if '/' in geoms[0] or '_cff' in geoms[0]:
								self.SimGeometryCFF=geoms[0]
			else:
				simGeometry=geoms[0]
				if self._options.gflash==True:
					self.SimGeometryCFF='Configuration/Geometry/Geometry'+geoms[0]+'GFlash_cff'
				else:
					self.SimGeometryCFF='Configuration/Geometry/Geometry'+geoms[0]+'_cff'

	# synchronize the geometry configuration and the FullSimulation sequence to be used
        if simGeometry not in defaultOptions.geometryExtendedOptions:
		self.SIMDefaultCFF="Configuration/StandardSequences/SimIdeal_cff"

	if self._options.scenario=='nocoll' or self._options.scenario=='cosmics':
            self.SIMDefaultCFF="Configuration/StandardSequences/SimNOBEAM_cff"
            self._options.beamspot='NoSmear'

        # fastsim requires some changes to the default cff files and sequences
	if self._options.fast:
		self.SIMDefaultCFF = 'FastSimulation.Configuration.SimIdeal_cff'
		self.RECODefaultCFF= 'FastSimulation.Configuration.Reconstruction_AftMix_cff'
		self.RECOBEFMIXDefaultCFF = 'FastSimulation.Configuration.Reconstruction_BefMix_cff'
		self.RECOBEFMIXDefaultSeq = 'reconstruction_befmix'
		self.L1RecoDefaultCFF='FastSimulation.Configuration.L1Reco_cff'
                self.DQMOFFLINEDefaultCFF="FastSimulation.Configuration.DQMOfflineMC_cff"

        # Mixing
	if self._options.pileup=='default':
		from Configuration.StandardSequences.Mixing import MixingDefaultKey
		self._options.pileup=MixingDefaultKey
		

	#not driven by a default cff anymore
	if self._options.isData:
		self._options.pileup=None

	if self._options.slhc:
		self.GeometryCFF='SLHCUpgradeSimulations.Geometry.%s_cmsSimIdealGeometryXML_cff'%(self._options.slhc,)
		if 'stdgeom' not in self._options.slhc:
			self.SimGeometryCFF='SLHCUpgradeSimulations.Geometry.%s_cmsSimIdealGeometryXML_cff'%(self._options.slhc,)
		self.DIGIDefaultCFF='SLHCUpgradeSimulations/Geometry/Digi_%s_cff'%(self._options.slhc,)
		if self._options.pileup!=defaultOptions.pileup:
			self._options.pileup='SLHC_%s_%s'%(self._options.pileup,self._options.slhc)

	self.REDIGIDefaultSeq=self.DIGIDefaultSeq

    # for alca, skims, etc
    def addExtraStream(self,name,stream,workflow='full'):
            # define output module and go from there
            output = cms.OutputModule("PoolOutputModule")
            if stream.selectEvents.parameters_().__len__()!=0:
                    output.SelectEvents = stream.selectEvents
            else:
                    output.SelectEvents = cms.untracked.PSet()
                    output.SelectEvents.SelectEvents=cms.vstring()
                    if isinstance(stream.paths,tuple):
                            for path in stream.paths:
                                    output.SelectEvents.SelectEvents.append(path.label())
                    else:
                            output.SelectEvents.SelectEvents.append(stream.paths.label())



            if isinstance(stream.content,str):
		    evtPset=getattr(self.process,stream.content)
		    for p in evtPset.parameters_():
			    setattr(output,p,getattr(evtPset,p))
                    if not self._options.inlineEventContent:
                            def doNotInlineEventContent(instance,label = "process."+stream.content+".outputCommands"):
                                    return label
                            output.outputCommands.__dict__["dumpPython"] = doNotInlineEventContent
            else:
                    output.outputCommands = stream.content


            output.fileName = cms.untracked.string(self._options.dirout+stream.name+'.root')

            output.dataset  = cms.untracked.PSet( dataTier = stream.dataTier,
                                                  filterName = cms.untracked.string(stream.name))

            if self._options.filtername:
                    output.dataset.filterName= cms.untracked.string(self._options.filtername+"_"+stream.name)

	    #add an automatic flushing to limit memory consumption
	    output.eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)

            if workflow in ("producers,full"):
                    if isinstance(stream.paths,tuple):
                            for path in stream.paths:
                                    self.schedule.append(path)
                    else:
                            self.schedule.append(stream.paths)


	    # in case of relvals we don't want to have additional outputs
            if (not self._options.relval) and workflow in ("full","output"):
                    self.additionalOutputs[name] = output
		    setattr(self.process,name,output)
	    
            if workflow == 'output':
                    # adjust the select events to the proper trigger results from previous process
                    filterList = output.SelectEvents.SelectEvents
                    for i, filter in enumerate(filterList):
                            filterList[i] = filter+":"+self._options.triggerResultsProcess

	    return output
    
    #----------------------------------------------------------------------------
    # here the methods to create the steps. Of course we are doing magic here ;)
    # prepare_STEPNAME modifies self.process and what else's needed.
    #----------------------------------------------------------------------------

    def loadDefaultOrSpecifiedCFF(self, sequence,defaultCFF,unsch=0):
            if ( len(sequence.split('.'))==1 ):
                    l=self.loadAndRemember(defaultCFF,unsch)
            elif ( len(sequence.split('.'))==2 ):
                    l=self.loadAndRemember(sequence.split('.')[0],unsch)
                    sequence=sequence.split('.')[1]
            else:
                    print "sub sequence configuration must be of the form dir/subdir/cff.a+b+c or cff.a"
                    print sequence,"not recognized"
                    raise
            return l

    def scheduleSequence(self,seq,prefix,what='Path'):
	    if '*' in seq:
		    #create only one path with all sequences in it
		    for i,s in enumerate(seq.split('*')):
			    if i==0:
				    setattr(self.process,prefix,getattr(cms,what)( getattr(self.process, s) ))
			    else:
				    p=getattr(self.process,prefix)
				    p+=getattr(self.process, s)
		    self.schedule.append(getattr(self.process,prefix))
		    return
	    else:
		    #create as many path as many sequences
		    if not '+' in seq:
			    if self.nextScheduleIsConditional:
				    self.conditionalPaths.append(prefix)
			    setattr(self.process,prefix,getattr(cms,what)( getattr(self.process, seq) ))
			    self.schedule.append(getattr(self.process,prefix))
		    else:
			    for i,s in enumerate(seq.split('+')):
				    sn=prefix+'%d'%(i)
				    setattr(self.process,sn,getattr(cms,what)( getattr(self.process, s) ))
				    self.schedule.append(getattr(self.process,sn))
		    return
    
    def scheduleSequenceAtEnd(self,seq,prefix):
	    self.scheduleSequence(seq,prefix,what='EndPath')
	    return
	    
    def prepare_ALCAPRODUCER(self, sequence = None):
        self.prepare_ALCA(sequence, workflow = "producers")

    def prepare_ALCAOUTPUT(self, sequence = None):
        self.prepare_ALCA(sequence, workflow = "output")

    def prepare_ALCA(self, sequence = None, workflow = 'full'):
        """ Enrich the process with alca streams """
	print 'DL enriching',workflow,sequence
        alcaConfig=self.loadDefaultOrSpecifiedCFF(sequence,self.ALCADefaultCFF)
        sequence = sequence.split('.')[-1]

        # decide which ALCA paths to use
        alcaList = sequence.split("+")
	maxLevel=0
	from Configuration.AlCa.autoAlca import autoAlca
	# support @X from autoAlca.py, and recursion support: i.e T0:@Mu+@EG+...
	self.expandMapping(alcaList,autoAlca)
	self.AlCaPaths=[]
        for name in alcaConfig.__dict__:
            alcastream = getattr(alcaConfig,name)
            shortName = name.replace('ALCARECOStream','')
            if shortName in alcaList and isinstance(alcastream,cms.FilteredStream):
	        output = self.addExtraStream(name,alcastream, workflow = workflow)
		self.executeAndRemember('process.ALCARECOEventContent.outputCommands.extend(process.OutALCARECO'+shortName+'_noDrop.outputCommands)')
		self.AlCaPaths.append(shortName)
		if 'DQM' in alcaList:
			if not self._options.inlineEventContent and hasattr(self.process,name):
				self.executeAndRemember('process.' + name + '.outputCommands.append("keep *_MEtoEDMConverter_*_*")')
			else:
				output.outputCommands.append("keep *_MEtoEDMConverter_*_*")
			
                #rename the HLT process name in the alca modules
                if self._options.hltProcess or 'HLT' in self.stepMap:
                        if isinstance(alcastream.paths,tuple):
                                for path in alcastream.paths:
                                        self.renameHLTprocessInSequence(path.label())
                        else:
                                self.renameHLTprocessInSequence(alcastream.paths.label())
				
                for i in range(alcaList.count(shortName)):
                        alcaList.remove(shortName)

            # DQM needs a special handling
            elif name == 'pathALCARECODQM' and 'DQM' in alcaList:
                    path = getattr(alcaConfig,name)
                    self.schedule.append(path)
                    alcaList.remove('DQM')

            if isinstance(alcastream,cms.Path):
                    #black list the alca path so that they do not appear in the cfg
                    self.blacklist_paths.append(alcastream)


        if len(alcaList) != 0:
                available=[]
                for name in alcaConfig.__dict__:
                        alcastream = getattr(alcaConfig,name)
                        if isinstance(alcastream,cms.FilteredStream):
                                available.append(name.replace('ALCARECOStream',''))
                print "The following alcas could not be found "+str(alcaList)
                print "available ",available
                #print "verify your configuration, ignoring for now"
                raise Exception("The following alcas could not be found "+str(alcaList))

    def prepare_LHE(self, sequence = None):
	    #load the fragment
	    ##make it loadable
	    loadFragment = self._options.evt_type.replace('.py','',).replace('.','_').replace('python/','').replace('/','.')
	    print "Loading lhe fragment from",loadFragment
	    __import__(loadFragment)
	    self.process.load(loadFragment)
	    ##inline the modules
	    self._options.inlineObjets+=','+sequence
	    
	    getattr(self.process,sequence).nEvents = int(self._options.number)
	    
	    #schedule it
	    self.process.lhe_step = cms.Path( getattr( self.process,sequence)  )
	    self.excludedPaths.append("lhe_step")
	    self.schedule.append( self.process.lhe_step )

    def prepare_GEN(self, sequence = None):
        """ load the fragment of generator configuration """
	loadFailure=False
        #remove trailing .py
        #support old style .cfi by changing into something.cfi into something_cfi
        #remove python/ from the name
        loadFragment = self._options.evt_type.replace('.py','',).replace('.','_').replace('python/','')
        #standard location of fragments
        if not '/' in loadFragment:
                loadFragment='Configuration.Generator.'+loadFragment
        else:
                loadFragment=loadFragment.replace('/','.')
	try:
		print "Loading generator fragment from",loadFragment
		__import__(loadFragment)
	except:
		loadFailure=True
		#if self.process.source and self.process.source.type_()=='EmptySource':
		if not (self._options.filein or self._options.dasquery):
			raise Exception("Neither gen fragment of input files provided: this is an inconsistent GEN step configuration")
			
	if not loadFailure:
		generatorModule=sys.modules[loadFragment]
		genModules=generatorModule.__dict__
		#remove lhe producer module since this should have been
		#imported instead in the LHE step
		if self.LHEDefaultSeq in genModules:
                        del genModules[self.LHEDefaultSeq]

		if self._options.hideGen:
			self.loadAndRemember(loadFragment)
		else:
			self.process.load(loadFragment)
			# expose the objects from that fragment to the configuration
			import FWCore.ParameterSet.Modules as cmstypes
			for name in genModules:
				theObject = getattr(generatorModule,name)
				if isinstance(theObject, cmstypes._Module):
					self._options.inlineObjets=name+','+self._options.inlineObjets
				elif isinstance(theObject, cms.Sequence) or isinstance(theObject, cmstypes.ESProducer):
					self._options.inlineObjets+=','+name

		if sequence == self.GENDefaultSeq or sequence == 'pgen_genonly':
			if 'ProductionFilterSequence' in genModules and ('generator' in genModules):
				self.productionFilterSequence = 'ProductionFilterSequence'
			elif 'generator' in genModules:
				self.productionFilterSequence = 'generator'

        """ Enrich the schedule with the rest of the generation step """
        self.loadDefaultOrSpecifiedCFF(sequence,self.GENDefaultCFF)
        genSeqName=sequence.split('.')[-1]

	if True:
                try:
			from Configuration.StandardSequences.VtxSmeared import VtxSmeared
			cffToBeLoaded=VtxSmeared[self._options.beamspot]
			self.loadAndRemember(cffToBeLoaded)
                except ImportError:
                        raise Exception("VertexSmearing type or beamspot "+self._options.beamspot+" unknown.")

                if self._options.scenario == 'HeavyIons': 
			if self._options.pileup=='HiMixGEN':
				self.loadAndRemember("Configuration/StandardSequences/GeneratorMix_cff")
			else:
				self.loadAndRemember("Configuration/StandardSequences/GeneratorHI_cff")

        self.process.generation_step = cms.Path( getattr(self.process,genSeqName) )
        self.schedule.append(self.process.generation_step)

	#register to the genstepfilter the name of the path (static right now, but might evolve)
	self.executeAndRemember('process.genstepfilter.triggerConditions=cms.vstring("generation_step")')
	
	if 'reGEN' in self.stepMap:
		#stop here
		return 

        """ Enrich the schedule with the summary of the filter step """
        #the gen filter in the endpath
        self.loadAndRemember("GeneratorInterface/Core/genFilterSummary_cff")
	self.scheduleSequenceAtEnd('genFilterSummary','genfiltersummary_step')
        return

    def prepare_SIM(self, sequence = None):
        """ Enrich the schedule with the simulation step"""
	self.loadDefaultOrSpecifiedCFF(sequence,self.SIMDefaultCFF)
	if not self._options.fast:
		if self._options.gflash==True:
			self.loadAndRemember("Configuration/StandardSequences/GFlashSIM_cff")

		if self._options.magField=='0T':
			self.executeAndRemember("process.g4SimHits.UseMagneticField = cms.bool(False)")
	else:
		if self._options.magField=='0T':
			self.executeAndRemember("process.famosSimHits.UseMagneticField = cms.bool(False)")

	self.scheduleSequence(sequence.split('.')[-1],'simulation_step')
        return

    def prepare_DIGI(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
        self.loadDefaultOrSpecifiedCFF(sequence,self.DIGIDefaultCFF)

        if self._options.gflash==True:
                self.loadAndRemember("Configuration/StandardSequences/GFlashDIGI_cff")

        if sequence == 'pdigi_valid':
            self.executeAndRemember("process.mix.digitizers = cms.PSet(process.theDigitizersValid)")

	if sequence != 'pdigi_nogen' and sequence != 'pdigi_valid_nogen' and not self.process.source.type_()=='EmptySource':
		if self._options.inputEventContent=='':
			self._options.inputEventContent='REGEN'
		else:
			self._options.inputEventContent=self._options.inputEventContent+',REGEN'


	self.scheduleSequence(sequence.split('.')[-1],'digitisation_step')
        return

    def prepare_DIGIPREMIX(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
        self.loadDefaultOrSpecifiedCFF(sequence,self.DIGIDefaultCFF)

	self.loadAndRemember("SimGeneral/MixingModule/digi_noNoise_cfi")

        if sequence == 'pdigi_valid':
		self.executeAndRemember("process.mix.digitizers = cms.PSet(process.theDigitizersNoNoiseValid)")
	else:
		self.executeAndRemember("process.mix.digitizers = cms.PSet(process.theDigitizersNoNoise)")

	self.scheduleSequence(sequence.split('.')[-1],'digitisation_step')
        return

    def prepare_DIGIPREMIX_S2(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
	self.loadDefaultOrSpecifiedCFF(sequence,self.DIGIDefaultCFF)

	self.loadAndRemember("SimGeneral/MixingModule/digi_MixPreMix_cfi")


        if sequence == 'pdigi_valid':
            self.executeAndRemember("process.mix.digitizers = cms.PSet(process.theDigitizersMixPreMixValid)")
	else:
	    self.executeAndRemember("process.mix.digitizers = cms.PSet(process.theDigitizersMixPreMix)")

	self.scheduleSequence(sequence.split('.')[-1],'digitisation_step')
        return

    def prepare_CFWRITER(self, sequence = None):
	    """ Enrich the schedule with the crossing frame writer step"""
	    self.loadAndRemember(self.CFWRITERDefaultCFF)
	    self.scheduleSequence('pcfw','cfwriter_step')
	    return

    def prepare_DATAMIX(self, sequence = None):
	    """ Enrich the schedule with the digitisation step"""
	    self.loadAndRemember(self.DATAMIXDefaultCFF)
	    self.scheduleSequence('pdatamix','datamixing_step')

	    if self._options.pileup_input:
		    theFiles=''
		    if self._options.pileup_input.startswith('dbs:') or self._options.pileup_input.startswith('das:'):
			    theFiles=filesFromDASQuery('file dataset = %s'%(self._options.pileup_input[4:],),self._options.pileup_dasoption)[0]
		    elif self._options.pileup_input.startswith("filelist:"):
			    theFiles= (filesFromList(self._options.pileup_input[9:]))[0]
		    else:
			    theFiles=self._options.pileup_input.split(',')
		    #print theFiles
		    self.executeAndRemember( "process.mixData.input.fileNames = cms.untracked.vstring(%s)"%(  theFiles ) )

	    return

    def prepare_DIGI2RAW(self, sequence = None):
            self.loadDefaultOrSpecifiedCFF(sequence,self.DIGI2RAWDefaultCFF)
	    self.scheduleSequence(sequence.split('.')[-1],'digi2raw_step')
	    if "DIGIPREMIX" in self.stepMap.keys():
		    self.executeAndRemember("process.esDigiToRaw.Label = cms.string('mix')")  ##terrible hack - bypass zero suppression
		    self.executeAndRemember("process.SiStripDigiToRaw.FedReadoutMode = cms.string('PREMIX_RAW')") ##special readout mode for StripTracker

            return

    def prepare_REPACK(self, sequence = None):
            self.loadDefaultOrSpecifiedCFF(sequence,self.REPACKDefaultCFF)
	    self.scheduleSequence(sequence.split('.')[-1],'digi2repack_step')
            return

    def prepare_L1(self, sequence = None):
	    """ Enrich the schedule with the L1 simulation step"""
            assert(sequence == None)
	    self.loadAndRemember(self.L1EMDefaultCFF)
	    self.scheduleSequence('SimL1Emulator','L1simulation_step')
	    return

    def prepare_L1REPACK(self, sequence = None):
            """ Enrich the schedule with the L1 simulation step, running the L1 emulator on data unpacked from the RAW collection, and repacking the result in a new RAW collection"""
	    supported = ['GT','GT1','GT2','GCTGT','Full','Full2015Data','uGT']
            if sequence in supported:
                self.loadAndRemember('Configuration/StandardSequences/SimL1EmulatorRepack_%s_cff'%sequence)
		if self._options.scenario == 'HeavyIons':
			self.renameInputTagsInSequence("SimL1Emulator","rawDataCollector","rawDataRepacker")
		self.scheduleSequence('SimL1Emulator','L1RePack_step')
	    else:
                print "L1REPACK with '",sequence,"' is not supported! Supported choices are: ",supported
                raise Exception('unsupported feature')


    def prepare_HLT(self, sequence = None):
        """ Enrich the schedule with the HLT simulation step"""
        if not sequence:
                print "no specification of the hlt menu has been given, should never happen"
                raise  Exception('no HLT sequence provided')

        if '@' in sequence:
                # case where HLT:@something was provided
                from Configuration.HLT.autoHLT import autoHLT
                key = sequence[1:]
                if key in autoHLT:
                  sequence = autoHLT[key]
                else:
                  raise ValueError('no HLT mapping key "%s" found in autoHLT' % key)

        if ',' in sequence:
                #case where HLT:something:something was provided
                self.executeAndRemember('import HLTrigger.Configuration.Utilities')
                optionsForHLT = {}
                if self._options.scenario == 'HeavyIons':
                  optionsForHLT['type'] = 'HIon'
                else:
                  optionsForHLT['type'] = 'GRun'
                optionsForHLTConfig = ', '.join('%s=%s' % (key, repr(val)) for (key, val) in optionsForHLT.iteritems())
		if sequence == 'run,fromSource':
			if hasattr(self.process.source,'firstRun'):
				self.executeAndRemember('process.loadHltConfiguration("run:%%d"%%(process.source.firstRun.value()),%s)'%(optionsForHLTConfig))
			elif hasattr(self.process.source,'setRunNumber'):
				self.executeAndRemember('process.loadHltConfiguration("run:%%d"%%(process.source.setRunNumber.value()),%s)'%(optionsForHLTConfig))
			else:
				raise Exception('Cannot replace menu to load %s'%(sequence))
		else:
			self.executeAndRemember('process.loadHltConfiguration("%s",%s)'%(sequence.replace(',',':'),optionsForHLTConfig))
        else:
		self.loadAndRemember('HLTrigger/Configuration/HLT_%s_cff'       % sequence)

        if self._options.isMC:
		if self._options.fast:
			self._options.customisation_file.append("HLTrigger/Configuration/customizeHLTforMC.customizeHLTforFastSim")
		else:
			self._options.customisation_file.append("HLTrigger/Configuration/customizeHLTforMC.customizeHLTforFullSim")

	if self._options.name != 'HLT':
		self.additionalCommands.append('from HLTrigger.Configuration.CustomConfigs import ProcessName')
		self.additionalCommands.append('process = ProcessName(process)')
                self.additionalCommands.append('')
		from HLTrigger.Configuration.CustomConfigs import ProcessName
		self.process = ProcessName(self.process)
		
        self.schedule.append(self.process.HLTSchedule)
        [self.blacklist_paths.append(path) for path in self.process.HLTSchedule if isinstance(path,(cms.Path,cms.EndPath))]

	#this is a fake, to be removed with fastim migration and HLT menu dump
	if self._options.fast:
		if not hasattr(self.process,'HLTEndSequence'):
			self.executeAndRemember("process.HLTEndSequence = cms.Sequence( process.dummyModule )")
		

    def prepare_RAW2RECO(self, sequence = None):
            if ','in sequence:
                    seqReco=sequence.split(',')[1]
                    seqDigi=sequence.split(',')[0]
            else:
                    print "RAW2RECO requires two specifications",sequence,"insufficient"

	    self.prepare_RAW2DIGI(seqDigi)
	    self.prepare_RECO(seqReco)
            return

    def prepare_RAW2DIGI(self, sequence = "RawToDigi"):
            self.loadDefaultOrSpecifiedCFF(sequence,self.RAW2DIGIDefaultCFF)
	    self.scheduleSequence(sequence,'raw2digi_step')
	    #	    if self._options.isRepacked:
	    #self.renameInputTagsInSequence(sequence)
            return

    def prepare_PATFILTER(self, sequence=None):
            self.loadAndRemember("PhysicsTools/PatAlgos/slimming/metFilterPaths_cff")
	    from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff import allMetFilterPaths
	    for filt in allMetFilterPaths:
		    self.schedule.append(getattr(self.process,'Flag_'+filt))

    def prepare_L1HwVal(self, sequence = 'L1HwVal'):
        ''' Enrich the schedule with L1 HW validation '''
        self.loadDefaultOrSpecifiedCFF(sequence,self.L1HwValDefaultCFF)
	#self.scheduleSequence(sequence.split('.')[-1],'l1hwval_step')
	print '\n\n\n DEPRECATED this has no action \n\n\n'
        return

    def prepare_L1Reco(self, sequence = "L1Reco"):
        ''' Enrich the schedule with L1 reconstruction '''
        self.loadDefaultOrSpecifiedCFF(sequence,self.L1RecoDefaultCFF)
	self.scheduleSequence(sequence.split('.')[-1],'L1Reco_step')
        return

    def prepare_L1TrackTrigger(self, sequence = "L1TrackTrigger"):
        ''' Enrich the schedule with L1 reconstruction '''
        self.loadDefaultOrSpecifiedCFF(sequence,self.L1TrackTriggerDefaultCFF)
	self.scheduleSequence(sequence.split('.')[-1],'L1TrackTrigger_step')
        return

    def prepare_FILTER(self, sequence = None):
        ''' Enrich the schedule with a user defined filter sequence '''
	## load the relevant part
	filterConfig=self.load(sequence.split('.')[0])
	filterSeq=sequence.split('.')[-1]
	## print it in the configuration
	class PrintAllModules(object):
		def __init__(self):
			self.inliner=''
			pass
		def enter(self,visitee):
			try:
				label=visitee.label()
				##needs to be in reverse order
				self.inliner=label+','+self.inliner
			except:
				pass
		def leave(self,v): pass

	expander=PrintAllModules()
	getattr(self.process,filterSeq).visit( expander )
	self._options.inlineObjets+=','+expander.inliner
	self._options.inlineObjets+=','+filterSeq
		
	## put the filtering path in the schedule
	self.scheduleSequence(filterSeq,'filtering_step')
	self.nextScheduleIsConditional=True
	## put it before all the other paths
	self.productionFilterSequence = filterSeq 
	
	return

    def prepare_RECO(self, sequence = "reconstruction"):
        ''' Enrich the schedule with reconstruction '''
        self.loadDefaultOrSpecifiedCFF(sequence,self.RECODefaultCFF)
	self.scheduleSequence(sequence.split('.')[-1],'reconstruction_step')
        return

    def prepare_RECOBEFMIX(self, sequence = "reconstruction"):
        ''' Enrich the schedule with the part of reconstruction that is done before mixing in FastSim'''
        if not self._options.fast:
                print "ERROR: this step is only implemented for FastSim"
                sys.exit()
        self.loadDefaultOrSpecifiedCFF(self.RECOBEFMIXDefaultSeq,self.RECOBEFMIXDefaultCFF)
        self.scheduleSequence(sequence.split('.')[-1],'reconstruction_befmix_step')
        return

    def prepare_PAT(self, sequence = "miniAOD"):
        ''' Enrich the schedule with PAT '''
	self.prepare_PATFILTER(self)
        self.loadDefaultOrSpecifiedCFF(sequence,self.PATDefaultCFF,1) #this is unscheduled
	if not self._options.runUnscheduled:	
		raise Exception("MiniAOD production can only run in unscheduled mode, please run cmsDriver with --runUnscheduled")
        if self._options.isData:
            self._options.customisation_file_unsch.append("PhysicsTools/PatAlgos/slimming/miniAOD_tools.miniAOD_customizeAllData")
        else:
            self._options.customisation_file_unsch.append("PhysicsTools/PatAlgos/slimming/miniAOD_tools.miniAOD_customizeAllMC")
            if self._options.fast:
                self._options.customisation_file_unsch.append("PhysicsTools/PatAlgos/slimming/metFilterPaths_cff.miniAOD_customizeMETFiltersFastSim")
        return

    def prepare_EI(self, sequence = None):
        ''' Enrich the schedule with event interpretation '''
	from Configuration.StandardSequences.EventInterpretation import EventInterpretation
	if sequence in EventInterpretation:
		self.EIDefaultCFF = EventInterpretation[sequence]
		sequence = 'EIsequence'
	else:
		raise Exception('Cannot set %s event interpretation'%( sequence) )
        self.loadDefaultOrSpecifiedCFF(sequence,self.EIDefaultCFF)
	self.scheduleSequence(sequence.split('.')[-1],'eventinterpretaion_step')
        return

    def prepare_SKIM(self, sequence = "all"):
        ''' Enrich the schedule with skimming fragments'''
        skimConfig = self.loadDefaultOrSpecifiedCFF(sequence,self.SKIMDefaultCFF)
        sequence = sequence.split('.')[-1]

        skimlist=sequence.split('+')
        ## support @Mu+DiJet+@Electron configuration via autoSkim.py
	from Configuration.Skimming.autoSkim import autoSkim
	self.expandMapping(skimlist,autoSkim)

        #print "dictionnary for skims:",skimConfig.__dict__
        for skim in skimConfig.__dict__:
                skimstream = getattr(skimConfig,skim)
                if isinstance(skimstream,cms.Path):
                    #black list the alca path so that they do not appear in the cfg
                    self.blacklist_paths.append(skimstream)
                if (not isinstance(skimstream,cms.FilteredStream)):
                        continue
                shortname = skim.replace('SKIMStream','')
                if (sequence=="all"):
                        self.addExtraStream(skim,skimstream)
                elif (shortname in skimlist):
                        self.addExtraStream(skim,skimstream)
                        #add a DQM eventcontent for this guy
			if self._options.datatier=='DQM':
                                self.process.load(self.EVTCONTDefaultCFF)
                                skimstreamDQM = cms.FilteredStream(
                                        responsible = skimstream.responsible,
                                        name = skimstream.name+'DQM',
                                        paths = skimstream.paths,
                                        selectEvents = skimstream.selectEvents,
                                        content = self._options.datatier+'EventContent',
                                        dataTier = cms.untracked.string(self._options.datatier)
                                        )
                                self.addExtraStream(skim+'DQM',skimstreamDQM)
                        for i in range(skimlist.count(shortname)):
                                skimlist.remove(shortname)



        if (skimlist.__len__()!=0 and sequence!="all"):
                print 'WARNING, possible typo with SKIM:'+'+'.join(skimlist)
                raise Exception('WARNING, possible typo with SKIM:'+'+'.join(skimlist))

    def prepare_USER(self, sequence = None):
        ''' Enrich the schedule with a user defined sequence '''
        self.loadDefaultOrSpecifiedCFF(sequence,self.USERDefaultCFF)
	self.scheduleSequence(sequence.split('.')[-1],'user_step')
        return

    def prepare_POSTRECO(self, sequence = None):
        """ Enrich the schedule with the postreco step """
        self.loadAndRemember(self.POSTRECODefaultCFF)
	self.scheduleSequence('postreco_generator','postreco_step')
        return


    def prepare_VALIDATION(self, sequence = 'validation'):
	    print sequence,"in preparing validation"
            self.loadDefaultOrSpecifiedCFF(sequence,self.VALIDATIONDefaultCFF)
	    from Validation.Configuration.autoValidation import autoValidation
            #in case VALIDATION:something:somethingelse -> something,somethingelse
            sequence=sequence.split('.')[-1]
            if sequence.find(',')!=-1:
                    prevalSeqName=sequence.split(',')[0].split('+')
                    valSeqName=sequence.split(',')[1].split('+')
		    self.expandMapping(prevalSeqName,autoValidation,index=0)
		    self.expandMapping(valSeqName,autoValidation,index=1)
            else:
		    if '@' in sequence:
			    prevalSeqName=sequence.split('+')
			    valSeqName=sequence.split('+')
			    self.expandMapping(prevalSeqName,autoValidation,index=0)
			    self.expandMapping(valSeqName,autoValidation,index=1)
		    else:
			    postfix=''
			    if sequence:
				    postfix='_'+sequence
			    prevalSeqName=['prevalidation'+postfix]
			    valSeqName=['validation'+postfix]
			    if not hasattr(self.process,valSeqName[0]):
				    prevalSeqName=['']
				    valSeqName=[sequence]

	    def NFI(index):
		    ##name from index, required to keep backward compatibility
		    if index==0:
			    return ''
		    else:
			    return '%s'%index


            #rename the HLT process in validation steps
	    if ('HLT' in self.stepMap and not self._options.fast) or self._options.hltProcess:
		    for s in valSeqName+prevalSeqName:
			    if s:
				    self.renameHLTprocessInSequence(s)
	    for (i,s) in enumerate(prevalSeqName):
		    if s:
			    setattr(self.process,'prevalidation_step%s'%NFI(i),  cms.Path( getattr(self.process, s)) )
			    self.schedule.append(getattr(self.process,'prevalidation_step%s'%NFI(i)))
			    
	    for (i,s) in enumerate(valSeqName):
		    setattr(self.process,'validation_step%s'%NFI(i), cms.EndPath( getattr(self.process, s)))
		    self.schedule.append(getattr(self.process,'validation_step%s'%NFI(i)))

            #needed in case the miniAODValidation sequence is run starting from AODSIM
	    if 'PAT' in self.stepMap and not 'RECO' in self.stepMap:
		    return

            if not 'DIGI' in self.stepMap and not self._options.fast and not any(map( lambda s : s.startswith('genvalid'), valSeqName)):
		    if self._options.restoreRNDSeeds==False and not self._options.restoreRNDSeeds==True:
			    self._options.restoreRNDSeeds=True

	    if not 'DIGI' in self.stepMap and not self._options.fast:
		    self.executeAndRemember("process.mix.playback = True")
		    self.executeAndRemember("process.mix.digitizers = cms.PSet()")
                    self.executeAndRemember("for a in process.aliases: delattr(process, a)")
                    self._options.customisation_file.append("SimGeneral/MixingModule/fullMixCustomize_cff.setCrossingFrameOn")

	    if hasattr(self.process,"genstepfilter") and len(self.process.genstepfilter.triggerConditions):
		    #will get in the schedule, smoothly
		    for (i,s) in enumerate(valSeqName):
			    getattr(self.process,'validation_step%s'%NFI(i))._seq = self.process.genstepfilter * getattr(self.process,'validation_step%s'%NFI(i))._seq

            return


    class MassSearchReplaceProcessNameVisitor(object):
            """Visitor that travels within a cms.Sequence, looks for a parameter and replace its value
            It will climb down within PSets, VPSets and VInputTags to find its target"""
            def __init__(self, paramSearch, paramReplace, verbose=False, whitelist=()):
                    self._paramReplace = paramReplace
                    self._paramSearch = paramSearch
                    self._verbose = verbose
                    self._whitelist = whitelist

            def doIt(self,pset,base):
                    if isinstance(pset, cms._Parameterizable):
                            for name in pset.parameters_().keys():
                                    # skip whitelisted parameters
                                    if name in self._whitelist:
                                        continue
                                    # if I use pset.parameters_().items() I get copies of the parameter values
                                    # so I can't modify the nested pset
                                    value = getattr(pset,name)
                                    type = value.pythonTypeName()
                                    if type in ('cms.PSet', 'cms.untracked.PSet'):
                                        self.doIt(value,base+"."+name)
                                    elif type in ('cms.VPSet', 'cms.untracked.VPSet'):
                                        for (i,ps) in enumerate(value): self.doIt(ps, "%s.%s[%d]"%(base,name,i) )
                                    elif type in ('cms.string', 'cms.untracked.string'):
                                        if value.value() == self._paramSearch:
                                            if self._verbose: print "set string process name %s.%s %s ==> %s"% (base, name, value, self._paramReplace)
                                            setattr(pset, name,self._paramReplace)
                                    elif type in ('cms.VInputTag', 'cms.untracked.VInputTag'):
                                            for (i,n) in enumerate(value):
                                                    if not isinstance(n, cms.InputTag):
                                                            n=cms.InputTag(n)
                                                    if n.processName == self._paramSearch:
                                                            # VInputTag can be declared as a list of strings, so ensure that n is formatted correctly
                                                            if self._verbose:print "set process name %s.%s[%d] %s ==> %s " % (base, name, i, n, self._paramReplace)
                                                            setattr(n,"processName",self._paramReplace)
                                                            value[i]=n
				    elif type in ('cms.vstring', 'cms.untracked.vstring'):
					    for (i,n) in enumerate(value):
						    if n==self._paramSearch:
							    getattr(pset,name)[i]=self._paramReplace
				    elif type in ('cms.InputTag', 'cms.untracked.InputTag'):
                                            if value.processName == self._paramSearch:
                                                    if self._verbose: print "set process name %s.%s %s ==> %s " % (base, name, value, self._paramReplace)
                                                    setattr(getattr(pset, name),"processName",self._paramReplace)

            def enter(self,visitee):
                    label = ''
                    try:
                            label = visitee.label()
                    except AttributeError:
                            label = '<Module not in a Process>'
                    except:
                            label = 'other execption'
                    self.doIt(visitee, label)

            def leave(self,visitee):
                    pass

    #visit a sequence to repalce all input tags
    def renameInputTagsInSequence(self,sequence,oldT="rawDataCollector",newT="rawDataRepacker"):
	    print "Replacing all InputTag %s => %s"%(oldT,newT)
	    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
	    massSearchReplaceAnyInputTag(getattr(self.process,sequence),oldT,newT)
	    loadMe='from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag'
	    if not loadMe in self.additionalCommands:
		    self.additionalCommands.append(loadMe)
	    self.additionalCommands.append('massSearchReplaceAnyInputTag(process.%s,"%s","%s",False,True)'%(sequence,oldT,newT))

    #change the process name used to address HLT results in any sequence
    def renameHLTprocessInSequence(self,sequence,proc=None,HLTprocess='HLT'):
	    if self._options.hltProcess:
		    proc=self._options.hltProcess
	    else:
		    proc=self.process.name_()
	    if proc==HLTprocess:    return
            # look up all module in dqm sequence
            print "replacing %s process name - sequence %s will use '%s'" % (HLTprocess,sequence, proc)
            getattr(self.process,sequence).visit(ConfigBuilder.MassSearchReplaceProcessNameVisitor(HLTprocess,proc,whitelist = ("subSystemFolder",)))
            if 'from Configuration.Applications.ConfigBuilder import ConfigBuilder' not in self.additionalCommands:
                    self.additionalCommands.append('from Configuration.Applications.ConfigBuilder import ConfigBuilder')
            self.additionalCommands.append('process.%s.visit(ConfigBuilder.MassSearchReplaceProcessNameVisitor("%s", "%s", whitelist = ("subSystemFolder",)))'% (sequence,HLTprocess, proc))


    def expandMapping(self,seqList,mapping,index=None):
	    maxLevel=20
	    level=0
	    while '@' in repr(seqList) and level<maxLevel:
		    level+=1
		    for specifiedCommand in seqList:
			    if specifiedCommand.startswith('@'):
				    location=specifiedCommand[1:]
				    if not location in mapping:
					    raise Exception("Impossible to map "+location+" from "+repr(mapping))
				    mappedTo=mapping[location]
				    if index!=None:
					    mappedTo=mappedTo[index]
				    seqList.remove(specifiedCommand)
				    seqList.extend(mappedTo.split('+'))
				    break;
	    if level==maxLevel:
		    raise Exception("Could not fully expand "+repr(seqList)+" from "+repr(mapping))
	    
    def prepare_DQM(self, sequence = 'DQMOffline'):
        # this one needs replacement

        self.loadDefaultOrSpecifiedCFF(sequence,self.DQMOFFLINEDefaultCFF)
        sequenceList=sequence.split('.')[-1].split('+')
        postSequenceList=sequence.split('.')[-1].split('+')
	from DQMOffline.Configuration.autoDQM import autoDQM
	self.expandMapping(sequenceList,autoDQM,index=0)
	self.expandMapping(postSequenceList,autoDQM,index=1)
	
	if len(set(sequenceList))!=len(sequenceList):
		sequenceList=list(set(sequenceList))
		print "Duplicate entries for DQM:, using",sequenceList

	pathName='dqmoffline_step'
	for (i,sequence) in enumerate(sequenceList):
		if (i!=0):
			pathName='dqmoffline_%d_step'%(i)
			
		if 'HLT' in self.stepMap.keys() or self._options.hltProcess:
			self.renameHLTprocessInSequence(sequence)

		# if both HLT and DQM are run in the same process, schedule [HLT]DQM in an EndPath
	        # not for fastsim
		if 'HLT' in self.stepMap.keys() and not self._options.fast:
			# need to put [HLT]DQM in an EndPath, to access the HLT trigger results
			setattr(self.process,pathName, cms.EndPath( getattr(self.process, sequence ) ) )
		else:
			# schedule DQM as a standard Path
			setattr(self.process,pathName, cms.Path( getattr(self.process, sequence) ) ) 
		self.schedule.append(getattr(self.process,pathName))

	pathName='dqmofflineOnPAT_step'
	for (i,sequence) in enumerate(postSequenceList):
                if (i!=0):
                        pathName='dqmofflineOnPAT_%d_step'%(i)

		# if both MINIAOD and DQM are run in the same process, schedule DQM in an EndPath
		if 'PAT' in self.stepMap.keys():
			# need to put DQM in an EndPath, to access the miniAOD filter results
			setattr(self.process,pathName, cms.EndPath( getattr(self.process, sequence ) ) )
		else:
			# schedule DQM as a standard Path
			setattr(self.process,pathName, cms.Path( getattr(self.process, sequence) ) )
		self.schedule.append(getattr(self.process,pathName))

    def prepare_HARVESTING(self, sequence = None):
        """ Enrich the process with harvesting step """
        self.DQMSaverCFF='Configuration/StandardSequences/DQMSaver'+self._options.harvesting+'_cff'
        self.loadAndRemember(self.DQMSaverCFF)

        harvestingConfig = self.loadDefaultOrSpecifiedCFF(sequence,self.HARVESTINGDefaultCFF)
        sequence = sequence.split('.')[-1]

        # decide which HARVESTING paths to use
        harvestingList = sequence.split("+")
	from DQMOffline.Configuration.autoDQM import autoDQM
	from Validation.Configuration.autoValidation import autoValidation
	import copy
	combined_mapping = copy.deepcopy( autoDQM )
	combined_mapping.update( autoValidation )
	self.expandMapping(harvestingList,combined_mapping,index=-1)
	
	if len(set(harvestingList))!=len(harvestingList):
		harvestingList=list(set(harvestingList))
		print "Duplicate entries for HARVESTING, using",harvestingList

	for name in harvestingList:
		if not name in harvestingConfig.__dict__:
			print name,"is not a possible harvesting type. Available are",harvestingConfig.__dict__.keys()
			continue
		harvestingstream = getattr(harvestingConfig,name)
		if isinstance(harvestingstream,cms.Path):
			self.schedule.append(harvestingstream)
			self.blacklist_paths.append(harvestingstream)
		if isinstance(harvestingstream,cms.Sequence):
			setattr(self.process,name+"_step",cms.Path(harvestingstream))
			self.schedule.append(getattr(self.process,name+"_step"))

        self.scheduleSequence('DQMSaver','dqmsave_step')
	return

    def prepare_ALCAHARVEST(self, sequence = None):
        """ Enrich the process with AlCaHarvesting step """
        harvestingConfig = self.loadAndRemember(self.ALCAHARVESTDefaultCFF)
        sequence=sequence.split(".")[-1]

        # decide which AlcaHARVESTING paths to use
        harvestingList = sequence.split("+")



 	from Configuration.AlCa.autoPCL import autoPCL
 	self.expandMapping(harvestingList,autoPCL)
	
        for name in harvestingConfig.__dict__:
            harvestingstream = getattr(harvestingConfig,name)
            if name in harvestingList and isinstance(harvestingstream,cms.Path):
               self.schedule.append(harvestingstream)
               self.executeAndRemember("process.PoolDBOutputService.toPut.append(process.ALCAHARVEST" + name + "_dbOutput)")
               self.executeAndRemember("process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVEST" + name + "_metadata)")
               harvestingList.remove(name)
	# append the common part at the end of the sequence
	lastStep = getattr(harvestingConfig,"ALCAHARVESTDQMSaveAndMetadataWriter")
	self.schedule.append(lastStep)
	
        if len(harvestingList) != 0 and 'dummyHarvesting' not in harvestingList :
            print "The following harvesting could not be found : ", harvestingList
            raise Exception("The following harvesting could not be found : "+str(harvestingList))



    def prepare_ENDJOB(self, sequence = 'endOfProcess'):
	    self.loadDefaultOrSpecifiedCFF(sequence,self.ENDJOBDefaultCFF)
	    self.scheduleSequenceAtEnd(sequence.split('.')[-1],'endjob_step')
	    return

    def finalizeFastSimHLT(self):
            self.process.reconstruction = cms.Path(self.process.reconstructionWithFamos)
            self.schedule.append(self.process.reconstruction)


    def build_production_info(self, evt_type, evtnumber):
        """ Add useful info for the production. """
        self.process.configurationMetadata=cms.untracked.PSet\
                                            (version=cms.untracked.string("$Revision: 1.19 $"),
                                             name=cms.untracked.string("Applications"),
                                             annotation=cms.untracked.string(evt_type+ " nevts:"+str(evtnumber))
                                             )

        self.addedObjects.append(("Production Info","configurationMetadata"))


    def prepare(self, doChecking = False):
        """ Prepare the configuration string and add missing pieces."""

        self.loadAndRemember(self.EVTCONTDefaultCFF)  #load the event contents regardless
        self.addMaxEvents()
        if self.with_input:
           self.addSource()
	self.addStandardSequences()
	##adding standard sequences might change the inputEventContent option and therefore needs to be finalized after
	self.completeInputCommand()
        self.addConditions()


        outputModuleCfgCode=""
        if not 'HARVESTING' in self.stepMap.keys() and not 'ALCAHARVEST' in self.stepMap.keys() and not 'ALCAOUTPUT' in self.stepMap.keys() and self.with_output:
                outputModuleCfgCode=self.addOutput()

        self.addCommon()

        self.pythonCfgCode =  "# Auto generated configuration file\n"
        self.pythonCfgCode += "# using: \n# "+__version__[1:-1]+"\n# "+__source__[1:-1]+'\n'
        self.pythonCfgCode += "# with command line options: "+self._options.arguments+'\n'
        self.pythonCfgCode += "import FWCore.ParameterSet.Config as cms\n\n"
        if hasattr(self._options,"era") and self._options.era :
            self.pythonCfgCode += "from Configuration.StandardSequences.Eras import eras\n\n"
            self.pythonCfgCode += "process = cms.Process('"+self.process.name_()+"'" # Start of the line, finished after the loop
            # Multiple eras can be specified in a comma seperated list
            for requestedEra in self._options.era.split(",") :
                self.pythonCfgCode += ",eras."+requestedEra
            self.pythonCfgCode += ")\n\n" # end of the line
        else :
            self.pythonCfgCode += "process = cms.Process('"+self.process.name_()+"')\n\n"

        self.pythonCfgCode += "# import of standard configurations\n"
        for module in self.imports:
            self.pythonCfgCode += ("process.load('"+module+"')\n")

        # production info
        if not hasattr(self.process,"configurationMetadata"):
                self.build_production_info(self._options.evt_type, self._options.number)
        else:
                #the PSet was added via a load
                self.addedObjects.append(("Production Info","configurationMetadata"))

        self.pythonCfgCode +="\n"
        for comment,object in self.addedObjects:
                if comment!="":
                        self.pythonCfgCode += "\n# "+comment+"\n"
                self.pythonCfgCode += dumpPython(self.process,object)

        # dump the output definition
        self.pythonCfgCode += "\n# Output definition\n"
        self.pythonCfgCode += outputModuleCfgCode

        # dump all additional outputs (e.g. alca or skim streams)
        self.pythonCfgCode += "\n# Additional output definition\n"
        #I do not understand why the keys are not normally ordered.
        nl=self.additionalOutputs.keys()
        nl.sort()
        for name in nl:
                output = self.additionalOutputs[name]
                self.pythonCfgCode += "process.%s = %s" %(name, output.dumpPython())
                tmpOut = cms.EndPath(output)
                setattr(self.process,name+'OutPath',tmpOut)
                self.schedule.append(tmpOut)

        # dump all additional commands
        self.pythonCfgCode += "\n# Other statements\n"
        for command in self.additionalCommands:
            self.pythonCfgCode += command + "\n"

        #comma separated list of objects that deserve to be inlined in the configuration (typically from a modified config deep down)
        for object in self._options.inlineObjets.split(','):
                if not object:
                        continue
                if not hasattr(self.process,object):
                        print 'cannot inline -'+object+'- : not known'
                else:
                        self.pythonCfgCode +='\n'
                        self.pythonCfgCode +=dumpPython(self.process,object)

        # dump all paths
        self.pythonCfgCode += "\n# Path and EndPath definitions\n"
        for path in self.process.paths:
            if getattr(self.process,path) not in self.blacklist_paths:
                self.pythonCfgCode += dumpPython(self.process,path)
		
        for endpath in self.process.endpaths:
            if getattr(self.process,endpath) not in self.blacklist_paths:
                self.pythonCfgCode += dumpPython(self.process,endpath)

        # dump the schedule
	self.pythonCfgCode += "\n# Schedule definition\n"
	result = "process.schedule = cms.Schedule("

        # handling of the schedule
	self.process.schedule = cms.Schedule()
	for item in self.schedule:
		if not isinstance(item, cms.Schedule):
			self.process.schedule.append(item)
		else:
			self.process.schedule.extend(item)

	if hasattr(self.process,"HLTSchedule"):
		beforeHLT = self.schedule[:self.schedule.index(self.process.HLTSchedule)]
		afterHLT = self.schedule[self.schedule.index(self.process.HLTSchedule)+1:]
		pathNames = ['process.'+p.label_() for p in beforeHLT]
		result += ','.join(pathNames)+')\n'
		result += 'process.schedule.extend(process.HLTSchedule)\n'
		pathNames = ['process.'+p.label_() for p in afterHLT]
		result += 'process.schedule.extend(['+','.join(pathNames)+'])\n'
	else:
		pathNames = ['process.'+p.label_() for p in self.schedule]
		result ='process.schedule = cms.Schedule('+','.join(pathNames)+')\n'

	self.pythonCfgCode += result

	if self._options.nThreads is not "1":
		self.pythonCfgCode +="\n"
		self.pythonCfgCode +="#Setup FWK for multithreaded\n"
		self.pythonCfgCode +="process.options.numberOfThreads=cms.untracked.uint32("+self._options.nThreads+")\n"
		self.pythonCfgCode +="process.options.numberOfStreams=cms.untracked.uint32(0)\n"
	#repacked version
	if self._options.isRepacked:
		self.pythonCfgCode +="\n"
		self.pythonCfgCode +="from Configuration.Applications.ConfigBuilder import MassReplaceInputTag\n"
		self.pythonCfgCode +="MassReplaceInputTag(process)\n"
		MassReplaceInputTag(self.process)
		
        # special treatment in case of production filter sequence 2/2
        if self.productionFilterSequence:
                self.pythonCfgCode +='# filter all path with the production filter sequence\n'
                self.pythonCfgCode +='for path in process.paths:\n'
		if len(self.conditionalPaths):
			self.pythonCfgCode +='\tif not path in %s: continue\n'%str(self.conditionalPaths)
                if len(self.excludedPaths):
                        self.pythonCfgCode +='\tif path in %s: continue\n'%str(self.excludedPaths)			
                self.pythonCfgCode +='\tgetattr(process,path)._seq = process.%s * getattr(process,path)._seq \n'%(self.productionFilterSequence,)
		pfs = getattr(self.process,self.productionFilterSequence)
		for path in self.process.paths:
			if not path in self.conditionalPaths: continue
                        if path in self.excludedPaths: continue
			getattr(self.process,path)._seq = pfs * getattr(self.process,path)._seq
			

        # dump customise fragment
	self.pythonCfgCode += self.addCustomise()

	if self._options.runUnscheduled:	
		# prune and delete paths
		#this is not supporting the blacklist at this point since I do not understand it
		self.pythonCfgCode+="#do not add changes to your config after this point (unless you know what you are doing)\n"
		self.pythonCfgCode+="from FWCore.ParameterSet.Utilities import convertToUnscheduled\n"
		self.pythonCfgCode+="process=convertToUnscheduled(process)\n"

		from FWCore.ParameterSet.Utilities import convertToUnscheduled
		self.process=convertToUnscheduled(self.process)

		#now add the unscheduled stuff
		for module in self.importsUnsch:
			self.process.load(module)
			self.pythonCfgCode += ("process.load('"+module+"')\n")

		#and clean the unscheduled stuff	
		self.pythonCfgCode+="from FWCore.ParameterSet.Utilities import cleanUnscheduled\n"
		self.pythonCfgCode+="process=cleanUnscheduled(process)\n"

		from FWCore.ParameterSet.Utilities import cleanUnscheduled
		self.process=cleanUnscheduled(self.process)


	self.pythonCfgCode += self.addCustomise(1)


	# make the .io file

	if self._options.io:
		#io=open(self._options.python_filename.replace('.py','.io'),'w')
		if not self._options.io.endswith('.io'): self._option.io+='.io'
		io=open(self._options.io,'w')
		ioJson={}
		if hasattr(self.process.source,"fileNames"):
			if len(self.process.source.fileNames.value()):
				ioJson['primary']=self.process.source.fileNames.value()
		if hasattr(self.process.source,"secondaryFileNames"):
			if len(self.process.source.secondaryFileNames.value()):
				ioJson['secondary']=self.process.source.secondaryFileNames.value()
		if self._options.pileup_input and (self._options.pileup_input.startswith('dbs:') or self._options.pileup_input.startswith('das:')):
			ioJson['pileup']=self._options.pileup_input[4:]
		for (o,om) in self.process.outputModules_().items():
			ioJson[o]=om.fileName.value()
		ioJson['GT']=self.process.GlobalTag.globaltag.value()
		if self.productionFilterSequence:
			ioJson['filter']=self.productionFilterSequence
		import json
		io.write(json.dumps(ioJson))
	return

