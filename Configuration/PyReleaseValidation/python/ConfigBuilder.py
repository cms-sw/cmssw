#! /usr/bin/env python

# This is a prototype for the new pyrelease validation package
# this class here takes the input of optparse in cmsDriver and
# creates a complete config file.
# relval_main + the custom config for it is not needed any more

__version__ = "$Revision: 1.27 $"
__source__ = "$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v $"

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Modules import _Module 


# some helper routines
def dumpPython(process,name):
    theObject = getattr(process,name)
    if isinstance(theObject,cms.Path) or isinstance(theObject,cms.EndPath) or isinstance(theObject,cms.Sequence):
        return "process."+name+" = " + theObject.dumpPython("process")
    elif isinstance(theObject,_Module):
        return "process."+name+" = " + theObject.dumpPython()


class ConfigBuilder(object):
    """The main building routines """
    
    def __init__(self,options):
        """options taken from old cmsDriver and optparse """
        self._options = options
        self.process = cms.Process(self._options.name)
        self.process.schedule = cms.Schedule()        
        # we are doing three things here:
        # creating a process to catch errors
        # building the code to re-create the process
        # check the code at the very end
        self.pythonCfgCode =  "# Auto generated configuration file\n"
        self.pythonCfgCode += "# using: \n# "+__version__+"\n# "+__source__+"\n"
        self.pythonCfgCode += "import FWCore.ParameterSet.Config as cms\n\n"
        self.pythonCfgCode += "process = cms.Process('"+self._options.name+"')\n\n"       
        self.imports = []  #could we use a set instead?
        self.commands = []
        # TODO: maybe a list of to be dumped objects would help as well        
        self.blacklist_paths = [] 
        self.additionalObjects = []
        self.productionFilterSequence = None

    def loadAndRemember(self, includeFile):
        """helper routine to load am memorize imports"""
        # we could make the imports a on-the-fly data method of the process instance itself
        # not sure if the latter is a good idea
        self.imports.append(includeFile)
        self.process.load(includeFile)

    def executeAndRemember(self, command):
        """helper routine to remember replace statements"""
        pass        
        
    def addCommon(self):
        self.process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound'),  wantSummary = cms.untracked.bool(True) )


    def addMaxEvents(self):
        """Here we decide how many evts will be processed"""
        self.process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(int(self._options.number)))
                        
    def addSource(self):
        """Here the source is built. Priority: file, generator"""
        if self._options.filein:
            self.process.source=cms.Source("PoolSource", fileNames = cms.untracked.vstring(self._options.filein))
        elif hasattr(self._options,'evt_type'):
            evt_type = self._options.evt_type.rstrip(".py").replace(".","_")
            if "/" in evt_type:
                evt_type = evt_type.replace("python/","")
            else:
                evt_type = 'Configuration/Generator/'+evt_type 

            sourceModule = __import__(evt_type)
            self.process.extend(sourceModule)
            # now add all modules and sequences to the process
            import FWCore.ParameterSet.Modules as cmstypes  
            for name in sourceModule.__dict__:
                theObject = getattr(sourceModule,name)
                if isinstance(theObject, cmstypes._Module):
                   self.additionalObjects.insert(0,name)
                if isinstance(theObject, cms.Sequence):
                   self.additionalObjects.append(name)

        return

    def addOutput(self):
        """ Add output module to the process """    
        
        self.loadAndRemember(self.contentFile)
        theEventContent = getattr(self.process, self._options.eventcontent+"EventContent")
 
        output = cms.OutputModule("PoolOutputModule",
                                  theEventContent,
                                  fileName = cms.untracked.string(self._options.outfile_name),
                                  dataset = cms.untracked.PSet(dataTier =cms.untracked.string(self._options.datatier))
                                 ) 

        # if there is a generation step in the process, that one should be used as filter decision
        if hasattr(self.process,"generation_step"):
            output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step')) 
        
        # if a filtername is given, use that one
        if self._options.filtername !="":
            output.dataset.filterName = cms.untracked.string(self._options.filtername)

        # and finally add the output to the process
        self.process.output = output
        self.process.out_step = cms.EndPath(self.process.output)
        self.process.schedule.append(self.process.out_step)

        # ATTENTION: major tweaking to avoid inlining of event content
        # should we do that?
        def dummy(instance,label = "process."+self._options.eventcontent+"EventContent.outputCommands"):
            return label
        
        self.process.output.outputCommands.__dict__["dumpPython"] = dummy
        
        return "\n"+self.process.output.dumpPython()
        
        
    def addStandardSequences(self):
        """
        Add selected standard sequences to the process
        """
        conditionsSP=self._options.conditions.split(',')

        # here we check if we have fastsim or fullsim
        if "FAST" in self._options.step:
            self.contentFile = "FastSimulation/Configuration/EventContent_cff"
            self.imports=['FastSimulation/Configuration/RandomServiceInitialization_cff']

        # no fast sim   
        else:
            self.contentFile = "Configuration/EventContent/EventContent_cff"
            self.imports=['Configuration/StandardSequences/Services_cff',
                          'Configuration/StandardSequences/Geometry_cff',
                          'FWCore/MessageService/MessageLogger_cfi',
                          'Configuration/StandardSequences/Generator_cff']         # rm    
            if self._options.magField == "3.8T":
               self.imports.append('Configuration/StandardSequences/MagneticField_38_cff')
            else:
                self.imports.append('Configuration/StandardSequences/MagneticField_cff')
           
            if self._options.PU_flag:
                self.imports.append('Configuration/StandardSequences/MixingLowLumiPileUp_cff')
            else:
                self.imports.append('Configuration/StandardSequences/MixingNoPileUp_cff')


        # what steps are provided by this class?
        stepList = [methodName.lstrip("prepare_") for methodName in self.__class__.__dict__ if methodName.startswith('prepare_')]

        # look which steps are requested and invoke the corresponding method
        for step in self._options.step.split(","):
            stepParts = step.split(":")   # for format STEP:alternativeSequence
            stepName = stepParts[0]
            if stepName not in stepList:
                raise ValueError("Step "+stepName+" unknown")
            if len(stepParts)==1:
                getattr(self,"prepare_"+step)()            
            elif len(stepParts)==2:
                getattr(self,"prepare_"+stepName)(sequence = stepParts[1])
            else:
                raise ValueError("Step definition "+step+" invalid")


    def addConditions(self):
        """Add conditions to the process"""
        conditionsSP=self._options.conditions.split(',')
        # FULL or FAST SIM ?
        if "FASTSIM" in self._options.step:
            # fake or real conditions?
            if len(conditionsSP)>1:
                self.loadAndRemember('FastSimulation/Configuration/CommonInputs_cff')
            else:
                self.loadAndRemember('FastSimulation/Configuration/CommonInputsFake_cff')
        else:
            self.loadAndRemember('Configuration/StandardSequences/'+conditionsSP[0]+'_cff')
        
        # set non-default conditions 
        if ( len(conditionsSP)>1 ):
            self.commands.append("process.GlobalTag.globaltag = '"+str(conditionsSP[1]+"'"))
                        
    def addCustomise(self):
        """Include the customise code """

        # let python search for that package and do syntax checking at the same time
        packageName = self._options.customisation_file.replace(".py","")
        package = __import__(packageName)

        # now ask the package for its definition and pick .py instead of .pyc
        customiseFile = package.__file__.rstrip("c")
        
        final_snippet='\n\n# Automatic addition of the customisation function\n'
        for line in file(customiseFile,'r'):
            if "import FWCore.ParameterSet.Config" in line:
                continue
            final_snippet += line
        
        final_snippet += '\n\n# End of customisation function definition'

        return final_snippet + "\n\nprocess = customise(process)"

    
    #----------------------------------------------------------------------------
    # here the methods to create the steps. Of course we are doing magic here ;)
    # prepare_STEPNAME modifies self.process and what else's needed.
    #----------------------------------------------------------------------------

    def prepare_GEN(self, sequence = None):
        """ Enrich the schedule with the generation step """    
        self.loadAndRemember("Configuration/StandardSequences/Generator_cff")
        
        # replace the VertexSmearing placeholder by a concrete beamspot definition
        try: 
            self.loadAndRemember('Configuration/StandardSequences/VtxSmeared'+self._options.beamspot+'_cff')
        except ImportError:
            print "VertexSmearing type or beamspot",self._options.beamspot, "unknown."
            raise
        self.process.generation_step = cms.Path( self.process.pgen )
        self.process.schedule.append(self.process.generation_step)

        # is there a production filter sequence given?
        if sequence:
            if sequence not in self.additionalObjects:
                raise AttributeError("There is no filter sequence '"+sequence+"' defined in "+self._options.evt_type)
            else:
                self.productionFilterSequence = sequence
        return

    def prepare_SIM(self, sequence = None):
        """ Enrich the schedule with the simulation step"""
        self.loadAndRemember("Configuration/StandardSequences/Simulation_cff")
        self.process.simulation_step = cms.Path( self.process.simulation )
        self.process.schedule.append(self.process.simulation_step)
        return     

    def prepare_DIGI(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
        self.loadAndRemember("Configuration/StandardSequences/Simulation_cff")
        self.process.digitisation_step = cms.Path(self.process.pdigi)    
        self.process.schedule.append(self.process.digitisation_step)
        return

    def prepare_DIGI2RAW(self, sequence = None):
        self.loadAndRemember("Configuration/StandardSequences/DigiToRaw_cff")
        self.process.digi2raw_step = cms.Path( self.process.DigiToRaw )
        self.process.schedule.append(self.process.digi2raw_step)
        return

    def prepare_L1(self, sequence = None):
        """ Enrich the schedule with the L1 simulation step"""
        self.loadAndRemember('Configuration/StandardSequences/SimL1Emulator_cff') 
        self.loadAndRemember('L1TriggerConfig/L1GtConfigProducers/Luminosity/lumi1030.L1Menu2008_2E30_Unprescaled_cff')
        self.process.L1simulation_step = cms.Path(self.process.SimL1Emulator)
        self.process.schedule.append(self.process.L1simulation_step)

    def prepare_HLT(self, sequence = None):
        """ Enrich the schedule with the HLT simulation step"""
        self.loadAndRemember("HLTrigger/Configuration/HLT_2E30_cff")
        hltconfig = __import__("HLTrigger/Configuration/HLT_2E30_cff")

        # BH: hack to let HLT run without explicit schedule 
        hltOrder = []
        for pathname in hltconfig.__dict__:
          if isinstance(getattr(hltconfig,pathname),cms.Path):
            if pathname == "HLTriggerFirstPath":
                hltOrder.insert(0,getattr(self.process,pathname))  # put explicitly in front
            elif pathname == "HLTriggerFinalPath":
                last = getattr(self.process,pathname)
            else:
                hltOrder.append(getattr(self.process,pathname))
        if last: hltOrder.append(last)

        self.process.schedule.extend(hltOrder)
               
        [self.blacklist_paths.append(name) for name in hltconfig.__dict__ if isinstance(getattr(hltconfig,name),cms.Path)]
        [self.process.schedule.append(getattr(self.process,name)) for name in hltconfig.__dict__ if isinstance(getattr(hltconfig,name),cms.EndPath)]
        [self.blacklist_paths.append(name) for name in hltconfig.__dict__ if isinstance(getattr(hltconfig,name),cms.EndPath)]
  

    def prepare_RAW2DIGI(self, sequence = None):
        self.loadAndRemember("Configuration/StandardSequences/RawToDigi_cff")
        self.process.raw2digi_step = cms.Path( self.process.RawToDigi )
        self.process.schedule.append(self.process.raw2digi_step)
        return

    def prepare_RECO(self, sequence = "reconstruction"):
        ''' Enrich the schedule with reconstruction '''
        self.loadAndRemember("Configuration/StandardSequences/Reconstruction_cff")
        self.process.reconstruction_step = cms.Path( getattr(self.process, sequence) )
        self.process.schedule.append(self.process.reconstruction_step)
        return

    def prepare_POSTRECO(self, sequence = None):
        """ Enrich the schedule with the postreco step """
        self.loadAndRemember("Configuration/StandardSequences/PostRecoGenerator_cff")
        self.process.postreco_step = cms.Path( self.process.postreco_generator )
        self.process.schedule.append(self.process.postreco_step)
        return                         


    def prepare_PATLayer0(self, sequence = None):
        """ In case people would like to have this"""
        pass

    def prepare_DQM(self, sequence = None):
        self.loadAndRemember("Configuration/StandardSequences/Validation_cff")
        self.process.validation_step = cms.Path( self.process.validation )
        self.process.schedule.append(self.process.validation_step)


    def prepare_FASTSIM(self, sequence = "famosWithEverything"):
        """ Enrich the schedule with fastsim """
        self.loadAndRemember("FastSimulation/Configuration/FamosSequences_cff")
        self.process.fastsim_step = cms.Path( getattr(self.process, sequence) )
        self.process.schedule.append(self.process.fastsim_step)
    
    def build_production_info(self, evt_type, energy, evtnumber):
        """ Add useful info for the production. """
        prod_info=cms.untracked.PSet\
              (version=cms.untracked.string("$Revision: 1.27 $"),
               name=cms.untracked.string("PyReleaseValidation"),
               annotation=cms.untracked.string(evt_type+" energy:"+str(energy)+" nevts:"+str(evtnumber))
              )
    
        return prod_info
 
   
    def prepare(self, doChecking = False):
        """ Prepare the configuration string and add missing pieces."""

        self.addMaxEvents()                    
        self.addSource()
        self.addStandardSequences()
        self.addConditions()
        self.addOutput()
        self.addCommon()
        
        self.pythonCfgCode += "# import of standard configurations\n"
        for module in self.imports:
            self.pythonCfgCode += ("process.load('"+module+"')\n")

        # dump ReleaseValidation PSet
        totnumevts = int(self._options.relval.split(",")[0])
        evtsperjob = int(self._options.relval.split(",")[1])
        dsetname="RelVal"+self._options.ext_process_name

        self.process.ReleaseValidation=cms.untracked.PSet(totalNumberOfEvents=cms.untracked.int32(totnumevts),
                                                     eventsPerJob=cms.untracked.int32(evtsperjob),
                                                     primaryDatasetName=cms.untracked.string(dsetname))
        self.pythonCfgCode += "\nprocess.ReleaseValidation = "+self.process.ReleaseValidation.dumpPython()
 
        # dump production info
        if not hasattr(self.process,"configurationMetadata"):
            self.process.configurationMetadata=self.build_production_info(self._options.evt_type, self._options.energy, self._options.number)
        self.pythonCfgCode += "\nprocess.configurationMetadata = "+self.process.configurationMetadata.dumpPython()       
        
        # dump max events block
        self.pythonCfgCode += "\nprocess.maxEvents = "+self.process.maxEvents.dumpPython()

        # dump the job options
        self.pythonCfgCode += "\nprocess.options = "+self.process.options.dumpPython()

        # dump the input definition
        self.pythonCfgCode += "\n# Input source\n"
        self.pythonCfgCode += "process.source = "+self.process.source.dumpPython() 
        
        # dump the output definition
        self.pythonCfgCode += "\n# Output definition\n"
        self.pythonCfgCode += "process.output = "+self.process.output.dumpPython()

        # dump all additional commands
        self.pythonCfgCode += "\n# Other statements\n"
        for command in self.commands:
            self.pythonCfgCode += command + "\n"

        # special treatment for a production filter sequence 
        if self.productionFilterSequence:
            # dump all additional definitions from the input definition file
            for name in self.additionalObjects:
                self.pythonCfgCode += dumpPython(self.process,name)
            # prepend the productionFilterSequence to all paths defined
            for path in self.process.paths:
                getattr(self.process,path)._seq = getattr(self.process,self.productionFilterSequence)*getattr(self.process,path)._seq
            # as HLT paths get modified as well, they have to be re-printed
            self.blacklist_paths = []
                
        # dump all paths
        self.pythonCfgCode += "\n# Path and EndPath definitions\n"
        for path in self.process.paths:
            if path not in self.blacklist_paths:
                self.pythonCfgCode += dumpPython(self.process,path)
        for endpath in self.process.endpaths:
            if endpath not in self.blacklist_paths:
                self.pythonCfgCode += dumpPython(self.process,endpath)

        # dump the schedule
        self.pythonCfgCode += "\n# Schedule definition\n"
        pathNames = ['process.'+p.label() for p in self.process.schedule]
        result ='process.schedule = cms.Schedule('+','.join(pathNames)+')\n'
        self.pythonCfgCode += result

        # dump customise fragment
        if self._options.customisation_file:
            self.pythonCfgCode += self.addCustomise()
         
        return
      
      
      
        
        
