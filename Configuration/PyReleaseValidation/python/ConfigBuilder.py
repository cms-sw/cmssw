#! /usr/bin/env python

# This is a prototype for the new pyrelease validation package
# this class here takes the input of optparse in cmsDriver and
# creates a complete config file.
# relval_main + the custom config for it is not needed any more

__version__ = "$Revision: 1.48 $"
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


def findName(object,dictionary):
    for name, item in dictionary.iteritems():
        if item == object:
            return name
             
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
        self.imports = []  #could we use a set instead?
        self.additionalCommands = []
        # TODO: maybe a list of to be dumped objects would help as well        
        self.blacklist_paths = [] 
        self.additionalObjects = []
        self.additionalOutputs = []
        self.productionFilterSequence = None

    def loadAndRemember(self, includeFile):
        """helper routine to load am memorize imports"""
        # we could make the imports a on-the-fly data method of the process instance itself
        # not sure if the latter is a good idea
        self.imports.append(includeFile)
        self.process.load(includeFile)
        return __import__(includeFile)

    def executeAndRemember(self, command):
        """helper routine to remember replace statements"""
        pass        
        
    def addCommon(self):
        self.process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

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
        theEventContent = getattr(self.process, self._options.eventcontent.split(',')[-1]+"EventContent")
 
        output = cms.OutputModule("PoolOutputModule",
                                  theEventContent,
                                  fileName = cms.untracked.string(self._options.outfile_name),
                                  dataset = cms.untracked.PSet(dataTier =cms.untracked.string(self._options.datatier))
                                 ) 

        # if there is a generation step in the process, that one should be used as filter decision
        if hasattr(self.process,"generation_step"):
            output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step')) 
        
        # if a filtername is given, use that one
        if self._options.filtername !='':
            output.dataset.filterName = cms.untracked.string(self._options.filtername)
        else:
            conditionsSP = self._options.conditions.split(',')
            if len(conditionsSP) > 1:
              output.dataset.filterName = cms.untracked.string(str(conditionsSP[1].split("::")[0]))

        # and finally add the output to the process
        self.process.output = output
        self.process.out_step = cms.EndPath(self.process.output)

        for item in self.additionalOutputs:
            self.process.out_step.append(item)
        self.process.schedule.append(self.process.out_step)

        # ATTENTION: major tweaking to avoid inlining of event content
        # should we do that?
        def dummy(instance,label = "process."+self._options.eventcontent.split(',')[-1]+"EventContent.outputCommands"):
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
            # this may get overriden by the user setting of --eventcontent
            self.contentFile = "Configuration/EventContent/EventContent_cff"
            self.imports=['Configuration/StandardSequences/Services_cff',
                          #'Configuration/StandardSequences/Geometry_cff',
                          'FWCore/MessageService/MessageLogger_cfi',
                          'Configuration/StandardSequences/Generator_cff']         # rm    

            # pile up handling is full sim specific
            if self._options.PU_flag:
                self.imports.append('Configuration/StandardSequences/MixingLowLumiPileUp_cff')
            else:
                self.imports.append('Configuration/StandardSequences/MixingNoPileUp_cff')

            # the geometry
            self.imports.append('Configuration/StandardSequences/Geometry'+self._options.geometry+'_cff')

        # the magnetic field
        self.imports.append('Configuration/StandardSequences/MagneticField_'+self._options.magField.replace('.','')+'_cff')
                                               

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
            elif len(stepParts)==3:
                getattr(self,"prepare_"+stepName)(sequence = stepParts[1]+','+stepParts[2])

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
            self.additionalCommands.append("process.GlobalTag.globaltag = '"+str(conditionsSP[1]+"'"))
                        
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

    def prepare_ALCA(self, sequence = None):
        """ Enrich the process with alca streams """
        alcaConfig = self.loadAndRemember("Configuration/StandardSequences/AlCaRecoStreams_cff")

        # decide which ALCA paths to use
        alcaList = sequence.split("+")
        alcaPathList = ["pathALCARECO"+name for name in alcaList]

        for name in alcaConfig.__dict__:
            alcastream = getattr(alcaConfig,name)
            if name in alcaList and isinstance(alcastream,alcaConfig.FilteredStream):
                alcaOutput = cms.OutputModule("PoolOutputModule")
                alcaOutput.SelectEvents = alcastream.SelectEvents
                alcaOutput.outputCommands = alcastream.content
                alcaOutput.dataset  = cms.untracked.PSet( dataTier = alcastream.dataTier)
                self.additionalOutputs.append(alcaOutput)
                setattr(self.process,name,alcaOutput) 

        # the schedule insertion is missing for now  

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
        self.loadAndRemember("Configuration/StandardSequences/Sim_cff")
        self.process.simulation_step = cms.Path( self.process.psim )
        self.process.schedule.append(self.process.simulation_step)
        return     

    def prepare_DIGI(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
        self.loadAndRemember("Configuration/StandardSequences/Digi_cff")
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

        self.process.schedule.extend(self.process.HLTSchedule)
        [self.blacklist_paths.append(path) for path in self.process.HLTSchedule if isinstance(path,(cms.Path,cms.EndPath))]
  
    def prepare_RAW2DIGI(self, sequence = "RawToDigi"):
        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember("Configuration/StandardSequences/RawToDigi_cff")
        else:    
            self.loadAndRemember(sequence.split(',')[0])
        self.process.raw2digi_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        self.process.schedule.append(self.process.raw2digi_step)
        return

    def prepare_RECO(self, sequence = "reconstruction"):
        ''' Enrich the schedule with reconstruction '''
        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember("Configuration/StandardSequences/Reconstruction_cff")
        else:    
            self.loadAndRemember(sequence.split(',')[0])
        self.process.reconstruction_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
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


    def prepare_FASTSIM(self, sequence = "all"):
        """Enrich the schedule with fastsim"""
        self.loadAndRemember("FastSimulation/Configuration/FamosSequences_cff")

        if sequence == "all":
            self.loadAndRemember("FastSimulation/Configuration/HLT_cff")

            # no need to repeat the definition later on in the created file 
            [self.blacklist_paths.append(path) for path in self.process.HLTSchedule if isinstance(path,(cms.Path,cms.EndPath))]

            # endpaths do logging only which should be suppressed in production
            self.process.HLTSchedule.remove(self.process.HLTAnalyzerEndpath)

            self.loadAndRemember("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
            self.additionalCommands.append("process.famosSimHits.SimulateCalorimetry = True")
            self.additionalCommands.append("process.famosSimHits.SimulateTracking = True")
            self.additionalCommands.append("process.famosPileUp.PileUpSimulator.averageNumber = 0.0")
            self.additionalCommands.append("process.caloRecHits.RecHitsFactory.doMiscalib = True")

            # Apply Tracker misalignment (ideal alignment though)
            self.additionalCommands.append("process.famosSimHits.ApplyAlignment = True")
            self.additionalCommands.append("process.misalignedTrackerGeometry.applyAlignment = True")
            self.additionalCommands.append("process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0")
            self.additionalCommands.append("process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0")

            self.additionalCommands.append("process.simulation = cms.Sequence(process.simulationWithFamos)")
            self.additionalCommands.append("process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)")

            # since we have HLT here, the process should be called HLT
            self._options.name = "HLT"

            self.process.schedule.extend(self.process.HLTSchedule)
            self.process.reconstruction = cms.Path(self.process.reconstructionWithFamos)
            self.process.schedule.append(self.process.reconstruction)
        else:
            self.process.fastsim_step = cms.Path( getattr(self.process, "famosWithEverything") )
            self.process.schedule.append(self.process.fastsim_step)

            # now the additional commands we need to make the config work
            self.additionalCommands.append("process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True")
                                            

    def build_production_info(self, evt_type, evtnumber):
        """ Add useful info for the production. """
        prod_info=cms.untracked.PSet\
              (version=cms.untracked.string("$Revision: 1.48 $"),
               name=cms.untracked.string("PyReleaseValidation"),
               annotation=cms.untracked.string(evt_type+ " nevts:"+str(evtnumber))
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

        self.pythonCfgCode =  "# Auto generated configuration file\n"
        self.pythonCfgCode += "# using: \n# "+__version__+"\n# "+__source__+"\n"
        self.pythonCfgCode += "import FWCore.ParameterSet.Config as cms\n\n"
        self.pythonCfgCode += "process = cms.Process('"+self._options.name+"')\n\n"
        
        self.pythonCfgCode += "# import of standard configurations\n"
        for module in self.imports:
            self.pythonCfgCode += ("process.load('"+module+"')\n")

        # dump ReleaseValidation PSet
        totnumevts = int(self._options.relval.split(",")[0])
        evtsperjob = int(self._options.relval.split(",")[1])
        dsetname="RelVal"+self._options.evt_type.replace(".","_").rstrip("_cfi")

        self.process.ReleaseValidation=cms.untracked.PSet(totalNumberOfEvents=cms.untracked.int32(totnumevts),
                                                     eventsPerJob=cms.untracked.int32(evtsperjob),
                                                     primaryDatasetName=cms.untracked.string(dsetname))
        self.pythonCfgCode += "\nprocess.ReleaseValidation = "+self.process.ReleaseValidation.dumpPython()
 
        # dump production info
        if not hasattr(self.process,"configurationMetadata"):
            self.process.configurationMetadata=self.build_production_info(self._options.evt_type, self._options.number)
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
        for command in self.additionalCommands:
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
            if getattr(self.process,path) not in self.blacklist_paths:
                self.pythonCfgCode += dumpPython(self.process,path)
        for endpath in self.process.endpaths:
            if getattr(self.process,endpath) not in self.blacklist_paths:
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
      
      
      
        
        
