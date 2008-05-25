#! /usr/bin/env python

# This is a prototype for the new pyrelease validation package
# this class here takes the input of optparse in cmsDriver and
# creates a complete config file.
# relval_main + the custom config for it is not needed any more

__version__ = "$Revision: 1.6 $"
__source__ = "$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v $"

import FWCore.ParameterSet.Config as cms
import new


def AccessCheckingInstanceDecorator(instance):
    """Enable an object to keep track of access to methods and data members"""

    instance._logSet = set()

    def loggingGetattribute(self, name):
        """record every attribute access"""
        self._logSet.add(name)
        return self.__dict__[name]
    instance.__getattr__ = new.instancemethod(loggingGetattribute, instance, instance.__class__)

    def listUnused(self):
        """compare accessed attributes with all non 'private' attributes"""
        unused = [name for name in self.__dict__ if name not in self._logSet and not name.startswith("_") ]
        return unused
    instance._listUnused = new.instancemethod(listUnused, instance, instance.__class__)

    return instance
                                                        


class ConfigBuilder(object):
    """The main building routines """
    
    def __init__(self,options):
        """options taken from old cmsDriver and optparse """
        self._options = AccessCheckingInstanceDecorator(options)  
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
        pass

    def addMaxEvents(self):
        """Here we decide how many evts will be processed"""
        self.process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(int(self._options.number)))
                        
    def addSource(self):
        """Here the source is built. Priority: pythia, file"""
        # prepared by D. Piparo 

        if self._options.filein:
            self.process.source=cms.Source("PoolSource", fileNames = cms.untracked.vstring(self._options.filein))

        elif hasattr(self._options,'evt_type'):
            import Configuration.PyReleaseValidation.Generation as newGenerator
            self.process.source=newGenerator.generate(self._options.evt_type,
                                                      self._options.energy,
                                                      self._options.number)
        return

    def addOutput(self):
        """ Add output module to the process """    
        
        self.loadAndRemember("Configuration/EventContent/EventContent_cff")
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
    
        # here the default includes
        self.imports=['Configuration/StandardSequences/Services_cff',
                            'Configuration/StandardSequences/Geometry_cff',
                            'Configuration/StandardSequences/MagneticField_cff',
                            'FWCore/MessageService/MessageLogger_cfi',
                            'Configuration/StandardSequences/Generator_cff',         # rm    
                            'Configuration/StandardSequences/'+conditionsSP[0]+'_cff']        # should get it's own block I would say     

        if self._options.PU_flag:
            self.imports.append('Configuration/StandardSequences/MixingLowLumiPileUp_cff')
        else:
            self.imports.append('Configuration/StandardSequences/MixingNoPileUp_cff')

        # what steps are provided by this class?
        stepList = [methodName.lstrip("prepare_") for methodName in self.__class__.__dict__ if methodName.startswith('prepare_')]
        # look which steps are requested and invoke the corresponding method
        for step in self._options.step.split(","):
            if step not in stepList:
                raise ValueError("Step "+step+" unknown")
            getattr(self,"prepare_"+step)()            


        if ( len(conditionsSP)>1 ):
            self.commands.append("process.GlobalTag.globaltag = '"+str(conditionsSP[1]+"'"))
                                   

    def addConditions(self):
        """Add conditions to the process"""
        # conditions stuff has to move here
        pass 
      
    def addCustomise(self):
        """Include the customise code """

        filename=self._options.customisation_file

        final_snippet='\n\n# Automatic addition of the customisation function'

        for line in file(filename,'r'):
            if "import FWCore.ParameterSet.Config" in line:
                continue
            final_snippet += line
        
        final_snippet += '\n\n# End of customisation function definition'

        return final_snippet + "\n\nprocess = customise (process)"

        #self.process=file.customise(self.process)
        #if process == None:
            #raise ValueError("Customise file returns no process. Please add a 'return process'.")



    
    #----------------------------------------------------------------------------
    # here the methods to create the steps. Of course we are doing magic here ;)
    # prepare_STEPNAME modifies self.process and what else's needed.
    #----------------------------------------------------------------------------

    def prepare_GEN(self):
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
        return

    def prepare_SIM(self):
        """ Enrich the schedule with the simulation step"""
        self.loadAndRemember("Configuration/StandardSequences/Simulation_cff")
        self.process.simulation_step = cms.Path( self.process.simulation )
        self.process.schedule.append(self.process.simulation_step)
        return     

    def prepare_DIGI(self):
        """ Enrich the schedule with the digitisation step"""
        self.loadAndRemember("Configuration/StandardSequences/Simulation_cff")
        self.process.digitisation_step = cms.Path(self.process.pdigi)    
        self.process.schedule.append(self.process.digitisation_step)
        return

    def prepare_DIGI2RAW(self):
        self.loadAndRemember("Configuration/StandardSequences/DigiToRaw_cff")
        self.process.digi2raw_step = cms.Path( self.process.DigiToRaw )
        self.process.schedule.append(self.process.digi2raw_step)
        return

    def prepare_L1(self):
        """ Enrich the schedule with the L1 simulation step"""
        self.loadAndRemember('Configuration/StandardSequences/SimL1Emulator_cff') 
        self.process.L1simulation_step = cms.Path(self.process.SimL1Emulator)
        self.process.schedule.append(self.process.L1simulation_step)

    def prepare_HLT(self):
        """ Enrich the schedule with the HLT simulation step"""
        self.loadAndRemember("OHJE")
        # now we have to loop every single path in the process and check if that's an HLT path.
        # TODO

    def prepare_RAW2DIGI(self):
        self.loadAndRemember("Configuration/StandardSequences/RawToDigi_cff")
        self.process.raw2digi_step = cms.Path( self.process.RawToDigi )
        self.process.schedule.append(self.process.raw2digi_step)
        return

    def prepare_RECO(self):
        ''' Enrich the schedule with reconstruction '''
        self.loadAndRemember("Configuration/StandardSequences/Reconstruction_cff")
        self.process.reconstruction_step = cms.Path( self.process.reconstruction )
        self.process.schedule.append(self.process.reconstruction_step)
        return

    def prepare_POSTRECO(self):
        """ Enrich the schedule with the postreco step """
        self.loadAndRemember("Configuration/StandardSequences/PostRecoGenerator_cff")
        self.process.postreco_step = cms.Path( self.process.postreco_generator )
        self.process.schedule.append(self.process.postreco_step)
        return                         


    def prepare_PATLayer0(self):
        """ In case people would like to have this"""
        pass

    def prepare_DQM(self):
        pass
    
    def build_production_info(evt_type, energy, evtnumber):
        """ Add useful info for the production. """
        prod_info=cms.untracked.PSet\
              (version=cms.untracked.string("$Revision: 1.6 $"),
               name=cms.untracked.string("PyReleaseValidation")#,
              # annotation=cms.untracked.string(self._options.evt_type+" energy:"+str(energy)+" nevts:"+str(evtnumber))
              )
    
        return prod_info
 
   
    def addFloatingPointException(self, options="1110"):
        """ A service for trapping floating point exceptions """
        fpe_service=cms.Service("EnableFloatingPointExceptions", enableDivByZeroEx=cms.untracked.bool(bool(options[0])),
                            enableInvalidEx=cms.untracked.bool(bool(options[1])),
                            enableOverflowEx=cms.untracked.bool(bool(options[2])),
                            enableUnderflowEx=cms.untracked.bool(bool(options[3]))
                           )  
    
        return fpe_service


    def prepare(self, doChecking = False):
        """ Prepare the configuration string and add missing pieces."""

        self.addMaxEvents()                    
        self.addSource()
        self.addStandardSequences()
        self.addOutput()
        
        self.pythonCfgCode += "# import of standard configurations\n"
        for module in self.imports:
            self.pythonCfgCode += ("process.load('"+module+"')\n")
        
        # dump max events block
        self.pythonCfgCode += "\nprocess.maxEvents = "+self.process.maxEvents.dumpPython()

        # dump the input definition
        self.pythonCfgCode += "\n# Input source\n"
        self.pythonCfgCode += "process.source = "+self.process.source.dumpPython() #TODO - that needs still definition
        
        # dump the output definition
        self.pythonCfgCode += "\n# Output definition\n"
        self.pythonCfgCode += "process.output = "+self.process.output.dumpPython()

        # dump all additional commands
        self.pythonCfgCode += "\n# Other statements\n"
        for command in self.commands:
            self.pythonCfgCode += command + "\n"
        

          
        # add all paths
        # todo: except for the bad trigger ones
        self.pythonCfgCode += "\n# Path and EndPath definitions\n"
        for path in self.process.paths:
            if 'HLT' not in path:
                self.pythonCfgCode += "process."+path+" = " + getattr(self.process,path).dumpPython("process")
        for endpath in self.process.endpaths:
            self.pythonCfgCode += "process."+endpath+" = " + getattr(self.process,endpath).dumpPython("process")

        # dump the schedule
        self.pythonCfgCode += "\n# Schedule definition\n"
        pathNames = ['process.'+p.label() for p in self.process.schedule]
        result ='process.schedule = cms.Schedule('+','.join(pathNames)+')\n'
        self.pythonCfgCode += result
        #finally put in the customise 
        ## missing 
         
        return
      
      
      
        
        
