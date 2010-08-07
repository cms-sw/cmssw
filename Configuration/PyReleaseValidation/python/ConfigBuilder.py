#! /usr/bin/env python

__version__ = "$Revision: 1.200 $"
__source__ = "$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v $"

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Modules import _Module
import sys
import re

class Options:
        pass

# the canonical defaults
defaultOptions = Options()
defaultOptions.datamix = 'DataOnSim'
defaultOptions.pileup = 'NoPileUp'
defaultOptions.geometry = 'DB'
defaultOptions.geometryExtendedOptions = ['ExtendedGFlash','Extended','NoCastor']
defaultOptions.magField = 'Default'
defaultOptions.conditions = 'auto:startup'
defaultOptions.scenarioOptions=['pp','cosmics','nocoll','HeavyIons']
defaultOptions.harvesting= 'AtRunEnd'
defaultOptions.gflash = False
defaultOptions.himix = False
defaultOptions.number = 0
defaultOptions.arguments = ""
defaultOptions.name = "NO NAME GIVEN"
defaultOptions.evt_type = ""
defaultOptions.filein = []
defaultOptions.customisation_file = ""
defaultOptions.particleTable = 'pythiapdt'
defaultOptions.particleTableList = ['pythiapdt','pdt']

# the pile up map
pileupMap = {'156BxLumiPileUp': 2.0,
             'LowLumiPileUp': 7.1,
             'NoPileUp': 0,
             'InitialPileUp': 3.8,
             'HighLumiPileUp': 25.4
             }

# some helper routines
def dumpPython(process,name):
    theObject = getattr(process,name)
    if isinstance(theObject,cms.Path) or isinstance(theObject,cms.EndPath) or isinstance(theObject,cms.Sequence):
        return "process."+name+" = " + theObject.dumpPython("process")
    elif isinstance(theObject,_Module) or isinstance(theObject,cms.ESProducer):
        return "process."+name+" = " + theObject.dumpPython()


def findName(object,dictionary):
    for name, item in dictionary.iteritems():
        if item == object:
            return name

def availableFileOptions(nameTemplate, path="Configuration/StandardSequences" ):
    """returns existing filenames given a name template"""
    pass


class ConfigBuilder(object):
    """The main building routines """

    def __init__(self, options, process = None, with_output = False, with_input = False ):
        """options taken from old cmsDriver and optparse """

        self._options = options
        self.with_output = with_output
        self.with_input = with_input
        if process == None:
            self.process = cms.Process(self._options.name)
        else:
            self.process = process
        self.imports = []
        self.define_Configs()
        self.schedule = list()

        # we are doing three things here:
        # creating a process to catch errors
        # building the code to re-create the process

        self.additionalCommands = []
        # TODO: maybe a list of to be dumped objects would help as well
        self.blacklist_paths = []
        self.additionalObjects = []
        self.additionalOutputs = {}
        self.productionFilterSequence = None

    def loadAndRemember(self, includeFile):
        """helper routine to load am memorize imports"""
        # we could make the imports a on-the-fly data method of the process instance itself
        # not sure if the latter is a good idea
        includeFile = includeFile.replace('/','.')
        self.imports.append(includeFile)
        self.process.load(includeFile)
        return sys.modules[includeFile]

    def executeAndRemember(self, command):
        """helper routine to remember replace statements"""
        self.additionalCommands.append(command)
        if not command.strip().startswith("#"):
            # substitute: process.foo = process.bar -> self.process.foo = self.process.bar
            exec(command.replace("process.","self.process."))

    def addCommon(self):
        if 'HARVESTING' in self._options.step or 'ALCAHARVEST' in self._options.step:
            self.process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound'),fileMode = cms.untracked.string('FULLMERGE'))
        else:
            self.process.options = cms.untracked.PSet( )

    def addMaxEvents(self):
        """Here we decide how many evts will be processed"""
        self.process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(int(self._options.number)))

    def addSource(self):
        """Here the source is built. Priority: file, generator"""
        if self._options.filein:
           if self._options.filetype == "EDM":
               self.process.source=cms.Source("PoolSource", fileNames = cms.untracked.vstring(self._options.filein))
           elif self._options.filetype == "LHE":
               self.process.source=cms.Source("LHESource", fileNames = cms.untracked.vstring(self._options.filein))
           elif self._options.filetype == "MCDB":
               self.process.source=cms.Source("MCDBSource", articleID = cms.uint32(int(self._options.filein)), supportedProtocols = cms.untracked.vstring("rfio"))

           if 'HARVESTING' in self._options.step or 'ALCAHARVEST' in self._options.step:
               self.process.source.processingMode = cms.untracked.string("RunsAndLumis")

           if self._options.dbsquery!='':
               self.process.source=cms.Source("PoolSource", fileNames = cms.untracked.vstring())
               import os
               print "the query is",self._options.dbsquery
               for line in os.popen('dbs search --query "%s"'%(self._options.dbsquery,)):
                   if (line.find(".root")!=-1):
                       self.process.source.fileNames.append(line.replace("\n",""))
               print "found files: ",self.process.source.fileNames.value()

        if 'GEN' in self._options.step or (not self._options.filein and hasattr(self._options, "evt_type")):
            if self.process.source is None:
                self.process.source=cms.Source("EmptySource")
            # if option himix is active, drop possibly duplicate DIGI-RAW info:
            if self._options.himix==True:
                self.process.source.inputCommands = cms.untracked.vstring('drop *','keep *_generator_*_*','keep *_g4SimHits_*_*')
                self.process.source.dropDescendantsOfDroppedBranches=cms.untracked.bool(False)

            evt_type = re.sub(r'\.py$', '', self._options.evt_type)
            evt_type = evt_type.replace(".","_")
            if "/" in evt_type:
                evt_type = evt_type.replace("python/","")
                evt_type = evt_type.replace("/",".")
            else:
                evt_type = ('Configuration.Generator.'+evt_type).replace('/','.')
            __import__(evt_type)
            generatorModule = sys.modules[evt_type]
            self.process.extend(generatorModule)
            # now add all modules and sequences to the process
            import FWCore.ParameterSet.Modules as cmstypes
            for name in generatorModule.__dict__:
                theObject = getattr(generatorModule,name)
                if isinstance(theObject, cmstypes._Module):
                   self.additionalObjects.insert(0,name)
                if isinstance(theObject, cms.Sequence):
                   self.additionalObjects.append(name)
                if isinstance(theObject, cmstypes.ESProducer):
                   self.additionalObjects.append(name)
        return

    def addOutput(self):
        """ Add output module to the process """

        theEventContent = getattr(self.process, self.eventcontent.split(',')[0]+"EventContent")
        output = cms.OutputModule("PoolOutputModule",
                                  theEventContent,
                                  fileName = cms.untracked.string(self._options.outfile_name),
                                  dataset = cms.untracked.PSet(dataTier = cms.untracked.string(self._options.datatier))
                                 )

        # check if a second (parallel to RECO) output was requested via the eventcontent option
        # this can (for now) only be of type "AOD","AODSIM" or "ALCARECO" and will use the datatier of the same name
        secondOutput = None
        for evtContent in ['AOD', 'AODSIM','ALCARECO']:
            if evtContent in self.eventcontent.split(',') :
                theSecondEventContent = getattr(self.process, evtContent+"EventContent")
                secondOutput = cms.OutputModule("PoolOutputModule",
                                           theSecondEventContent,
                                           fileName = cms.untracked.string(self._options.outfile_name.replace('.root','_secondary.root')),
                                           dataset = cms.untracked.PSet(dataTier = cms.untracked.string(evtContent))
                                           )
                if evtContent=='ALCARECO':
                        secondOutput.dataset.filterName = cms.untracked.string('StreamALCACombined')

        if 'DQM' in self.eventcontent.split(','):
                dqmOutput = cms.OutputModule("PoolOutputModule",
                                             outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*'),
                                             fileName = cms.untracked.string(self._options.dirout+'DQMStream.root'),
                                             dataset = cms.untracked.PSet(filterName = cms.untracked.string(''),dataTier = cms.untracked.string('DQM'))
                                             )
                self.additionalOutputs['DQMStream'] = dqmOutput
                setattr(self.process,'DQMStream',dqmOutput)

        # if there is a generation step in the process, that one should be used as filter decision
        if hasattr(self.process,"generation_step"):
            output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step'))

        # add the filtername
        output.dataset.filterName = cms.untracked.string(self._options.filtername)

        # if the only step is alca we don't need to put in an output
        if not self._options.step.split(',')[0].split(':')[0] == 'ALCA':
            self.process.output = output
            self.process.out_step = cms.EndPath(self.process.output)
            self.schedule.append(self.process.out_step)

            # ATTENTION: major tweaking to avoid inlining of event content
            # should we do that?
            def dummy(instance,label = "process."+self.eventcontent.split(',')[0]+"EventContent.outputCommands"):
                return label

            self.process.output.outputCommands.__dict__["dumpPython"] = dummy
            result = "\n"+self.process.output.dumpPython()

            # now do the same for the second output, if required
            if secondOutput:
                self.process.secondOutput = secondOutput
                self.process.out_stepSecond = cms.EndPath(self.process.secondOutput)
                self.schedule.append(self.process.out_stepSecond)
                # ATTENTION: major tweaking to avoid inlining of event content
                # should we do that?
                # Note: we need to return the _second_ arg from the list of eventcontents ...
                if len(self.eventcontent.split(',')) > 1 :
                    def dummy2(instance,label = "process."+self.eventcontent.split(',')[1]+"EventContent.outputCommands"):
                        return label

                    self.process.secondOutput.outputCommands.__dict__["dumpPython"] = dummy2
                    result += "\n"+self.process.secondOutput.dumpPython()

            return result


    def addStandardSequences(self):
        """
        Add selected standard sequences to the process
        """
        conditionsSP=self._options.conditions.split(',')

        # here we check if we have fastsim or fullsim
        if "FAST" in self._options.step:
            self.loadAndRemember('FastSimulation/Configuration/RandomServiceInitialization_cff')

            # pile up handling for fastsim
            # TODO - do we want a map config - number or actual values?
            if self._options.pileup not in pileupMap.keys():
                    print "Pile up option",self._options.pileup,"unknown."
                    print "Possible options are:", pileupMap.keys()
                    sys.exit(-1)
            else:
                    self.loadAndRemember("FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi")
                    self.loadAndRemember("FastSimulation/Configuration/FamosSequences_cff")
                    self.executeAndRemember('process.famosPileUp.PileUpSimulator = process.PileUpSimulatorBlock.PileUpSimulator')
                    self.executeAndRemember("process.famosPileUp.PileUpSimulator.averageNumber = %s" %pileupMap[self._options.pileup])

        # no fast sim
        else:
            # load the pile up file
            if not self.PileupCFF == '':
                    try:
                        self.loadAndRemember(self.PileupCFF)
                    except ImportError:
                        print "Pile up option",self._options.pileup,"unknown."
                        raise

            # load the geometry file
            try:
                self.loadAndRemember(self.GeometryCFF)
            except ImportError:
                print "Geometry option",self._options.geometry,"unknown."
                raise

        self.loadAndRemember(self.magFieldCFF)


        # what steps are provided by this class?
        stepList = [re.sub(r'^prepare_', '', methodName) for methodName in ConfigBuilder.__dict__ if methodName.startswith('prepare_')]

        ### Benedikt can we add here a check that assure that we are going to generate a correct config file?
        ### i.e. the harvesting do not have to include other step......

        # look which steps are requested and invoke the corresponding method
        for step in self._options.step.split(","):
            if step == "":
                continue
            print step
            stepParts = step.split(":")   # for format STEP:alternativeSequence
            stepName = stepParts[0]
            if stepName not in stepList:
                raise ValueError("Step "+stepName+" unknown")
            if len(stepParts)==1:
                getattr(self,"prepare_"+step)(sequence = getattr(self,step+"DefaultSeq"))
            elif len(stepParts)==2:
                getattr(self,"prepare_"+stepName)(sequence = stepParts[1])
            elif len(stepParts)==3:
                getattr(self,"prepare_"+stepName)(sequence = stepParts[1]+','+stepParts[2])

            else:
                raise ValueError("Step definition "+step+" invalid")

    def addConditions(self):
        """Add conditions to the process"""
        # remove FrontierConditions_GlobalTag only for backwards compatibility
        # the option can be a list of GT name and connection string
        conditions=self._options.conditions.replace("FrontierConditions_GlobalTag,",'').split(',')
        gtName = conditions[0]

        # FULL or FAST SIM ?
        if "FASTSIM" in self._options.step:
            self.loadAndRemember('FastSimulation/Configuration/CommonInputs_cff')

            if "START" in gtName:
                self.executeAndRemember("# Apply ECAL/HCAL miscalibration")
                self.executeAndRemember("process.ecalRecHit.doMiscalib = True")
                self.executeAndRemember("process.hbhereco.doMiscalib = True")
                self.executeAndRemember("process.horeco.doMiscalib = True")
                self.executeAndRemember("process.hfreco.doMiscalib = True")

            # Apply Tracker and Muon misalignment
            self.executeAndRemember("# Apply Tracker and Muon misalignment")
            self.executeAndRemember("process.famosSimHits.ApplyAlignment = True")
            self.executeAndRemember("process.misalignedTrackerGeometry.applyAlignment = True\n")
            self.executeAndRemember("process.misalignedDTGeometry.applyAlignment = True")
            self.executeAndRemember("process.misalignedCSCGeometry.applyAlignment = True\n")

        else:
            self.loadAndRemember(self.ConditionsDefaultCFF)

        # set the global tag
        self.executeAndRemember("process.GlobalTag.globaltag = '"+str(gtName)+"'")
        if len(conditions) > 1:
            self.executeAndRemember("process.GlobalTag.connect = '"+str(conditions[1])+"'")


    def addCustomise(self):
        """Include the customise code """

        # let python search for that package and do syntax checking at the same time
        packageName = self._options.customisation_file.replace(".py","").replace("/",".")
        __import__(packageName)
        package = sys.modules[packageName]

        # now ask the package for its definition and pick .py instead of .pyc
        customiseFile = re.sub(r'\.pyc$', '.py', package.__file__)

        final_snippet='\n\n# Automatic addition of the customisation function\n'
        for line in file(customiseFile,'r'):
            if "import FWCore.ParameterSet.Config" in line:
                continue
            final_snippet += line

        final_snippet += '\n\n# End of customisation function definition'

        return final_snippet + "\n\nprocess = %s(process)\n"%(self._options.cust_function,)

    #----------------------------------------------------------------------------
    # here the methods to define the python includes for each step or
    # conditions
    #----------------------------------------------------------------------------
    def define_Configs(self):
        if ( self._options.scenario not in defaultOptions.scenarioOptions):
                print 'Invalid scenario provided. Options are:'
                print defaultOptions.scenarioOptions
                sys.exit(-1)

        self.loadAndRemember('Configuration/StandardSequences/Services_cff')
        if self._options.particleTable not in defaultOptions.particleTableList:
            print 'Invalid particle table provided. Options are:'
            print defaultOptions.particleTable
            sys.exit(-1)
        else:
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
        self.RECODefaultCFF="Configuration/StandardSequences/Reconstruction_cff"
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

        # synchronize the geometry configuration and the FullSimulation sequence to be used
        if self._options.geometry not in defaultOptions.geometryExtendedOptions:
            self.SIMDefaultCFF="Configuration/StandardSequences/SimIdeal_cff"

        if "DATAMIX" in self._options.step:
            self.DATAMIXDefaultCFF="Configuration/StandardSequences/DataMixer"+self._options.datamix+"_cff"
            self.DIGIDefaultCFF="Configuration/StandardSequences/DigiDM_cff"
            self.DIGI2RAWDefaultCFF="Configuration/StandardSequences/DigiToRawDM_cff"
            self.L1EMDefaultCFF='Configuration/StandardSequences/SimL1EmulatorDM_cff'

        self.ALCADefaultSeq=None
        self.SIMDefaultSeq=None
        self.GENDefaultSeq=None
        self.DIGIDefaultSeq=None
        self.DATAMIXDefaultSeq=None
        self.DIGI2RAWDefaultSeq=None
        self.HLTDefaultSeq=None
        self.L1DefaultSeq=None
        self.HARVESTINGDefaultSeq=None
        self.ALCAHARVESTDefaultSeq=None
        self.CFWRITERDefaultSeq=None
        self.RAW2DIGIDefaultSeq='RawToDigi'
        self.L1RecoDefaultSeq='L1Reco'
        self.RECODefaultSeq='reconstruction'
        self.POSTRECODefaultSeq=None
        self.L1HwValDefaultSeq='L1HwVal'
        self.DQMDefaultSeq='DQMOffline'
        self.FASTSIMDefaultSeq='all'
        self.VALIDATIONDefaultSeq='validation'
        self.PATLayer0DefaultSeq='all'
        self.ENDJOBDefaultSeq='endOfProcess'

        self.EVTCONTDefaultCFF="Configuration/EventContent/EventContent_cff"
        self.defaultMagField='38T'
        self.defaultBeamSpot='Realistic7TeVCollision'

        # if fastsim switch event content
        if "FASTSIM" in self._options.step:
                self.EVTCONTDefaultCFF = "FastSimulation/Configuration/EventContent_cff"

        # if its MC then change the raw2digi
        if self._options.isMC==True:
                self.RAW2DIGIDefaultCFF="Configuration/StandardSequences/RawToDigi_cff"
                self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineMC_cff"
                self.ALCADefaultCFF="Configuration/StandardSequences/AlCaRecoStreamsMC_cff"

        # now for #%#$#! different scenarios

        if self._options.scenario=='nocoll' or self._options.scenario=='cosmics':
            self.SIMDefaultCFF="Configuration/StandardSequences/SimNOBEAM_cff"
            self.defaultBeamSpot='NoSmear'

        if self._options.scenario=='cosmics':
            self.DIGIDefaultCFF="Configuration/StandardSequences/DigiCosmics_cff"
            self.RECODefaultCFF="Configuration/StandardSequences/ReconstructionCosmics_cff"
            self.EVTCONTDefaultCFF="Configuration/EventContent/EventContentCosmics_cff"
            self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineCosmics_cff"
            if self._options.isMC==True:
                self.DQMOFFLINEDefaultCFF="DQMOffline/Configuration/DQMOfflineCosmicsMC_cff"
            self.HARVESTINGDefaultCFF="Configuration/StandardSequences/HarvestingCosmics_cff"
            self.RECODefaultSeq='reconstructionCosmics'
            self.DQMDefaultSeq='DQMOfflineCosmics'
            self.eventcontent='FEVT'

        if self._options.scenario=='HeavyIons':
            self.VALIDATIONDefaultCFF="Configuration/StandardSequences/ValidationHeavyIons_cff"
            self.VALIDATIONDefaultSeq='validationHeavyIons'
            self.EVTCONTDefaultCFF="Configuration/EventContent/EventContentHeavyIons_cff"
            self.RECODefaultCFF="Configuration/StandardSequences/ReconstructionHeavyIons_cff"
            self.RECODefaultSeq='reconstructionHeavyIons'


        # the magnetic field
        if self._options.magField=='Default':
            self._options.magField=self.defaultMagField
        self.magFieldCFF = 'Configuration/StandardSequences/MagneticField_'+self._options.magField.replace('.','')+'_cff'
        self.magFieldCFF = self.magFieldCFF.replace("__",'_')
        if self._options.gflash==True:
                self.GeometryCFF='Configuration/StandardSequences/Geometry'+self._options.geometry+'GFlash_cff'
        else:
                self.GeometryCFF='Configuration/StandardSequences/Geometry'+self._options.geometry+'_cff'

        # Mixing
        if self._options.isMC==True and self._options.himix==False:
            self.PileupCFF='Configuration/StandardSequences/Mixing'+self._options.pileup+'_cff'
        elif self._options.isMC==True and self._options.himix==True:
            self.PileupCFF='Configuration/StandardSequences/HiEventMixing_cff'
        else:
            self.PileupCFF=''

        # beamspot
        if self._options.beamspot != None:
            self.beamspot=self._options.beamspot
        else:
            self.beamspot=self.defaultBeamSpot

        if self._options.eventcontent != None:
            self.eventcontent=self._options.eventcontent

    # for alca, skims, etc
    def addExtraStream(self,name,stream,workflow='full'):
    # define output module and go from there
        output = cms.OutputModule("PoolOutputModule")
        output.SelectEvents = stream.selectEvents
        output.outputCommands = stream.content
        if (hasattr(self._options, 'dirout')):
                output.fileName = cms.untracked.string(self._options.dirout+stream.name+'.root')
        else:
                output.fileName = cms.untracked.string(stream.name+'.root')
        output.dataset  = cms.untracked.PSet( dataTier = stream.dataTier,
                                              filterName = cms.untracked.string(stream.name))
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



    #----------------------------------------------------------------------------
    # here the methods to create the steps. Of course we are doing magic here ;)
    # prepare_STEPNAME modifies self.process and what else's needed.
    #----------------------------------------------------------------------------

    def prepare_ALCAPRODUCER(self, sequence = None):
        self.prepare_ALCA(sequence, workflow = "producers")

    def prepare_ALCAOUTPUT(self, sequence = None):
        self.prepare_ALCA(sequence, workflow = "output")

    def prepare_ALCA(self, sequence = None, workflow = 'full'):
        """ Enrich the process with alca streams """
        if ( len(sequence.split(','))==1 ):
            alcaConfig = self.loadAndRemember(self.ALCADefaultCFF)
        else:
            alcaConfig = self.loadAndRemember(sequence.split(',')[0])
            sequence = sequence.split(',')[1]
        # decide which ALCA paths to use
        alcaList = sequence.split("+")
        for name in alcaConfig.__dict__:
            alcastream = getattr(alcaConfig,name)
            shortName = name.replace('ALCARECOStream','')
            if shortName in alcaList and isinstance(alcastream,cms.FilteredStream):
                self.addExtraStream(name,alcastream, workflow = workflow)
                alcaList.remove(shortName)
            # DQM needs a special handling
            elif name == 'pathALCARECODQM' and 'DQM' in alcaList:
                    path = getattr(alcaConfig,name)
                    self.schedule.append(path)
                    alcaList.remove('DQM')
        if len(alcaList) != 0:
            raise Exception("The following alcas could not be found"+str(alcaList))

    def prepare_GEN(self, sequence = None):
        """ Enrich the schedule with the generation step """
        self.loadAndRemember(self.GENDefaultCFF)

        #check if we are dealing with fastsim -> no vtx smearing
        if "FASTSIM" in self._options.step:
          self.process.pgen.remove(self.process.VertexSmearing)
          self.process.pgen.remove(self.process.GeneInfo)
          self.process.pgen.remove(self.process.genJetMET)
          self.process.generation_step = cms.Path( self.process.pgen)
          self.process.generation_step._seq = self.process.pgen._seq

        # replace the VertexSmearing placeholder by a concrete beamspot definition
        else:
          try:
            self.loadAndRemember('Configuration/StandardSequences/VtxSmeared'+self.beamspot+'_cff')
          except ImportError:
            print "VertexSmearing type or beamspot",self.beamspot, "unknown."
            raise

          if self._options.scenario == 'HeavyIons' and self._options.himix==False:
              self.process.generation_step = cms.Path( self.process.pgen_hi )
          elif self._options.himix==True:
              self.process.generation_step = cms.Path( self.process.pgen_himix )
              self.loadAndRemember("SimGeneral/MixingModule/himixGEN_cff")
          else:
              self.process.generation_step = cms.Path( self.process.pgen )



        self.schedule.append(self.process.generation_step)

        # is there a production filter sequence given?
        if "ProductionFilterSequence" in self.additionalObjects and ("generator" in self.additionalObjects or 'hiSignal' in self.additionalObjects) and sequence == None:
            sequence = "ProductionFilterSequence"
        elif "generator" in self.additionalObjects and sequence == None:
            sequence = "generator"

        if sequence:
            if sequence not in self.additionalObjects:
                raise AttributeError("There is no filter sequence '"+sequence+"' defined in "+self._options.evt_type)
            else:
                self.productionFilterSequence = sequence
        return

    def prepare_SIM(self, sequence = None):
        """ Enrich the schedule with the simulation step"""
        self.loadAndRemember(self.SIMDefaultCFF)
        if self._options.gflash==True:
                             self.loadAndRemember("Configuration/StandardSequences/GFlashSIM_cff")

        if self._options.magField=='0T':
            self.executeAndRemember("process.g4SimHits.UseMagneticField = cms.bool(False)")

        if self._options.himix==True:
            if self._options.geometry in defaultOptions.geometryExtendedOptions:
                self.loadAndRemember("SimGeneral/MixingModule/himixSIMExtended_cff")
            else:
                self.loadAndRemember("SimGeneral/MixingModule/himixSIMIdeal_cff")

        self.process.simulation_step = cms.Path( self.process.psim )
        self.schedule.append(self.process.simulation_step)
        return

    def prepare_DIGI(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
        self.loadAndRemember(self.DIGIDefaultCFF)
        if self._options.gflash==True:
                self.loadAndRemember("Configuration/StandardSequences/GFlashDIGI_cff")

        if self._options.himix==True:
            self.loadAndRemember("SimGeneral/MixingModule/himixDIGI_cff")

        self.process.digitisation_step = cms.Path(self.process.pdigi)
        self.schedule.append(self.process.digitisation_step)
        return
    def prepare_CFWRITER(self, sequence = None):
        """ Enrich the schedule with the crossing frame writer step"""
        self.loadAndRemember(self.CFWRITERDefaultCFF)

        self.process.cfwriter_step = cms.Path(self.process.pcfw)
        self.schedule.append(self.process.cfwriter_step)
        return

    def prepare_DATAMIX(self, sequence = None):
        """ Enrich the schedule with the digitisation step"""
        self.loadAndRemember(self.DATAMIXDefaultCFF)
        self.process.datamixing_step = cms.Path(self.process.pdatamix)
        self.schedule.append(self.process.datamixing_step)
        return

    def prepare_DIGI2RAW(self, sequence = None):
        self.loadAndRemember(self.DIGI2RAWDefaultCFF)
        self.process.digi2raw_step = cms.Path( self.process.DigiToRaw )
        self.schedule.append(self.process.digi2raw_step)
        return

    def prepare_L1(self, sequence = None):
        """ Enrich the schedule with the L1 simulation step"""
        if not sequence:
            self.loadAndRemember(self.L1EMDefaultCFF)
        else:
            # let the L1 package decide for the scenarios available
            from L1Trigger.Configuration.ConfigBuilder import getConfigsForScenario
            listOfImports = getConfigsForScenario(sequence)
            for file in listOfImports:
                self.loadAndRemember(file)
        self.process.L1simulation_step = cms.Path(self.process.SimL1Emulator)
        self.schedule.append(self.process.L1simulation_step)

    def hardCodedTriggerMenuFromConditions(self):
            #horible hack!!! hardwire based on global tag to sync with l1
            if 'MC' in self._options.conditions or 'DESIGN' in self._options.conditions or 'IDEAL' in self._options.conditions:
                    return '1E31'
            else:
                    return '8E29'

    def prepare_HLT(self, sequence = None):
        """ Enrich the schedule with the HLT simulation step"""
        loadDir='HLTrigger'
        fastSim=False
        if 'FASTSIM' in self._options.step:
                fastSim=True
                loadDir='FastSimulation'
        if not sequence:
            menu=self.hardCodedTriggerMenuFromConditions()
            print 'loading %s menu'%(menu,)
            self.loadAndRemember("%s/Configuration/HLT_%s_cff"%(loadDir,menu))
        else:
            # let the HLT package decide for the scenarios available
            listOfImports=[]
            t=__import__('%s.Configuration.ConfigBuilder'%(loadDir,),globals(),locals(),['getConfigsForScenario'])
            listOfImports = t.getConfigsForScenario(sequence)
            for file in listOfImports:
                self.loadAndRemember(file)

        self.schedule.append(self.process.HLTSchedule)
        [self.blacklist_paths.append(path) for path in self.process.HLTSchedule if isinstance(path,(cms.Path,cms.EndPath))]
        if (fastSim and 'HLT' in self._options.step):
                self.finalizeFastSimHLT()


    def prepare_RAW2DIGI(self, sequence = "RawToDigi"):
        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.RAW2DIGIDefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        self.process.raw2digi_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        self.schedule.append(self.process.raw2digi_step)
        return

    def prepare_L1HwVal(self, sequence = 'L1HwVal'):
        ''' Enrich the schedule with L1 HW validation '''
        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.L1HwValDefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        self.process.l1hwval_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        self.schedule.append( self.process.l1hwval_step )
        return

    def prepare_L1Reco(self, sequence = "L1Reco"):
        ''' Enrich the schedule with L1 reconstruction '''
        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.L1RecoDefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        self.process.L1Reco_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        self.schedule.append(self.process.L1Reco_step)
        return

    def prepare_RECO(self, sequence = "reconstruction"):
        ''' Enrich the schedule with reconstruction '''
        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.RECODefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        self.process.reconstruction_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        self.schedule.append(self.process.reconstruction_step)
        return

    def prepare_SKIM(self, sequence = "all"):
        ''' Enrich the schedule with skimming fragments'''
        if (sequence=="all"):
                print "not implemented yet, please specify a list of skims with -s SKIM:skim1+skim2+..."
        else:
                skimlist=sequence.split('+')
                skimConfig = self.loadAndRemember(self.SKIMDefaultCFF)
                #print "dictionnary for skims:",skimConfig.__dict__
                for skim in skimConfig.__dict__:
                        skimstream = getattr(skimConfig,skim)
                        if (not isinstance(skimstream,cms.FilteredStream)):
                                continue
                        shortname = skim.replace('SKIMStream','')
                        if (not shortname in skimlist):
                                continue
                        self.addExtraStream(skim,skimstream)
                        skimlist.remove(shortname)


    def prepare_POSTRECO(self, sequence = None):
        """ Enrich the schedule with the postreco step """
        self.loadAndRemember(self.POSTRECODefaultCFF)
        self.process.postreco_step = cms.Path( self.process.postreco_generator )
        self.schedule.append(self.process.postreco_step)
        return


    def prepare_PATLayer0(self, sequence = None):
        """ In case people would like to have this"""
        pass


    def prepare_VALIDATION(self, sequence = 'validation'):
        if "FASTSIM" in self._options.step:
            self.loadAndRemember("FastSimulation.Configuration.Validation_cff")
            self.process.prevalidation_step = cms.Path( self.process.prevalidation )
            self.schedule.append( self.process.prevalidation_step )
            self.process.validation_step = cms.EndPath( getattr(self.process, sequence.split(',')[-1]) )
            self.schedule.append(self.process.validation_step)
            return
        elif ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.VALIDATIONDefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        self.process.validation_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        if 'genvalid' in sequence.split(',')[-1]:
            self.loadAndRemember("IOMC.RandomEngine.IOMC_cff")
        self.schedule.append(self.process.validation_step)
        print self._options.step
        if not "DIGI"  in self._options.step.split(","):
            self.executeAndRemember("process.mix.playback = True")
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
                    self.doIt(visitee, label)

            def leave(self,visitee):
                    pass


    def prepare_DQM(self, sequence = 'DQMOffline'):
        # this one needs replacement

        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.DQMOFFLINEDefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        self.process.dqmoffline_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )
        self.schedule.append(self.process.dqmoffline_step)

        ## add a specific case of HLT,DQM, need to change the process name to the current process name
        if 'HLT' in self._options.step:
                #look up all module in dqm sequence
                print "replacing process name"
                self.dqmMassaging  = """
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
process.%s.visit(ConfigBuilder.MassSearchReplaceProcessNameVisitor("HLT", "%s", whitelist = ('subSystemFolder',)))
""" % (sequence.split(',')[-1],self.process.name_())

    def prepare_HARVESTING(self, sequence = None):
        """ Enrich the process with harvesting step """
        self.EDMtoMECFF='Configuration/StandardSequences/EDMtoME'+self._options.harvesting+'_cff'
        self.loadAndRemember(self.EDMtoMECFF)
        self.process.edmtome_step = cms.Path(self.process.EDMtoME)
        self.schedule.append(self.process.edmtome_step)

        if ( len(sequence.split(','))==1 ):
            harvestingConfig = self.loadAndRemember(self.HARVESTINGDefaultCFF)
        else:
            harvestingConfig = self.loadAndRemember(sequence.split(',')[0])
            sequence = sequence.split(',')[1]
        # decide which HARVESTING paths to use
        harvestingList = sequence.split("+")
        for name in harvestingConfig.__dict__:
            harvestingstream = getattr(harvestingConfig,name)
            if name in harvestingList and isinstance(harvestingstream,cms.Path):
               self.schedule.append(harvestingstream)
               harvestingList.remove(name)
        # This if statment must disappears once some config happens in the alca harvesting step
        if 'alcaHarvesting' in harvestingList:
            harvestingList.remove('alcaHarvesting')

        if len(harvestingList) != 0 and 'dummyHarvesting' not in harvestingList :
            print "The following harvesting could not be found : ", harvestingList
            raise

        self.process.dqmsave_step = cms.Path(self.process.DQMSaver)
        self.schedule.append(self.process.dqmsave_step)


    def prepare_ALCAHARVEST(self, sequence = None):
        """ Enrich the process with AlCaHarvesting step """
        harvestingConfig = self.loadAndRemember(self.ALCAHARVESTDefaultCFF)
        # decide which AlcaHARVESTING paths to use
        harvestingList = sequence.split("+")
        for name in harvestingConfig.__dict__:
            harvestingstream = getattr(harvestingConfig,name)
            if name in harvestingList and isinstance(harvestingstream,cms.Path):
               self.schedule.append(harvestingstream)
               self.executeAndRemember("process.PoolDBOutputService.toPut.append(process.ALCAHARVEST" + name + "_dbOutput)")
               harvestingList.remove(name)

        if len(harvestingList) != 0 and 'dummyHarvesting' not in harvestingList :
            print "The following harvesting could not be found : ", harvestingList
            raise



    def prepare_ENDJOB(self, sequence = 'endOfProcess'):
        # this one needs replacement

        if ( len(sequence.split(','))==1 ):
            self.loadAndRemember(self.ENDJOBDefaultCFF)
        else:
            self.loadAndRemember(sequence.split(',')[0])
        if "FASTSIM" in self._options.step:
            self.process.endjob_step = cms.EndPath( getattr(self.process, sequence.split(',')[-1]) )
        else:
            self.process.endjob_step = cms.Path( getattr(self.process, sequence.split(',')[-1]) )

        self.schedule.append(self.process.endjob_step)

    def finalizeFastSimHLT(self):
            self.process.HLTSchedule.remove(self.process.HLTAnalyzerEndpath)
            self.process.reconstruction = cms.Path(self.process.reconstructionWithFamos)
            self.schedule.append(self.process.reconstruction)

    def prepare_FASTSIM(self, sequence = "all"):
        """Enrich the schedule with fastsim"""
        self.loadAndRemember("FastSimulation/Configuration/FamosSequences_cff")

        if sequence in ('all','allWithHLTFiltering',''):
            if not 'HLT' in self._options.step:
                    self.prepare_HLT(sequence=None)

            self.executeAndRemember("process.famosSimHits.SimulateCalorimetry = True")
            self.executeAndRemember("process.famosSimHits.SimulateTracking = True")

            # the settings have to be the same as for the generator to stay consistent
            print '  The pile up is taken from 7 TeV files. To switch to other files remove the inclusion of "PileUpSimulator7TeV_cfi"'

            self.executeAndRemember("process.simulation = cms.Sequence(process.simulationWithFamos)")
            self.executeAndRemember("process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)")

            # since we have HLT here, the process should be called HLT
            self._options.name = "HLT"

            # if we don't want to filter after HLT but simulate everything regardless of what HLT tells, we have to add reconstruction explicitly
            if sequence == 'all' and not 'HLT' in self._options.step: #(a)
                self.finalizeFastSimHLT()
        elif sequence == 'famosWithEverything':
            self.process.fastsim_step = cms.Path( getattr(self.process, "famosWithEverything") )
            self.schedule.append(self.process.fastsim_step)

            # now the additional commands we need to make the config work
            self.executeAndRemember("process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True")
        else:
             print "FastSim setting", sequence, "unknown."
             raise ValueError
        # the vertex smearing settings
        beamspotName = 'process.%sVtxSmearingParameters' %(self.beamspot)
        if 'Flat' in self.beamspot:
            beamspotType = 'Flat'
        elif 'Gauss' in self.beamspot:
            beamspotType = 'Gaussian'
        else:
            print "  Assuming vertex smearing engine as BetaFunc"
            beamspotType = 'BetaFunc'
        self.loadAndRemember('IOMC.EventVertexGenerators.VtxSmearedParameters_cfi')
        beamspotName = 'process.%sVtxSmearingParameters' %(self.beamspot)
        self.executeAndRemember('\n# set correct vertex smearing')
        self.executeAndRemember(beamspotName+'.type = cms.string("%s")'%(beamspotType))
        self.executeAndRemember('process.famosSimHits.VertexGenerator = '+beamspotName)
        self.executeAndRemember('process.famosPileUp.VertexGenerator = '+beamspotName)

    def build_production_info(self, evt_type, evtnumber):
        """ Add useful info for the production. """
        prod_info=cms.untracked.PSet\
              (version=cms.untracked.string("$Revision: 1.200 $"),
               name=cms.untracked.string("PyReleaseValidation"),
               annotation=cms.untracked.string(evt_type+ " nevts:"+str(evtnumber))
              )

        return prod_info


    def prepare(self, doChecking = False):
        """ Prepare the configuration string and add missing pieces."""

        self.addMaxEvents()
        if self.with_input:
           self.addSource()
        self.addStandardSequences()
        self.addConditions()
        self.loadAndRemember(self.EVTCONTDefaultCFF)  #load the event contents regardless

        if not 'HARVESTING' in self._options.step and not 'SKIM' in self._options.step and not 'ALCAHARVEST' in self._options.step and not 'ALCAOUTPUT' in self._options.step and self.with_output:
            self.addOutput()

        self.addCommon()

        self.pythonCfgCode =  "# Auto generated configuration file\n"
        self.pythonCfgCode += "# using: \n# "+__version__[1:-1]+"\n# "+__source__[1:-1]+'\n'
        self.pythonCfgCode += "# with command line options: "+self._options.arguments+'\n'
        self.pythonCfgCode += "import FWCore.ParameterSet.Config as cms\n\n"
        self.pythonCfgCode += "process = cms.Process('"+self._options.name+"')\n\n"

        self.pythonCfgCode += "# import of standard configurations\n"
        for module in self.imports:
            self.pythonCfgCode += ("process.load('"+module+"')\n")

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
        if hasattr(self.process,"output"):
            self.pythonCfgCode += "\n# Output definition\n"
            self.pythonCfgCode += "process.output = "+self.process.output.dumpPython()
        if hasattr(self.process,"secondOutput"):
            self.pythonCfgCode += "\n# Second Output definition\n"
            self.pythonCfgCode += "process.secondOutput = "+self.process.secondOutput.dumpPython()

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

        # special treatment for a production filter sequence 1/2
        if self.productionFilterSequence:
            # dump all additional definitions from the input definition file
            for name in self.additionalObjects:
                self.pythonCfgCode += dumpPython(self.process,name)

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

        if hasattr(self,"dqmMassaging"):
                result += self.dqmMassaging

        self.pythonCfgCode += result

        # special treatment in case of production filter sequence 2/2
        if self.productionFilterSequence:
                modifierCode = """
# special treatment in case of production filter sequence
for path in process.paths: \n    getattr(process,path)._seq = process."""+self.productionFilterSequence+"""*getattr(process,path)._seq
"""
                self.pythonCfgCode += modifierCode

        # dump customise fragment
        if self._options.customisation_file:
            self.pythonCfgCode += self.addCustomise()
        return


def installFilteredStream(process, schedule, streamName, definitionFile = "Configuration/StandardSequences/AlCaRecoStreams_cff" ):

    __import__(definitionFile)
    definitionModule = sys.modules[definitionFile]
    process.extend(definitionModule)
    stream = getattr(definitionModule,streamName)
    output = cms.OutputModule("PoolOutputModule")
    output.SelectEvents = stream.selectEvents
    output.outputCommands = stream.content
    output.dataset  = cms.untracked.PSet( dataTier = stream.dataTier)
    setattr(process,streamName,output)
    for path in stream.paths:
        schedule.append(path)


def installPromptReco(process, recoOutputModule, aodOutputModule = None):
    """
    _promptReco_

    Method to install the standard PromptReco configuration into
    a basic process containing source and output modules.

    process is the CMS Process instance to be populated

    recoOutputModule is the output module used to write the
    RECO data tier

    aodOutputModule is the output module used to write
    the AOD data tier, if this is not none, any AOD sequences
    should be added.
    """
    cb = ConfigBuilder(defaultOptions, process = process)
    cb._options.step = 'RAW2DIGI,RECO'
    cb.addStandardSequences()
    cb.addConditions()
    process.load(cb.EVTCONTDefault)
    recoOutputModule.eventContent = process.RECOEventContent
    if aodOutputModule != None:
        aodOutputModule.eventContent = process.AODEventContent
    return process


promptReco = installPromptReco


def addOutputModule(process, tier, content):
    """
    _addOutputModule_

    Function to add an output module to a given process with given data tier and event content
    """
    moduleName = "output%s%s" % (tier, content)
    pathName = "%sPath" % moduleName
    contentName = "%sEventContent" % content
    contentAttr = getattr(process, contentName)
    setattr(process, moduleName,
            cms.OutputModule("PoolOutputModule",
                              fileName = cms.untracked.string('%s.root' % moduleName),
                              dataset = cms.untracked.PSet(
                                dataTier = cms.untracked.string(tier),
                              ),
                              eventContent = contentAttr
                           )
            )
    print getattr(process,moduleName)
    # put it in an EndPath and put the EndPath into the schedule
    setattr(process, pathName, cms.EndPath(getattr(process,moduleName)) )
    process.schedule.append(getattr(process, pathName))

    return


def addALCAPaths(process, listOfALCANames, definitionFile = "Configuration/StandardSequences/AlCaRecoStreams_cff"):
    """
    _addALCAPaths_

    Function to add alignment&calibration sequences to an existing process
    """
    __import__(definitionFile)
    definitionModule = sys.modules[definitionFile]
    process.extend(definitionModule)

    for alca in listOfALCANames:
       streamName = "ALCARECOStream%s" % alca
       stream = getattr(definitionModule, streamName)
       for path in stream.paths:
            schedule.append(path)

    return
