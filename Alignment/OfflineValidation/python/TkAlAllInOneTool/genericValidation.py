from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from abc import ABCMeta, abstractmethod, abstractproperty
import os
import re
import json
from . import globalDictionaries
from . import configTemplates
from .dataset import Dataset
from .helperFunctions import replaceByMap, addIndex, getCommandOutput2, boolfromstring, pythonboolstring
from .TkAlExceptions import AllInOneError
from six import with_metaclass

class ValidationMetaClass(ABCMeta):
    sets = ["mandatories", "optionals", "needpackages"]
    dicts = ["defaults"]
    def __new__(cls, clsname, bases, dct):
        for setname in cls.sets:
            if setname not in dct: dct[setname] = set()
            dct[setname] = set.union(dct[setname], *(getattr(base, setname) for base in bases if hasattr(base, setname)))

        for dictname in cls.dicts:
            if dictname not in dct: dct[dictname] = {}
            for base in bases:
                if not hasattr(base, dictname): continue
                newdict = getattr(base, dictname)
                for key in set(newdict) & set(dct[dictname]):
                    if newdict[key] != dct[dictname][key]:
                        raise ValueError("Inconsistent values of defaults[{}]: {}, {}".format(key, newdict[key], dct[dictname][key]))
                dct[dictname].update(newdict)

        for setname in cls.sets:      #e.g. removemandatories, used in preexistingvalidation
                                      #use with caution
            if "remove"+setname not in dct: dct["remove"+setname] = set()
            dct["remove"+setname] = set.union(dct["remove"+setname], *(getattr(base, "remove"+setname) for base in bases if hasattr(base, "remove"+setname)))

            dct[setname] -= dct["remove"+setname]

        return super(ValidationMetaClass, cls).__new__(cls, clsname, bases, dct)

class GenericValidation(with_metaclass(ValidationMetaClass,object)):
    defaultReferenceName = "DEFAULT"
    mandatories = set()
    defaults = {
                "cmssw":        os.environ['CMSSW_BASE'],
                "parallelJobs": "1",
                "jobid":        "",
                "needsproxy":   "false",
               }
    needpackages = {"Alignment/OfflineValidation"}
    optionals = {"jobmode"}

    def __init__(self, valName, alignment, config):
        import random
        self.name = valName
        self.alignmentToValidate = alignment
        self.general = config.getGeneral()
        self.randomWorkdirPart = "%0i"%random.randint(1,10e9)
        self.configFiles = []
        self.config = config
        self.jobid = ""

        theUpdate = config.getResultingSection(self.valType+":"+self.name,
                                               defaultDict = self.defaults,
                                               demandPars = self.mandatories)
        self.general.update(theUpdate)
        self.jobmode = self.general["jobmode"]
        self.NJobs = int(self.general["parallelJobs"])
        self.needsproxy = boolfromstring(self.general["needsproxy"], "needsproxy")

        # limit maximum number of parallel jobs to 40
        # (each output file is approximately 20MB)
        maximumNumberJobs = 40
        if self.NJobs > maximumNumberJobs:
            msg = ("Maximum allowed number of parallel jobs "
                   +str(maximumNumberJobs)+" exceeded!!!")
            raise AllInOneError(msg)
        if self.NJobs > 1 and not isinstance(self, ParallelValidation):
            raise AllInOneError("Parallel jobs not implemented for {}!\n"
                                "Please set parallelJobs = 1.".format(type(self).__name__))

        self.jobid = self.general["jobid"]
        if self.jobid:
            try:  #make sure it's actually a valid jobid
                output = getCommandOutput2("bjobs %(jobid)s 2>&1"%self.general)
                if "is not found" in output: raise RuntimeError
            except RuntimeError:
                raise AllInOneError("%s is not a valid jobid.\nMaybe it finished already?"%self.jobid)

        self.cmssw = self.general["cmssw"]
        badcharacters = r"\'"
        for character in badcharacters:
            if character in self.cmssw:
                raise AllInOneError("The bad characters " + badcharacters + " are not allowed in the cmssw\n"
                                    "path name.  If you really have it in such a ridiculously named location,\n"
                                    "try making a symbolic link somewhere with a decent name.")
        try:
            os.listdir(self.cmssw)
        except OSError:
            raise AllInOneError("Your cmssw release " + self.cmssw + ' does not exist')

        if self.cmssw == os.environ["CMSSW_BASE"]:
            self.scramarch = os.environ["SCRAM_ARCH"]
            self.cmsswreleasebase = os.environ["CMSSW_RELEASE_BASE"]
        else:
            command = ("cd '" + self.cmssw + "' && eval `scramv1 ru -sh 2> /dev/null`"
                       ' && echo "$CMSSW_BASE\n$SCRAM_ARCH\n$CMSSW_RELEASE_BASE"')
            commandoutput = getCommandOutput2(command).split('\n')
            self.cmssw = commandoutput[0]
            self.scramarch = commandoutput[1]
            self.cmsswreleasebase = commandoutput[2]

        self.packages = {}
        for package in self.needpackages:
            for placetolook in self.cmssw, self.cmsswreleasebase:
                pkgpath = os.path.join(placetolook, "src", package)
                if os.path.exists(pkgpath):
                    self.packages[package] = pkgpath
                    break
            else:
                raise AllInOneError("Package {} does not exist in {} or {}!".format(package, self.cmssw, self.cmsswreleasebase))

        self.AutoAlternates = True
        if config.has_option("alternateTemplates","AutoAlternates"):
            try:
                self.AutoAlternates = json.loads(config.get("alternateTemplates","AutoAlternates").lower())
            except ValueError:
                raise AllInOneError("AutoAlternates needs to be true or false, not %s" % config.get("alternateTemplates","AutoAlternates"))

        knownOpts = set(self.defaults.keys())|self.mandatories|self.optionals
        ignoreOpts = []
        config.checkInput(self.valType+":"+self.name,
                          knownSimpleOptions = knownOpts,
                          ignoreOptions = ignoreOpts)

    def getRepMap(self, alignment = None):
        from .plottingOptions import PlottingOptions
        if alignment == None:
            alignment = self.alignmentToValidate
        try:
            result = PlottingOptions(self.config, self.valType)
        except KeyError:
            result = {}
        result.update(alignment.getRepMap())
        result.update(self.general)
        result.update({
                "workdir": os.path.join(self.general["workdir"],
                                        self.randomWorkdirPart),
                "datadir": self.general["datadir"],
                "logdir": self.general["logdir"],
                "CommandLineTemplate": ("#run configfile and post-proccess it\n"
                                        "cmsRun %(cfgFile)s\n"
                                        "%(postProcess)s "),
                "CMSSW_BASE": self.cmssw,
                "SCRAM_ARCH": self.scramarch,
                "CMSSW_RELEASE_BASE": self.cmsswreleasebase,
                "alignmentName": alignment.name,
                "condLoad": alignment.getConditions(),
                "LoadGlobalTagTemplate": configTemplates.loadGlobalTagTemplate,
                })
        result.update(self.packages)
        return result

    @abstractproperty
    def filesToCompare(self):
        pass

    def getCompareStrings( self, requestId = None, plain = False ):
        result = {}
        repMap = self.getRepMap().copy()
        for validationId in self.filesToCompare:
            repMap["file"] = self.filesToCompare[ validationId ]
            if repMap["file"].startswith( "/castor/" ):
                repMap["file"] = "rfio:%(file)s"%repMap
            elif repMap["file"].startswith( "/store/" ):
                repMap["file"] = "root://eoscms.cern.ch//eos/cms%(file)s"%repMap
            if plain:
                result[validationId]=repMap["file"]
            else:
                result[validationId]= "%(file)s=%(title)s|%(color)s|%(style)s"%repMap
        if requestId == None:
            return result
        else:
            if not "." in requestId:
                requestId += ".%s"%self.defaultReferenceName
            if not requestId.split(".")[-1] in result:
                msg = ("could not find %s in reference Objects!"
                       %requestId.split(".")[-1])
                raise AllInOneError(msg)
            return result[ requestId.split(".")[-1] ]

    def createFiles(self, fileContents, path, repMap = None, repMaps = None):
        """repMap: single map for all files
           repMaps: a dict, with the filenames as the keys"""
        if repMap is not None and repMaps is not None:
            raise AllInOneError("createFiles can only take repMap or repMaps (or neither), not both")
        result = []
        for fileName in fileContents:
            filePath = os.path.join(path, fileName)
            result.append(filePath)

            for (i, filePathi) in enumerate(addIndex(filePath, self.NJobs)):
                theFile = open( filePathi, "w" )
                fileContentsi = fileContents[ fileName ]
                if repMaps is not None:
                    repMap = repMaps[fileName]
                if repMap is not None:
                    repMap.update({"nIndex": str(i)})
                    fileContentsi = replaceByMap(fileContentsi, repMap)
                theFile.write( fileContentsi )
                theFile.close()

        return result

    def createConfiguration(self, fileContents, path, schedule = None, repMap = None, repMaps = None):
        self.configFiles = self.createFiles(fileContents,
                                            path, repMap = repMap, repMaps = repMaps)
        if not schedule == None:
            schedule = [os.path.join( path, cfgName) for cfgName in schedule]
            for cfgName in schedule:
                if not cfgName in self.configFiles:
                    msg = ("scheduled %s missing in generated configfiles: %s"
                           %(cfgName, self.configFiles))
                    raise AllInOneError(msg)
            for cfgName in self.configFiles:
                if not cfgName in schedule:
                    msg = ("generated configuration %s not scheduled: %s"
                           %(cfgName, schedule))
                    raise AllInOneError(msg)
            self.configFiles = schedule
        return self.configFiles

    def createScript(self, fileContents, path, downloadFiles=[], repMap = None, repMaps = None):
        self.scriptFiles = self.createFiles(fileContents,
                                            path, repMap = repMap, repMaps = repMaps)
        for script in self.scriptFiles:
            for scriptwithindex in addIndex(script, self.NJobs):
                os.chmod(scriptwithindex,0o755)
        return self.scriptFiles

    def createCrabCfg(self, fileContents, path ):
        if self.NJobs > 1:
            msg =  ("jobmode 'crab' not supported for parallel validation."
                    " Please set parallelJobs = 1.")
            raise AllInOneError(msg)
        self.crabConfigFiles = self.createFiles(fileContents, path)
        return self.crabConfigFiles


class GenericValidationData(GenericValidation):
    """
    Subclass of `GenericValidation` which is the base for validations using
    datasets.
    """
    needParentFiles = False
    mandatories = {"dataset", "maxevents"}
    defaults = {
                "runRange": "",
                "firstRun": "",
                "lastRun": "",
                "begin": "",
                "end": "",
                "JSON": "",
                "dasinstance": "prod/global",
                "ttrhbuilder":"WithAngleAndTemplate",
                "usepixelqualityflag": "True",
               }
    optionals = {"magneticfield"}
    
    def __init__(self, valName, alignment, config):
        """
        This method adds additional items to the `self.general` dictionary
        which are only needed for validations using datasets.
        
        Arguments:
        - `valName`: String which identifies individual validation instances
        - `alignment`: `Alignment` instance to validate
        - `config`: `BetterConfigParser` instance which includes the
                    configuration of the validations
        """

        super(GenericValidationData, self).__init__(valName, alignment, config)

        # if maxevents is not specified, cannot calculate number of events for
        # each parallel job, and therefore running only a single job
        if int( self.general["maxevents"] ) < 0 and self.NJobs > 1:
            msg = ("Maximum number of events (maxevents) not specified: "
                   "cannot use parallel jobs.")
            raise AllInOneError(msg)
        if int( self.general["maxevents"] ) / self.NJobs != float( self.general["maxevents"] ) / self.NJobs:
            msg = ("maxevents has to be divisible by parallelJobs")
            raise AllInOneError(msg)

        tryPredefinedFirst = (not self.jobmode.split( ',' )[0] == "crab" and self.general["JSON"]    == ""
                              and self.general["firstRun"] == ""         and self.general["lastRun"] == ""
                              and self.general["begin"]    == ""         and self.general["end"]     == "")

        if self.general["dataset"] not in globalDictionaries.usedDatasets:
            globalDictionaries.usedDatasets[self.general["dataset"]] = {}

        if self.cmssw not in globalDictionaries.usedDatasets[self.general["dataset"]]:
            if globalDictionaries.usedDatasets[self.general["dataset"]] != {}:
                print(("Warning: you use the same dataset '%s' in multiple cmssw releases.\n"
                       "This is allowed, but make sure it's not a mistake") % self.general["dataset"])
            globalDictionaries.usedDatasets[self.general["dataset"]][self.cmssw] = {False: None, True: None}

        Bfield = self.general.get("magneticfield", None)
        if globalDictionaries.usedDatasets[self.general["dataset"]][self.cmssw][tryPredefinedFirst] is None:
            dataset = Dataset(
                self.general["dataset"], tryPredefinedFirst = tryPredefinedFirst,
                cmssw = self.cmssw, cmsswrelease = self.cmsswreleasebase, magneticfield = Bfield,
                dasinstance = self.general["dasinstance"])
            globalDictionaries.usedDatasets[self.general["dataset"]][self.cmssw][tryPredefinedFirst] = dataset
            if tryPredefinedFirst and not dataset.predefined():                              #No point finding the data twice in that case
                globalDictionaries.usedDatasets[self.general["dataset"]][self.cmssw][False] = dataset

        self.dataset = globalDictionaries.usedDatasets[self.general["dataset"]][self.cmssw][tryPredefinedFirst]
        self.general["magneticField"] = self.dataset.magneticField()
        self.general["defaultMagneticField"] = "MagneticField"
        if self.general["magneticField"] == "unknown":
            print("Could not get the magnetic field for this dataset.")
            print("Using the default: ", self.general["defaultMagneticField"])
            self.general["magneticField"] = '.oO[defaultMagneticField]Oo.'
        
        if not self.jobmode.split( ',' )[0] == "crab":
            try:
                self.general["datasetDefinition"] = self.dataset.datasetSnippet(
                    jsonPath = self.general["JSON"],
                    firstRun = self.general["firstRun"],
                    lastRun = self.general["lastRun"],
                    begin = self.general["begin"],
                    end = self.general["end"],
                    parent = self.needParentFiles )
            except AllInOneError as e:
                msg = "In section [%s:%s]: "%(self.valType, self.name)
                msg += str(e)
                raise AllInOneError(msg)
        else:
            if self.dataset.predefined():
                msg = ("For jobmode 'crab' you cannot use predefined datasets "
                       "(in your case: '%s')."%( self.dataset.name() ))
                raise AllInOneError( msg )
            try:
                theUpdate = config.getResultingSection(self.valType+":"+self.name,
                                                       demandPars = ["parallelJobs"])
            except AllInOneError as e:
                msg = str(e)[:-1]+" when using 'jobmode: crab'."
                raise AllInOneError(msg)
            self.general.update(theUpdate)
            if self.general["begin"] or self.general["end"]:
                ( self.general["begin"],
                  self.general["end"],
                  self.general["firstRun"],
                  self.general["lastRun"] ) = self.dataset.convertTimeToRun(
                    firstRun = self.general["firstRun"],
                    lastRun = self.general["lastRun"],
                    begin = self.general["begin"],
                    end = self.general["end"],
                    shortTuple = False)
                if self.general["begin"] == None:
                    self.general["begin"] = ""
                if self.general["end"] == None:
                    self.general["end"] = ""
                self.general["firstRun"] = str( self.general["firstRun"] )
                self.general["lastRun"] = str( self.general["lastRun"] )
            if ( not self.general["firstRun"] ) and \
                   ( self.general["end"] or self.general["lastRun"] ):
                self.general["firstRun"] = str(
                    self.dataset.runList()[0]["run_number"])
            if ( not self.general["lastRun"] ) and \
                   ( self.general["begin"] or self.general["firstRun"] ):
                self.general["lastRun"] = str(
                    self.dataset.runList()[-1]["run_number"])
            if self.general["firstRun"] and self.general["lastRun"]:
                if int(self.general["firstRun"]) > int(self.general["lastRun"]):
                    msg = ( "The lower time/runrange limit ('begin'/'firstRun') "
                            "chosen is greater than the upper time/runrange limit "
                            "('end'/'lastRun').")
                    raise AllInOneError( msg )
                self.general["runRange"] = (self.general["firstRun"]
                                            + '-' + self.general["lastRun"])
            try:
                self.general["datasetDefinition"] = self.dataset.datasetSnippet(
                    jsonPath = self.general["JSON"],
                    firstRun = self.general["firstRun"],
                    lastRun = self.general["lastRun"],
                    begin = self.general["begin"],
                    end = self.general["end"],
                    crab = True )
            except AllInOneError as e:
                msg = "In section [%s:%s]: "%(self.valType, self.name)
                msg += str( e )
                raise AllInOneError( msg )

        self.general["usepixelqualityflag"] = pythonboolstring(self.general["usepixelqualityflag"], "usepixelqualityflag")

    def getRepMap(self, alignment = None):
        result = super(GenericValidationData, self).getRepMap(alignment)
        outputfile = os.path.expandvars(replaceByMap(
                           "%s_%s_.oO[name]Oo..root" % (self.outputBaseName, self.name)
                                 , result))
        resultfile = os.path.expandvars(replaceByMap(("/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./" +
                           "%s_%s_.oO[name]Oo..root" % (self.resultBaseName, self.name))
                                 , result))
        result.update({
                "resultFile": ".oO[resultFiles[.oO[nIndex]Oo.]]Oo.",
                "resultFiles": addIndex(resultfile, self.NJobs),
                "finalResultFile": resultfile,
                "outputFile": ".oO[outputFiles[.oO[nIndex]Oo.]]Oo.",
                "outputFiles": addIndex(outputfile, self.NJobs),
                "finalOutputFile": outputfile,
                "ProcessName": self.ProcessName,
                "Bookkeeping": self.Bookkeeping,
                "LoadBasicModules": self.LoadBasicModules,
                "TrackSelectionRefitting": self.TrackSelectionRefitting,
                "ValidationConfig": self.ValidationTemplate,
                "FileOutputTemplate": self.FileOutputTemplate,
                "DefinePath": self.DefinePath,
                })
        return result

    @property
    def cfgName(self):
        return "%s.%s.%s_cfg.py"%( self.configBaseName, self.name,
                                   self.alignmentToValidate.name )
    @abstractproperty
    def ProcessName(self):
        pass

    @property
    def cfgTemplate(self):
        return configTemplates.cfgTemplate

    @abstractproperty
    def ValidationTemplate(self):
        pass

    @property
    def filesToCompare(self):
        return {self.defaultReferenceName: self.getRepMap()["finalResultFile"]}

    def createConfiguration(self, path ):
        repMap = self.getRepMap()
        cfgs = {self.cfgName: self.cfgTemplate}
        super(GenericValidationData, self).createConfiguration(cfgs, path, repMap=repMap)

    def createScript(self, path, template = configTemplates.scriptTemplate, downloadFiles=[], repMap = None, repMaps = None):
        scriptName = "%s.%s.%s.sh"%(self.scriptBaseName, self.name,
                                    self.alignmentToValidate.name )
        if repMap is None and repMaps is None:
            repMap = self.getRepMap()
            repMap["CommandLine"]=""
            for cfg in self.configFiles:
                repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":addIndex(cfg, self.NJobs, ".oO[nIndex]Oo."),
                                                      "postProcess":""
                                                     }
        scripts = {scriptName: template}
        return super(GenericValidationData, self).createScript(scripts, path, downloadFiles = downloadFiles,
                                                               repMap = repMap, repMaps = repMaps)

    def createCrabCfg(self, path, crabCfgBaseName):
        """
        Method which creates a `crab.cfg` for a validation on datasets.
        
        Arguments:
        - `path`: Path at which the file will be stored.
        - `crabCfgBaseName`: String which depends on the actual type of
                             validation calling this method.
        """
        crabCfgName = "crab.%s.%s.%s.cfg"%( crabCfgBaseName, self.name,
                                            self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["script"] = "dummy_script.sh"
        # repMap["crabOutputDir"] = os.path.basename( path )
        repMap["crabWorkingDir"] = crabCfgName.split( '.cfg' )[0]
        self.crabWorkingDir = repMap["crabWorkingDir"]
        repMap["numberOfJobs"] = self.general["parallelJobs"]
        repMap["cfgFile"] = self.configFiles[0]
        repMap["queue"] = self.jobmode.split( ',' )[1].split( '-q' )[1]
        if self.dataset.dataType() == "mc":
            repMap["McOrData"] = "events = .oO[nEvents]Oo."
        elif self.dataset.dataType() == "data":
            repMap["McOrData"] = "lumis = -1"
            if self.jobmode.split( ',' )[0] == "crab":
                print ("For jobmode 'crab' the parameter 'maxevents' will be "
                       "ignored and all events will be processed.")
        else:
            raise AllInOneError("Unknown data type!  Can't run in crab mode")
        crabCfg = {crabCfgName: replaceByMap( configTemplates.crabCfgTemplate,
                                              repMap ) }
        return super(GenericValidationData, self).createCrabCfg( crabCfg, path )

    @property
    def Bookkeeping(self):
        return configTemplates.Bookkeeping
    @property
    def LoadBasicModules(self):
        return configTemplates.LoadBasicModules
    @abstractproperty
    def TrackSelectionRefitting(self):
        pass
    @property
    def FileOutputTemplate(self):
        return configTemplates.FileOutputTemplate
    @abstractproperty
    def DefinePath(self):
        pass

class GenericValidationData_CTSR(GenericValidationData):
    #common track selection and refitting
    defaults = {
        "momentumconstraint": "None",
        "openmasswindow": "False",
        "cosmicsdecomode": "True",
        "removetrackhitfiltercommands": "",
        "appendtrackhitfiltercommands": "",
    }
    def getRepMap(self, alignment=None):
        result = super(GenericValidationData_CTSR, self).getRepMap(alignment)

        from .trackSplittingValidation import TrackSplittingValidation
        result.update({
            "ValidationSequence": self.ValidationSequence,
            "istracksplitting": str(isinstance(self, TrackSplittingValidation)),
            "cosmics0T": str(self.cosmics0T),
            "use_d0cut": str(self.use_d0cut),
            "ispvvalidation": str(self.isPVValidation) 
        })

        commands = []
        for removeorappend in "remove", "append":
            optionname = removeorappend + "trackhitfiltercommands"
            if result[optionname]:
                for command in result[optionname].split(","):
                    command = command.strip()
                    commands.append('process.TrackerTrackHitFilter.commands.{}("{}")'.format(removeorappend, command))
        result["trackhitfiltercommands"] = "\n".join(commands)

        return result
    @property
    def use_d0cut(self):
        return "Cosmics" not in self.general["trackcollection"]  #use it for collisions only
    @property
    def isPVValidation(self):
        return False  # only for PV Validation sequence
    @property
    def TrackSelectionRefitting(self):
        return configTemplates.CommonTrackSelectionRefitting
    @property
    def DefinePath(self):
        return configTemplates.DefinePath_CommonSelectionRefitting
    @abstractproperty
    def ValidationSequence(self):
        pass
    @property
    def cosmics0T(self):
        if "Cosmics" not in self.general["trackcollection"]: return False
        Bfield = self.dataset.magneticFieldForRun()
        if Bfield < 0.5: return True
        if isinstance(Bfield, str):
            if "unknown " in Bfield:
                msg = Bfield.replace("unknown ","",1)
            elif Bfield == "unknown":
                msg = "Can't get the B field for %s." % self.dataset.name()
            else:
                msg = "B field = {}???".format(Bfield)
            raise AllInOneError(msg + "\n"
                                "To use this dataset, specify magneticfield = [value] in your .ini config file.")
        return False

class ParallelValidation(GenericValidation):
    @classmethod
    def initMerge(cls):
        return ""
    @abstractmethod
    def appendToMerge(self):
        pass

    @classmethod
    def doInitMerge(cls):
        from .plottingOptions import PlottingOptions
        result = cls.initMerge()
        result = replaceByMap(result, PlottingOptions(None, cls))
        if result and result[-1] != "\n": result += "\n"
        return result
    def doMerge(self):
        result = self.appendToMerge()
        if result[-1] != "\n": result += "\n"
        result += ("if [[ tmpMergeRetCode -eq 0 ]]; then\n"
                   "  xrdcp -f .oO[finalOutputFile]Oo. root://eoscms//eos/cms.oO[finalResultFile]Oo.\n"
                   "fi\n"
                   "if [[ ${tmpMergeRetCode} -gt ${mergeRetCode} ]]; then\n"
                   "  mergeRetCode=${tmpMergeRetCode}\n"
                   "fi\n")
        result = replaceByMap(result, self.getRepMap())
        return result

class ValidationWithPlots(GenericValidation):
    @classmethod
    def runPlots(cls, validations):
        return ("cp .oO[plottingscriptpath]Oo. .\n"
                "root -x -b -q .oO[plottingscriptname]Oo.++")
    @abstractmethod
    def appendToPlots(self):
        pass
    @abstractmethod
    def plottingscriptname(cls):
        """override with a classmethod"""
    @abstractmethod
    def plottingscripttemplate(cls):
        """override with a classmethod"""
    @abstractmethod
    def plotsdirname(cls):
        """override with a classmethod"""

    @classmethod
    def doRunPlots(cls, validations):
        from .plottingOptions import PlottingOptions
        cls.createPlottingScript(validations)
        result = cls.runPlots(validations)
        result = replaceByMap(result, PlottingOptions(None, cls))
        if result and result[-1] != "\n": result += "\n"
        return result
    @classmethod
    def createPlottingScript(cls, validations):
        from .plottingOptions import PlottingOptions
        repmap = PlottingOptions(None, cls).copy()
        filename = replaceByMap(".oO[plottingscriptpath]Oo.", repmap)
        repmap["PlottingInstantiation"] = "\n".join(
                                                    replaceByMap(v.appendToPlots(), v.getRepMap()).rstrip("\n")
                                                         for v in validations
                                                   )
        plottingscript = replaceByMap(cls.plottingscripttemplate(), repmap)
        with open(filename, 'w') as f:
            f.write(plottingscript)

class ValidationWithPlotsSummaryBase(ValidationWithPlots):
    class SummaryItem(object):
        def __init__(self, name, values, format=None, latexname=None, latexformat=None):
            """
            name:        name of the summary item, goes on top of the column
            values:      value for each alignment (in order of rows)
            format:      python format string (default: {:.3g}, meaning up to 3 significant digits)
            latexname:   name in latex form, e.g. if name=sigma you might want latexname=\sigma (default: name)
            latexformat: format for latex (default: format)
            """
            if format is None: format = "{:.3g}"
            if latexname is None: latexname = name
            if latexformat is None: latexformat = format

            self.__name = name
            self.__values = values
            self.__format = format
            self.__latexname = latexname
            self.__latexformat = latexformat

        def name(self, latex=False):
            if latex:
                return self.__latexname
            else:
                return self.__name

        def format(self, value, latex=False):
            if latex:
                fmt = self.__latexformat
            else:
                fmt = self.__format
            if re.match(".*[{][^}]*[fg][}].*", fmt):
                value = float(value)
            return fmt.format(value)

        def values(self, latex=False):
            result = [self.format(v, latex=latex) for v in self.__values]
            return result

        def value(self, i, latex):
            return self.values(latex)[i]

    @abstractmethod
    def getsummaryitems(cls, folder):
        """override with a classmethod that returns a list of SummaryItems
           based on the plots saved in folder"""

    __summaryitems = None
    __lastfolder = None

    @classmethod
    def summaryitemsstring(cls, folder=None, latex=False, transpose=True):
        if folder is None: folder = cls.plotsdirname()
        if folder.startswith( "/castor/" ):
            folder = "rfio:%(file)s"%repMap
        elif folder.startswith( "/store/" ):
            folder = "root://eoscms.cern.ch//eos/cms%(file)s"%repMap

        if cls.__summaryitems is None or cls.__lastfolder != folder:
            cls.__lastfolder = folder
            cls.__summaryitems = cls.getsummaryitems(folder)

        summaryitems = cls.__summaryitems

        if not summaryitems:
            raise AllInOneError("No summary items!")
        size = {len(_.values(latex)) for _ in summaryitems}
        if len(size) != 1:
            raise AllInOneError("Some summary items have different numbers of values\n{}".format(size))
        size = size.pop()

        if transpose:
            columnwidths = ([max(len(_.name(latex)) for _ in summaryitems)]
                          + [max(len(_.value(i, latex)) for _ in summaryitems) for i in range(size)])
        else:
            columnwidths = [max(len(entry) for entry in [_.name(latex)] + _.values(latex)) for _ in summaryitems]

        if latex:
            join = " & "
        else:
            join = " "
        row = join.join("{{:{}}}".format(width) for width in columnwidths)

        if transpose:
            rows = [row.format(*[_.name(latex)]+_.values(latex)) for _ in summaryitems]
        else:
            rows = []
            rows.append(row.format(*(_.name for _ in summaryitems)))
            for i in range(size):
                rows.append(row.format(*(_.value(i, latex) for _ in summaryitems)))

        if latex:
            join = " \\\\\n"
        else:
            join = "\n"
        result = join.join(rows)
        if latex:
            result = (r"\begin{{tabular}}{{{}}}".format("|" + "|".join("c"*(len(columnwidths))) + "|") + "\n"
                         + result + "\n"
                         + r"\end{tabular}")
        return result

    @classmethod
    def printsummaryitems(cls, *args, **kwargs):
        print(cls.summaryitemsstring(*args, **kwargs))
    @classmethod
    def writesummaryitems(cls, filename, *args, **kwargs):
        with open(filename, "w") as f:
            f.write(cls.summaryitemsstring(*args, **kwargs)+"\n")

class ValidationWithPlotsSummary(ValidationWithPlotsSummaryBase):
    @classmethod
    def getsummaryitems(cls, folder):
        result = []
        with open(os.path.join(folder, "{}Summary.txt".format(cls.__name__))) as f:
            for line in f:
                split = line.rstrip("\n").split("\t")
                kwargs = {}
                for thing in split[:]:
                    if thing.startswith("format="):
                        kwargs["format"] = thing.replace("format=", "", 1)
                        split.remove(thing)
                    if thing.startswith("latexname="):
                        kwargs["latexname"] = thing.replace("latexname=", "", 1)
                        split.remove(thing)
                    if thing.startswith("latexformat="):
                        kwargs["latexformat"] = thing.replace("latexformat=", "", 1)
                        split.remove(thing)

                name = split[0]
                values = split[1:]
                result.append(cls.SummaryItem(name, values, **kwargs))
        return result

class ValidationWithComparison(GenericValidation):
    @classmethod
    def doComparison(cls, validations):
        from .plottingOptions import PlottingOptions
        repmap = PlottingOptions(None, cls).copy()
        repmap["compareStrings"] = " , ".join(v.getCompareStrings("OfflineValidation") for v in validations)
        repmap["compareStringsPlain"] = " , ".join(v.getCompareStrings("OfflineValidation", True) for v in validations)
        comparison = replaceByMap(cls.comparisontemplate(), repmap)
        return comparison

    @classmethod
    def comparisontemplate(cls):
        return configTemplates.compareAlignmentsExecution
    @classmethod
    def comparealignmentspath(cls):
        return ".oO[Alignment/OfflineValidation]Oo./scripts/.oO[compareAlignmentsName]Oo."
    @abstractmethod
    def comparealignmentsname(cls):
        """classmethod"""

class ValidationForPresentation(ValidationWithPlots):
    @abstractmethod
    def presentationsubsections(cls):
        """classmethod"""
