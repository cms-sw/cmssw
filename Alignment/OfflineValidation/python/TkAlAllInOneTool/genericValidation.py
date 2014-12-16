import os
import re
import json
import globalDictionaries
import configTemplates
from dataset import Dataset
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class GenericValidation:
    defaultReferenceName = "DEFAULT"
    def __init__(self, valName, alignment, config, valType,
                 addDefaults = {}, addMandatories=[]):
        import random
        self.name = valName
        self.alignmentToValidate = alignment
        self.general = config.getGeneral()
        self.randomWorkdirPart = "%0i"%random.randint(1,10e9)
        self.configFiles = []
        self.filesToCompare = {}
        self.config = config

        defaults = {"jobmode": self.general["jobmode"],
                    "cmssw":   os.environ['CMSSW_BASE']
                   }
        defaults.update(addDefaults)
        mandatories = []
        mandatories += addMandatories
        theUpdate = config.getResultingSection(valType+":"+self.name,
                                               defaultDict = defaults,
                                               demandPars = mandatories)
        self.general.update(theUpdate)
        self.jobmode = self.general["jobmode"]
        try:
            self.cmssw = self.general["cmssw"]
            currentrelease = os.path.basename(os.path.normpath(os.environ['CMSSW_BASE']))
            newrelease = os.path.basename(os.path.normpath(self.cmssw))

            self.scramarch = None
            for sa in os.listdir(os.path.join(self.cmssw, "bin")):
                if re.match("slc[0-9]+_amd[0-9]+_gcc[0-9]+",sa):
                    try:
                        crb = os.environ['CMSSW_RELEASE_BASE'] \
                                  .replace(currentrelease, newrelease) \
                                  .replace(os.environ['SCRAM_ARCH'],sa)
                        if "patch" in newrelease:
                            crb = crb.replace("cms/cmssw/","cms/cmssw-patch/")
                        else:
                            crb = crb.replace("cms/cmssw-patch/","cms/cmssw/")
                        os.listdir(crb)
                        self.scramarch = sa
                        self.cmsswreleasebase = crb
                        break
                    except OSError:
                        pass
            if self.scramarch is None:
                raise OSError
        except OSError:
                msg = ("Your CMSSW release %s does not exist or is not set up properly.\n"
                       "If you're sure it's there, run scram b and then try again." % self.cmssw)
                raise AllInOneError(msg)

        self.AutoAlternates = True
        if config.has_option("alternateTemplates","AutoAlternates"):
            try:
                self.AutoAlternates = json.loads(config.get("alternateTemplates","AutoAlternates").lower())
            except ValueError:
                raise AllInOneError("AutoAlternates needs to be true or false, not %s" % config.get("alternateTemplates","AutoAlternates"))

        knownOpts = defaults.keys()+mandatories
        ignoreOpts = []
        if self.jobmode.split(",")[0] == "crab" \
                or self.__class__.__name__=="OfflineValidationParallel":
            knownOpts.append("parallelJobs")
        else:
            ignoreOpts.append("parallelJobs")
        config.checkInput(valType+":"+self.name,
                          knownSimpleOptions = knownOpts,
                          ignoreOptions = ignoreOpts)

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        result = alignment.getRepMap()
        result.update( self.general )
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
                "condLoad": alignment.getConditions()
                })
        return result

    def getCompareStrings( self, requestId = None, plain = False ):
        result = {}
        repMap = self.alignmentToValidate.getRepMap()
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
                requestId += ".%s"%GenericValidation.defaultReferenceName
            if not requestId.split(".")[-1] in result:
                msg = ("could not find %s in reference Objects!"
                       %requestId.split(".")[-1])
                raise AllInOneError(msg)
            return result[ requestId.split(".")[-1] ]

    def createFiles( self, fileContents, path ):
        result = []
        for fileName in fileContents:
            filePath = os.path.join( path, fileName)
            theFile = open( filePath, "w" )
            theFile.write( fileContents[ fileName ] )
            theFile.close()
            result.append( filePath )
        return result

    def createConfiguration(self, fileContents, path, schedule= None):
        self.configFiles = GenericValidation.createFiles(self, fileContents,
                                                         path) 
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

    def createScript(self, fileContents, path, downloadFiles=[] ):        
        self.scriptFiles = GenericValidation.createFiles(self, fileContents,
                                                         path)
        for script in self.scriptFiles:
            os.chmod(script,0755)
        return self.scriptFiles

    def createCrabCfg(self, fileContents, path ):        
        self.crabConfigFiles = GenericValidation.createFiles(self, fileContents,
                                                             path)
        return self.crabConfigFiles


class GenericValidationData(GenericValidation):
    """
    Subclass of `GenericValidation` which is the base for validations using
    datasets.
    """
    
    def __init__(self, valName, alignment, config, valType,
                 addDefaults = {}, addMandatories=[]):
        """
        This method adds additional items to the `self.general` dictionary
        which are only needed for validations using datasets.
        
        Arguments:
        - `valName`: String which identifies individual validation instances
        - `alignment`: `Alignment` instance to validate
        - `config`: `BetterConfigParser` instance which includes the
                    configuration of the validations
        - `valType`: String which specifies the type of validation
        - `addDefaults`: Dictionary which contains default values for individual
                         validations in addition to the general default values
        - `addMandatories`: List which contains mandatory parameters for
                            individual validations in addition to the general
                            mandatory parameters
        """

        defaults = {"runRange": "",
                    "firstRun": "",
                    "lastRun": "",
                    "begin": "",
                    "end": "",
                    "JSON": ""
                    }
        defaults.update(addDefaults)
        mandatories = [ "dataset", "maxevents" ]
        mandatories += addMandatories
        GenericValidation.__init__(self, valName, alignment, config, valType, defaults, mandatories)

        tryPredefinedFirst = (not self.jobmode.split( ',' )[0] == "crab" and self.general["JSON"]    == ""
                              and self.general["firstRun"] == ""         and self.general["lastRun"] == ""
                              and self.general["begin"]    == ""         and self.general["end"]     == "")
        if self.general["dataset"] not in globalDictionaries.usedDatasets:
            globalDictionaries.usedDatasets[self.general["dataset"]] = [None, None]
        if globalDictionaries.usedDatasets[self.general["dataset"]][tryPredefinedFirst] is None:
            dataset = Dataset(
                self.general["dataset"], tryPredefinedFirst = tryPredefinedFirst,
                cmssw = self.getRepMap()["CMSSW_BASE"], cmsswrelease = self.getRepMap()["CMSSW_RELEASE_BASE"] )
            globalDictionaries.usedDatasets[self.general["dataset"]][tryPredefinedFirst] = dataset
            if tryPredefinedFirst and not dataset.predefined():                              #No point finding the data twice in that case
                globalDictionaries.usedDatasets[self.general["dataset"]][False] = dataset

        self.dataset = globalDictionaries.usedDatasets[self.general["dataset"]][tryPredefinedFirst]
        self.general["magneticField"] = self.dataset.magneticField()
        self.general["defaultMagneticField"] = "38T"
        if self.general["magneticField"] == "unknown":
            print "Could not get the magnetic field for this dataset."
            print "Using the default: ", self.general["defaultMagneticField"]
            self.general["magneticField"] = '.oO[defaultMagneticField]Oo.'
        
        if not self.jobmode.split( ',' )[0] == "crab":
            try:
                self.general["datasetDefinition"] = self.dataset.datasetSnippet(
                    jsonPath = self.general["JSON"],
                    nEvents = self.general["maxevents"],
                    firstRun = self.general["firstRun"],
                    lastRun = self.general["lastRun"],
                    begin = self.general["begin"],
                    end = self.general["end"] )
            except AllInOneError, e:
                msg = "In section [%s:%s]: "%(valType, self.name)
                msg += str(e)
                raise AllInOneError(msg)
        else:
            if self.dataset.predefined():
                msg = ("For jobmode 'crab' you cannot use predefined datasets "
                       "(in your case: '%s')."%( self.dataset.name() ))
                raise AllInOneError( msg )
            try:
                theUpdate = config.getResultingSection(valType+":"+self.name,
                                                       demandPars = ["parallelJobs"])
            except AllInOneError, e:
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
                    nEvents = self.general["maxevents"],
                    firstRun = self.general["firstRun"],
                    lastRun = self.general["lastRun"],
                    begin = self.general["begin"],
                    end = self.general["end"],
                    crab = True )
            except AllInOneError, e:
                msg = "In section [%s:%s]: "%(valType, self.name)
                msg += str( e )
                raise AllInOneError( msg )

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
        return GenericValidation.createCrabCfg( self, crabCfg, path )
