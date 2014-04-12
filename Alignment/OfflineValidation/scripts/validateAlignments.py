#!/usr/bin/env python
#test execute: export CMSSW_BASE=/tmp/CMSSW && ./validateAlignments.py -c defaultCRAFTValidation.ini,test.ini -n -N test
import os
import sys
import optparse
import datetime
import shutil
import fnmatch

import Alignment.OfflineValidation.TkAlAllInOneTool.configTemplates \
    as configTemplates
import Alignment.OfflineValidation.TkAlAllInOneTool.crabWrapper as crabWrapper
from Alignment.OfflineValidation.TkAlAllInOneTool.TkAlExceptions \
    import AllInOneError
from Alignment.OfflineValidation.TkAlAllInOneTool.helperFunctions \
    import replaceByMap, getCommandOutput2
from Alignment.OfflineValidation.TkAlAllInOneTool.betterConfigParser \
    import BetterConfigParser
from Alignment.OfflineValidation.TkAlAllInOneTool.alignment import Alignment

from Alignment.OfflineValidation.TkAlAllInOneTool.genericValidation \
    import GenericValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.geometryComparison \
    import GeometryComparison
from Alignment.OfflineValidation.TkAlAllInOneTool.offlineValidation \
    import OfflineValidation, OfflineValidationDQM, OfflineValidationParallel
from Alignment.OfflineValidation.TkAlAllInOneTool.monteCarloValidation \
    import MonteCarloValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.trackSplittingValidation \
    import TrackSplittingValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.zMuMuValidation \
    import ZMuMuValidation
import Alignment.OfflineValidation.TkAlAllInOneTool.globalDictionaries \
    as globalDictionaries


####################--- Classes ---############################
class ValidationJob:
    def __init__( self, validation, config, options ):
        if validation[1] == "":
            # intermediate syntax
            valString = validation[0].split( "->" )[0]
            alignments = validation[0].split( "->" )[1]
            # force user to use the normal syntax
            if "->" in validation[0]:
                msg = ("Instead of using the intermediate syntax\n'"
                       +valString.strip()+"-> "+alignments.strip()
                       +":'\nyou have to use the now fully supported syntax \n'"
                       +valString.strip()+": "
                       +alignments.strip()+"'.")
                raise AllInOneError(msg)
        else:
            valString = validation[0]
            alignments = validation[1]
        valString = valString.split()
        self.__valType = valString[0]
        self.__valName = valString[1]
        self.__commandLineOptions = options
        self.__config = config
        # workaround for intermediate parallel version
        if self.__valType == "offlineParallel":
            section = "offline" + ":" + self.__valName
        else:
            section = self.__valType + ":" + self.__valName
        if not self.__config.has_section( section ):
            raise AllInOneError, ("Validation '%s' of type '%s' is requested in"
                                  " '[validation]' section, but is not defined."
                                  "\nYou have to add a '[%s]' section."
                                  %( self.__valName, self.__valType, section ))
        self.validation = self.__getValidation( self.__valType, self.__valName,
                                                alignments, self.__config,
                                                options )

    def __getValidation( self, valType, name, alignments, config, options ):
        if valType == "compare":
            alignmentsList = alignments.split( "," )
            firstAlignList = alignmentsList[0].split()
            firstAlignName = firstAlignList[0].strip()
            if firstAlignName == "IDEAL":
                raise AllInOneError, ("'IDEAL' has to be the second (reference)"
                                      " alignment in 'compare <val_name>: "
                                      "<alignment> <reference>'.")
            if len( firstAlignList ) > 1:
                firstRun = firstAlignList[1]
            else:
                firstRun = "1"
            firstAlign = Alignment( firstAlignName, self.__config, firstRun )
            secondAlignList = alignmentsList[1].split()
            secondAlignName = secondAlignList[0].strip()
            if len( secondAlignList ) > 1:
                secondRun = secondAlignList[1]
            else:
                secondRun = "1"
            if secondAlignName == "IDEAL":
                secondAlign = secondAlignName
            else:
                secondAlign = Alignment( secondAlignName, self.__config,
                                         secondRun )
            # check if alignment was already compared previously
            try:
                randomWorkdirPart = \
                    globalDictionaries.alignRandDict[firstAlignName]
            except KeyError:
                randomWorkdirPart = None
                
            validation = GeometryComparison( name, firstAlign, secondAlign,
                                             self.__config,
                                             self.__commandLineOptions.getImages,
                                             randomWorkdirPart )
            globalDictionaries.alignRandDict[firstAlignName] = \
                validation.randomWorkdirPart
            if not secondAlignName == "IDEAL":
                globalDictionaries.alignRandDict[secondAlignName] = \
                    validation.randomWorkdirPart
        elif valType == "offline":
            validation = OfflineValidation( name, 
                Alignment( alignments.strip(), self.__config ), self.__config )
        elif valType == "offlineDQM":
            validation = OfflineValidationDQM( name, 
                Alignment( alignments.strip(), self.__config ), self.__config )
        elif valType == "offlineParallel":
            validation = OfflineValidationParallel( name, 
                Alignment( alignments.strip(), self.__config ), self.__config )
        elif valType == "mcValidate":
            validation = MonteCarloValidation( name, 
                Alignment( alignments.strip(), self.__config ), self.__config )
        elif valType == "split":
            validation = TrackSplittingValidation( name, 
                Alignment( alignments.strip(), self.__config ), self.__config )
        elif valType == "zmumu":
            validation = ZMuMuValidation( name, 
                Alignment( alignments.strip(), self.__config ), self.__config )
        else:
            raise AllInOneError, "Unknown validation mode '%s'"%valType
        return validation

    def __createJob( self, jobMode, outpath ):
        """This private method creates the needed files for the validation job.
           """
        self.validation.createConfiguration( outpath )
        self.__scripts = self.validation.createScript( outpath )
        if jobMode.split( ',' )[0] == "crab":
            self.validation.createCrabCfg( outpath )
        return None

    def createJob(self):
        """This is the method called to create the job files."""
        self.__createJob( self.validation.jobmode,
                          os.path.abspath( self.__commandLineOptions.Name) )

    def runJob( self ):
        general = self.__config.getGeneral()
        log = ""
        for script in self.__scripts:
            name = os.path.splitext( os.path.basename( script) )[0]
            if self.__commandLineOptions.dryRun:
                print "%s would run: %s"%( name, os.path.basename( script) )
                continue
            log = ">             Validating "+name
            print ">             Validating "+name
            if self.validation.jobmode == "interactive":
                log += getCommandOutput2( script )
            elif self.validation.jobmode.split(",")[0] == "lxBatch":
                repMap = { 
                    "commands": self.validation.jobmode.split(",")[1],
                    "logDir": general["logdir"],
                    "jobName": name,
                    "script": script,
                    "bsub": "/afs/cern.ch/cms/caf/scripts/cmsbsub"
                    }
                log+=getCommandOutput2("%(bsub)s %(commands)s -J %(jobName)s "
                                       "-o %(logDir)s/%(jobName)s.stdout -e "
                                       "%(logDir)s/%(jobName)s.stderr "
                                       "%(script)s"%repMap)
            elif self.validation.jobmode.split( "," )[0] == "crab":
                os.chdir( general["logdir"] )
                crabName = "crab." + os.path.basename( script )[:-3]
                theCrab = crabWrapper.CrabWrapper()
                options = { "-create": "",
                            "-cfg": crabName + ".cfg",
                            "-submit": "" }
                try:
                    theCrab.run( options )
                except AllInOneError, e:
                    print "crab:", str(e).split("\n")[0]
                    exit(1)
            else:
                raise AllInOneError, ("Unknown 'jobmode'!\n"
                                      "Please change this parameter either in "
                                      "the [general] or in the ["
                                      + self.__valType + ":" + self.__valName
                                      + "] section to one of the following "
                                      "values:\n"
                                      "\tinteractive\n\tlxBatch, -q <queue>\n"
                                      "\tcrab, -q <queue>")
        return log

    def getValidation( self ):
        return self.validation


####################--- Functions ---############################
def createOfflineJobsMergeScript(offlineValidationList, outFilePath):
    repMap = offlineValidationList[0].getRepMap() # bit ugly since some special features are filled
    repMap[ "mergeOfflinParJobsInstantiation" ] = "" #give it a "" at first in order to get the initialisation back
    
    theFile = open( outFilePath, "w" )
    theFile.write( replaceByMap( configTemplates.mergeOfflineParJobsTemplate ,repMap ) )
    theFile.close()

def createExtendedValidationScript(offlineValidationList, outFilePath):
    repMap = offlineValidationList[0].getRepMap() # bit ugly since some special features are filled
    repMap[ "extendedInstantiation" ] = "" #give it a "" at first in order to get the initialisation back

    for validation in offlineValidationList:
        repMap[ "extendedInstantiation" ] = validation.appendToExtendedValidation( repMap[ "extendedInstantiation" ] )
    
    theFile = open( outFilePath, "w" )
    # theFile.write( replaceByMap( configTemplates.extendedValidationTemplate ,repMap ) )
    theFile.write( replaceByMap( configTemplates.extendedValidationTemplate ,repMap ) )
    theFile.close()
    
def createMergeScript( path, validations ):
    if(len(validations) == 0):
        msg = "Cowardly refusing to merge nothing!"
        raise AllInOneError(msg)

    repMap = validations[0].getRepMap() #FIXME - not nice this way
    repMap.update({
            "DownloadData":"",
            "CompareAlignments":"",
            "RunExtendedOfflineValidation":""
            })

    comparisonLists = {} # directory of lists containing the validations that are comparable
    for validation in validations:
        for referenceName in validation.filesToCompare:
            validationName = "%s.%s"%(validation.__class__.__name__, referenceName)
            validationName = validationName.split(".%s"%GenericValidation.defaultReferenceName )[0]
            if validationName in comparisonLists:
                comparisonLists[ validationName ].append( validation )
            else:
                comparisonLists[ validationName ] = [ validation ]

    if "OfflineValidation" in comparisonLists:
        repMap["extendeValScriptPath"] = \
            os.path.join(path, "TkAlExtendedOfflineValidation.C")
        createExtendedValidationScript(comparisonLists["OfflineValidation"],
                                       repMap["extendeValScriptPath"] )
        repMap["RunExtendedOfflineValidation"] = \
            replaceByMap(configTemplates.extendedValidationExecution, repMap)

    repMap["CompareAlignments"] = "#run comparisons"
    for validationId in comparisonLists:
        compareStrings = [ val.getCompareStrings(validationId) for val in comparisonLists[validationId] ]
            
        repMap.update({"validationId": validationId,
                       "compareStrings": " , ".join(compareStrings) })
        
        repMap["CompareAlignments"] += \
            replaceByMap(configTemplates.compareAlignmentsExecution, repMap)
      
    filePath = os.path.join(path, "TkAlMerge.sh")
    theFile = open( filePath, "w" )
    theFile.write( replaceByMap( configTemplates.mergeTemplate, repMap ) )
    theFile.close()
    os.chmod(filePath,0755)
    
    return filePath
    
def createParallelMergeScript( path, validations ):
    if( len(validations) == 0 ):
        raise AllInOneError, "cowardly refusing to merge nothing!"

    repMap = validations[0].getRepMap() #FIXME - not nice this way
    repMap.update({
            "DownloadData":"",
            "CompareAlignments":"",
            "RunExtendedOfflineValidation":""
            })

    comparisonLists = {} # directory of lists containing the validations that are comparable
    for validation in validations:
        for referenceName in validation.filesToCompare:    
            validationName = "%s.%s"%(validation.__class__.__name__, referenceName)
            validationName = validationName.split(".%s"%GenericValidation.defaultReferenceName )[0]
            if validationName in comparisonLists:
                comparisonLists[ validationName ].append( validation )
            else:
                comparisonLists[ validationName ] = [ validation ]

    if "OfflineValidationParallel" in comparisonLists:
        repMap["extendeValScriptPath"] = os.path.join(path, "TkAlExtendedOfflineValidation.C")
        createExtendedValidationScript( comparisonLists["OfflineValidationParallel"], repMap["extendeValScriptPath"] )
        repMap["mergeOfflineParJobsScriptPath"] = os.path.join(path, "TkAlOfflineJobsMerge.C")
        createOfflineJobsMergeScript( comparisonLists["OfflineValidationParallel"],
                                      repMap["mergeOfflineParJobsScriptPath"] )

        # introduced to merge individual validation outputs separately
        #  -> avoids problems with merge script
        repMap["haddLoop"] = "mergeRetCode=0\n"
        repMap["rmUnmerged"] = "if [[ mergeRetCode -eq 0 ]]; then\n"
        for validation in comparisonLists["OfflineValidationParallel"]:
            repMap["haddLoop"] = validation.appendToMergeParJobs(repMap["haddLoop"])
            repMap["haddLoop"] += "tmpMergeRetCode=${?}\n"
            repMap["haddLoop"] += ("if [[ mergeRetCode -eq 0 ]]; "
                                   "then mergeRetCode=${tmpMergeRetCode}; "
                                   "fi\n")
            repMap["haddLoop"] += ("cmsStage -f "
                                   +validation.getRepMap()["outputFile"]
                                   +" "
                                   +validation.getRepMap()["resultFile"]
                                   +"\n")
            for f in validation.outputFiles:
                longName = os.path.join("/store/caf/user/$USER/",
                                        validation.getRepMap()["eosdir"], f)
                repMap["rmUnmerged"] += "    cmsRm "+longName+"\n"
        repMap["rmUnmerged"] += ("else\n"
                                 "    echo \"WARNING: Merging failed, unmerged"
                                 " files won't be deleted.\"\n"
                                 "fi\n")

        repMap["RunExtendedOfflineValidation"] = \
            replaceByMap(configTemplates.extendedValidationExecution, repMap)

        # DownloadData is the section which merges output files from parallel jobs
        # it uses the file TkAlOfflineJobsMerge.C
        repMap["DownloadData"] += replaceByMap("rfcp .oO[mergeOfflineParJobsScriptPath]Oo. .", repMap)
        repMap["DownloadData"] += replaceByMap( configTemplates.mergeOfflineParallelResults, repMap )

    repMap["CompareAlignments"] = "#run comparisons"
    for validationId in comparisonLists:
        compareStrings = [ val.getCompareStrings(validationId) for val in comparisonLists[validationId] ]
            
        repMap.update({"validationId": validationId,
                       "compareStrings": " , ".join(compareStrings) })
        
        repMap["CompareAlignments"] += \
            replaceByMap(configTemplates.compareAlignmentsExecution, repMap)
      
    filePath = os.path.join(path, "TkAlMerge.sh")
    theFile = open( filePath, "w" )
    theFile.write( replaceByMap( configTemplates.mergeTemplate, repMap ) )
    theFile.close()
    os.chmod(filePath,0755)
    
    return filePath
    
def loadTemplates( config ):
    if config.has_section("alternateTemplates"):
        for templateName in config.options("alternateTemplates"):
            newTemplateName = config.get("alternateTemplates", templateName )
            #print "replacing default %s template by %s"%( templateName, newTemplateName)
            configTemplates.alternateTemplate(templateName, newTemplateName)

    
####################--- Main ---############################
def main(argv = None):
    if argv == None:
       argv = sys.argv[1:]
    optParser = optparse.OptionParser()
    optParser.description = """ all-in-one alignment Validation 
    This will run various validation procedures either on batch queues or interactviely. 
    
    If no name is given (-N parameter) a name containing time and date is created automatically
    
    To merge the outcome of all validation procedures run TkAlMerge.sh in your validation's directory.
    """
    optParser.add_option("-n", "--dryRun", dest="dryRun", action="store_true", default=False,
                         help="create all scripts and cfg File but do not start jobs (default=False)")
    optParser.add_option( "--getImages", dest="getImages", action="store_true", default=False,
                          help="get all Images created during the process (default= False)")
    defaultConfig = "TkAlConfig.ini"
    optParser.add_option("-c", "--config", dest="config", default = defaultConfig,
                         help="configuration to use (default TkAlConfig.ini) this can be a comma-seperated list of all .ini file you want to merge", metavar="CONFIG")
    optParser.add_option("-N", "--Name", dest="Name",
                         help="Name of this validation (default: alignmentValidation_DATE_TIME)", metavar="NAME")
    optParser.add_option("-r", "--restrictTo", dest="restrictTo",
                         help="restrict validations to given modes (comma seperated) (default: no restriction)", metavar="RESTRICTTO")
    optParser.add_option("-s", "--status", dest="crabStatus", action="store_true", default = False,
                         help="get the status of the crab jobs", metavar="STATUS")
    optParser.add_option("-d", "--debug", dest="debugMode", action="store_true",
                         default = False,
                         help="Run the tool to get full traceback of errors.",
                         metavar="DEBUG")

    (options, args) = optParser.parse_args(argv)

    if not options.restrictTo == None:
        options.restrictTo = options.restrictTo.split(",")
    
    options.config = [ os.path.abspath( iniFile ) for iniFile in \
                       options.config.split( "," ) ]
    config = BetterConfigParser()
    outputIniFileSet = set( config.read( options.config ) )
    failedIniFiles = [ iniFile for iniFile in options.config if iniFile not in outputIniFileSet ]

    # Check for missing ini file
    if options.config == [ os.path.abspath( defaultConfig ) ]:
        if ( not options.crabStatus ) and \
               ( not os.path.exists( defaultConfig ) ):
                raise AllInOneError, ( "Default 'ini' file '%s' not found!\n"
                                       "You can specify another name with the "
                                       "command line option '-c'/'--config'."
                                       %( defaultConfig ))
    else:
        for iniFile in failedIniFiles:
            if not os.path.exists( iniFile ):
                raise AllInOneError, ( "'%s' does not exist. Please check for "
                                       "typos in the filename passed to the "
                                       "'-c'/'--config' option!"
                                       %( iniFile ) )
            else:
                raise AllInOneError, ( "'%s' does exist, but parsing of the "
                                       "content failed!" )

    # get the job name
    if options.Name == None:
        if not options.crabStatus:
            options.Name = "alignmentValidation_%s"%(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
        else:
            existingValDirs = fnmatch.filter( os.walk( '.' ).next()[1],
                                              "alignmentValidation_*" )
            if len( existingValDirs ) > 0:
                options.Name = existingValDirs[-1]
            else:
                print "Cannot guess last working directory!"
                print ( "Please use the parameter '-N' or '--Name' to specify "
                        "the task for which you want a status report." )
                return 1

    # set output path
    outPath = os.path.abspath( options.Name )

    # Check status of submitted jobs and return
    if options.crabStatus:
        os.chdir( outPath )
        crabLogDirs = fnmatch.filter( os.walk('.').next()[1], "crab.*" )
        if len( crabLogDirs ) == 0:
            print "Found no crab tasks for job name '%s'"%( options.Name )
            return 1
        theCrab = crabWrapper.CrabWrapper()
        for crabLogDir in crabLogDirs:
            print
            print "*" + "=" * 78 + "*"
            print ( "| Status report and output retrieval for:"
                    + " " * (77 - len( "Status report and output retrieval for:" ) )
                    + "|" )
            taskName = crabLogDir.replace( "crab.", "" )
            print "| " + taskName + " " * (77 - len( taskName ) ) + "|"
            print "*" + "=" * 78 + "*"
            print
            crabOptions = { "-getoutput":"",
                            "-c": crabLogDir }
            try:
                theCrab.run( crabOptions )
            except AllInOneError, e:
                print "crab:  No output retrieved for this task."
            crabOptions = { "-status": "",
                            "-c": crabLogDir }
            theCrab.run( crabOptions )
        return

    general = config.getGeneral()
    config.set("internals","workdir",os.path.join(general["workdir"],options.Name) )
    config.set("general","datadir",os.path.join(general["datadir"],options.Name) )
    config.set("general","logdir",os.path.join(general["logdir"],options.Name) )
    config.set("general","eosdir",os.path.join("AlignmentValidation", general["eosdir"], options.Name) )

    # clean up of log directory to avoid cluttering with files with different
    # random numbers for geometry comparison
    if os.path.isdir( outPath ):
        shutil.rmtree( outPath )
    
    if not os.path.exists( outPath ):
        os.makedirs( outPath )
    elif not os.path.isdir( outPath ):
        raise AllInOneError,"the file %s is in the way rename the Job or move it away"%outPath

    # replace default templates by the ones specified in the "alternateTemplates" section
    loadTemplates( config )

    #save backup configuration file
    backupConfigFile = open( os.path.join( outPath, "usedConfiguration.ini" ) , "w"  )
    config.write( backupConfigFile )

    validations = []
    for validation in config.items("validation"):
        alignmentList = validation[1].split(config.getSep())
        validationsToAdd = [(validation[0],alignment) \
                                for alignment in alignmentList]
        validations.extend(validationsToAdd)
    jobs = [ ValidationJob( validation, config, options) \
                 for validation in validations ]
    map( lambda job: job.createJob(), jobs )
    validations = [ job.getValidation() for job in jobs ]

    if "OfflineValidationParallel" not in [val.__class__.__name__ for val in validations]:
        createMergeScript(outPath, validations)
    else:
        createParallelMergeScript( outPath, validations )

    print
    map( lambda job: job.runJob(), jobs )
    

if __name__ == "__main__":        
    # main(["-n","-N","test","-c","defaultCRAFTValidation.ini,latestObjects.ini","--getImages"])
    if "-d" in sys.argv[1:] or "--debug" in sys.argv[1:]:
        main()
    else:
        try:
            main()
        except AllInOneError, e:
            print "\nAll-In-One Tool:", str(e)
            exit(1)
