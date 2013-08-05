#!/usr/bin/env python
#test execute: export CMSSW_BASE=/tmp/CMSSW && ./validateAlignments.py -c defaultCRAFTValidation.ini,test.ini -n -N test
import os
import sys
import ConfigParser
import optparse
import datetime
from pprint import pprint

import configTemplates #BETA as configTemplates

####################--- Helpers ---############################
#replaces .oO[id]Oo. by map[id] in target
def replaceByMap(target, map):
    result = target
    for id in map:
        #print "  "+id+": "+map[id]
        lifeSaver = 10e3
        iteration = 0
        while ".oO[" in result and "]Oo." in result:
            for id in map:
                result = result.replace(".oO["+id+"]Oo.",map[id])
                iteration += 1
            if iteration > lifeSaver:
                problematicLines = ""
                print map.keys()
                for line in result.splitlines():
                    if  ".oO[" in result and "]Oo." in line:
                        problematicLines += "%s\n"%line
                raise StandardError, "Oh Dear, there seems to be an endless loop in replaceByMap!!\n%s\nrepMap"%problematicLines
    return result

#excute [command] and return output
def getCommandOutput2(command):
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError, '%s failed w/ exit code %d' % (command, err)
    return data

#check if a directory exsits on castor
def castorDirExists(path):
    if path[-1] == "/":
        path = path[:-1]
    containingPath = os.path.join( *path.split("/")[:-1] )
    dirInQuestion = path.split("/")[-1]
    try:
        rawLines =getCommandOutput2("rfdir /"+containingPath).splitlines()
    except RuntimeError:
        return False
    for line in rawLines:
        if line.split()[0][0] == "d":
            if line.split()[8] == dirInQuestion:
                return True
    return False

####################--- Classes ---############################
class BetterConfigParser(ConfigParser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr
    
    def exists( self, section, option):
        try:
            items = self.items(section) 
        except ConfigParser.NoSectionError:
            return False
        for item in items:
            if item[0] == option:
                return True
        return False
        
    def __updateDict( self, dictionary, section ):
        try:
            for option in self.options( section ):
                dictionary[option] = self.get( section, option )
            if "local"+section.title() in self.sections():
                for option in self.options( "local"+section.title() ):
                    dictionary[option] = self.get( "local"+section.title(),
                                                   option )
#                 for option in dictionary:
#                     if option in [item[0] for item in \
#                                   self.items( "local"+section.title() )]:
#                         dictionary[option] = self.get( "local"+section.title(),
#                                                        option )
        except ConfigParser.NoSectionError, section:
            raise StandardError, ("%s in configuration files. This section is "
                                  "mandatory."
                                  %( str( section ).replace( ":", "", 1 ) ) )

    def getResultingSection( self, section, defaultDict = {}, demandPars = [] ):
        result = defaultDict
        for option in demandPars:
            try:
                result[option] = self.get( section, option )
            except ConfigParser.NoOptionError, globalSectionError:
                globalSection = str( globalSectionError ).split( "'" )[-2]
                print section
                print section.split(":")
                splittedSectionName = section.split( ":" )
                if len( splittedSectionName ) > 1:
                    localSection = ("local"+section.split( ":" )[0].title()+":"
                                    +section.split(":")[1])
                else:
                    localSection = ("local"+section.split( ":" )[0].title())
                print localSection,option
                if self.has_section( localSection ):
                    try:
                        result[option] = self.get( localSection, option )
                    except ConfigParser.NoOptionError, option:
                        raise StandardError, ("%s. This option is mandatory."
                                              %( str( option )\
                                                 .replace( ":", "", 1 )\
                                                 .replace( "section", "section '"\
                                                           +globalSection+"' or", 1 )
                                                 )
                                              )
                else:
                    raise StandardError, ("%s. This option is mandatory."
                                          %( str( globalSectionError )\
                                                 .replace( ":", "", 1 )
                                             )
                                          )
                    
        self.__updateDict( result, section )
        return result

    def getAlignments( self ):
        alignments = []
        for section in self.sections():
            if "alignment:" in section:
                alignments.append( Alignment( section.split( "alignment:" )[1],
                                              self ) )
        return alignments

    def getCompares( self ):
        compares = {}
        for section in self.sections():
            if "compare:" in section:
                levels = self.get( section, "levels" )
                dbOutput = self.get( section, "dbOutput" )
                compares[section.split(":")[1]] = ( levels, dbOutput )
        return compares

    def getGeneral( self ):
        defaults = {
            "jobmode":"interactive",
            "workdir":os.getcwd(),
            "datadir":os.getcwd(),
            "logdir":os.getcwd()# ,
#             "parallelJobs":"1"
            }
        general = self.getResultingSection( "general", defaultDict = defaults )
        return general


class Alignment:
    def __init__(self, name, config, runGeomComp = "1"):
        section = "alignment:%s"%name
        self.name = name
        self.runGeomComp = runGeomComp
        if not config.has_section( section ):
            raise StandardError, "section %s not found. Please define the alignment!"%section
        # self.mode = config.get(section, "mode").split()
        self.globaltag = config.get( section, "globaltag" )
 
        self.conditions = []
        for option in config.options( section ):
            if option.startswith( "condition " ):
                rcdName = option.split( "condition " )[1]
                conditionParameters = config.get( section, option ).split( "," )
                if len( conditionParameters ) < 2:
                    raise StandardError, "'%s' is used with too few arguments. A connect_string and a tag are required!"%( option )
                if len( conditionParameters ) < 3:
                    conditionParameters.append( "" )
                self.conditions.append({"rcdName": rcdName.strip(),
                                        "connectString": conditionParameters[0].strip(),
                                        "tagName": conditionParameters[1].strip(),
                                        "labelName": conditionParameters[2].strip()})
        # removed backward compatibility
        self.dbpath = ""
        self.tag = ""
#         if "TrackerAlignmentRcd" not in [ condition["rcdName"] \
#                                       for condition in self.conditions ]:
#             self.dbpath = config.get(section, "dbpath")
#             self.tag = config.get(section,"tag")
#             self.__testDbExist( self.dbpath )
#         else:
#             self.dbpath = ""
#             self.tag = ""

        self.errordbpath = "frontier://FrontierProd/CMS_COND_31X_FROM21X"
        self.errortag = "TrackerIdealGeometryErrors210_mc"
        if config.has_option(section,"errordbpath") and config.has_option(section,"errortag"):
            self.errordbpath = config.get(section, "errordbpath")
            self.__testDbExist( self.errordbpath )
            self.errortag = config.get(section,"errortag")
        else:
            if config.has_option(section,"errordbpath") or config.has_option(section,"errortag"):
                raise StandardError, "in alignment:%s you have to provide either both errordbpath _and_ errortag or none of both."%name
        if "TrackerAlignmentErrorRcd" in \
               [ condition["rcdName"] for condition in self.conditions ]\
               and config.has_option(section,"errordbpath"):
            raise StandardError,("Please do not use the old (errordbpath, "
                                 "errortag) and new syntax for configuring the"
                                 " APE at the same time.\nThe new syntax is "
                                 "'condition TrackerAlignmentErrorRcd: "
                                 "<connect_string>, <tag> (, <label>)'")
                                 
        
        self.color = config.get(section,"color")
        self.style = config.get(section,"style")

        self.kinksAndBows = ""
        self.kbdbpath = ""
        self.kbtag = ""
        if config.has_option(section,"kbdbpath") and config.has_option(section,"kbtag"):
            self.kinksAndBows = configTemplates.kinksAndBowsTemplate
            self.kbdbpath = config.get(section, "kbdbpath")
            self.__testDbExist( self.kbdbpath )
            self.kbtag = config.get(section,"kbtag")
        else:
            if config.has_option(section,"kbdbpath") or config.has_option(section,"kbtag"):
                raise StandardError, "in alignment:%s you have to provide either both kbdbpath _and_ kbtag or none of both."%name
        if "TrackerSurfaceDeformationRcd" in \
               [ condition["rcdName"] for condition in self.conditions ]\
               and config.has_option(section,"kbdbpath"):
            raise StandardError,("Please do not use the old (errordbpath, "
                                 "errortag) and new syntax for configuring the"
                                 " kinks and bows at the same time.\nThe new "
                                 "syntax is "
                                 "'condition TrackerSurfaceDeformationRcd: "
                                 "<connect_string>, <tag> (, <label>)'")

    def __testDbExist(self, dbpath):
        #FIXME delete return to end train debuging
        return
        if not dbpath.startswith("sqlite_file:"):
            print "WARNING: could not check existence for",dbpath
        else:
            if not os.path.exists( dbpath.split("sqlite_file:")[1] ):
                raise "could not find file: '%s'"%dbpath.split("sqlite_file:")[1]
 
    def restrictTo( self, restriction ):
        result = []
        if not restriction == None:
            for mode in self.mode:
                if mode in restriction:
                    result.append( mode )
            self.mode = result
        
    def getRepMap( self ):
        result = {
            "name": self.name,
            "dbpath": self.dbpath,
            "errordbpath": self.errordbpath,
            "tag": self.tag,
            "errortag": self.errortag,
            "color": self.color,
            "style": self.style,
            "runGeomComp": self.runGeomComp,
            "kinksAndBows": self.kinksAndBows,
            "kbdbpath": self.kbdbpath,
            "kbtag": self.kbtag,
            "GlobalTag": self.globaltag
            }
        return result  

    def getLoadTemplate(self):
        """This function still exists only for backward compatibility to the
           old syntax for overriding the global tag conditions.
           """
        # removed backward compatibility
        return ""
#         if "TrackerAlignmentRcd" in [ condition["rcdName"] \
#                                       for condition in self.conditions ]:
#             return ""
#         else:
#             return replaceByMap( configTemplates.dbLoadTemplate,
#                                  self.getRepMap() )

    def getAPETemplate(self):
        """This function still exists only for backward compatibility to the
           old syntax for overriding the global tag conditions.
           """
        if "TrackerAlignmentErrorRcd" in [ condition["rcdName"] \
                                           for condition in self.conditions ]:
            return ""
        else:
            return replaceByMap( configTemplates.APETemplate,
                                 self.getRepMap() )

    def getConditions(self):
        """This function creates the configuration snippet to override
           global tag conditions.
           """
        if len( self.conditions ):
            loadCond = ("\nimport CalibTracker.Configuration."
                        "Common.PoolDBESSource_cfi\n")
            for cond in self.conditions:
                if not cond["labelName"] == "":
                    temp = configTemplates.conditionsTemplate.replace(
                        "tag = cms.string('.oO[tagName]Oo.')",
                        ("tag = cms.string('.oO[tagName]Oo.'),"
                         "\nlabel = cms.untracked.string('.oO[labelName]Oo.')"))
                else:    
                    temp = configTemplates.conditionsTemplate
                loadCond += replaceByMap( temp, cond )
        else:
            loadCond = ""
        return loadCond


class GenericValidation:
    defaultReferenceName = "DEFAULT"

    def __init__(self, valName, alignment, config):
    # def __init__(self, alignment, config):
        import random
        self.name = valName
        self.alignmentToValidate = alignment
        self.general = config.getGeneral()
        self.randomWorkdirPart = "%0i"%random.randint(1,10e9)
        self.configFiles = []
        self.filesToCompare = {}
#        self.configFileSchedule = None
        self.jobmode = self.general["jobmode"]

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        result = alignment.getRepMap()
        result.update( self.general )
        result.update({
#                 "nJobs": self.general["parallelJobs"],
                "workdir": os.path.join(self.general["workdir"],self.randomWorkdirPart),
                "datadir": self.general["datadir"],
                "logdir": self.general["logdir"],
                "dbLoad": alignment.getLoadTemplate(),
                "APE": alignment.getAPETemplate(),
                "CommandLineTemplate": """#run configfile and post-proccess it
cmsRun %(cfgFile)s
%(postProcess)s """,
                "CMSSW_BASE": os.environ['CMSSW_BASE'],
                "SCRAM_ARCH": os.environ['SCRAM_ARCH'],
                "alignmentName": alignment.name,
                "condLoad": alignment.getConditions()
                })

        return result

    def getCompareStrings( self, requestId = None ):
        result = {}
        repMap = self.alignmentToValidate.getRepMap()
        for validationId in self.filesToCompare:
            repMap["file"] = self.filesToCompare[ validationId ]
            if repMap["file"].startswith( "/castor/" ):
                repMap["file"] = "rfio:%(file)s"%repMap
            result[ validationId ]=  "%(file)s=%(name)s|%(color)s|%(style)s"%repMap 
        if requestId == None:
            return result
        else:
            if not "." in requestId:
                requestId += ".%s"%GenericValidation.defaultReferenceName
            if not requestId.split(".")[-1] in result:
                raise StandardError, "could not find %s in reference Objects!"%requestId.split(".")[-1]
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
        self.configFiles = GenericValidation.createFiles( self, fileContents, path ) 
        if not schedule == None:
            schedule = [  os.path.join( path, cfgName) for cfgName in schedule]
            for cfgName in schedule:
                if not cfgName in self.configFiles:
                    raise StandardError, "scheduled %s missing in generated configfiles: %s"% (cfgName, self.configFiles)
            for cfgName in self.configFiles:
                if not cfgName in schedule:
                    raise StandardError, "generated configuration %s not scheduled: %s"% (cfgName, schedule)
            self.configFiles = schedule
        return self.configFiles

    def createScript(self, fileContents, path, downloadFiles=[] ):        
        self.scriptFiles =  GenericValidation.createFiles( self, fileContents, path )
        for script in self.scriptFiles:
            os.chmod(script,0755)
        return self.scriptFiles

    
class GeometryComparison(GenericValidation):
    """
object representing a geometry comparison job
alignemnt is the alignment to analyse
config is the overall configuration
copyImages indicates wether plot*.eps files should be copied back from the farm
"""
    def __init__( self, valName, alignment, referenceAlignment,
                  config, copyImages = True, randomWorkdirPart = None):
        GenericValidation.__init__(self, valName, alignment, config)
        if not randomWorkdirPart == None:
            self.randomWorkdirPart = randomWorkdirPart
        self.referenceAlignment = referenceAlignment
        # self.__compares = {}
        # allCompares = config.getCompares()
        self.__compares = config.getCompares()
        try:
            self.jobmode = config.getResultingSection( "compare:"+self.name)["jobmode"]
        except KeyError:
            pass
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name

        # #test if all compare sections are present
        # for compareName in self.alignmentToValidate.compareTo[ referenceName ]:
        #     if compareName in allCompares:
        #         self.__compares[compareName] = allCompares[compareName]
        #     else:
        #         raise StandardError, "could not find compare section '%s' in '%s'"%(compareName, allCompares)                  
        self.copyImages = copyImages
    
    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = GenericValidation.getRepMap( self, alignment )
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name
        
        repMap.update({
            "comparedGeometry": (".oO[workdir]Oo./.oO[alignmentName]Oo."
                                 "ROOTGeometry.root"),
            "referenceGeometry": "IDEAL", # will be replaced later
                                          #  if not compared to IDEAL
            "reference": referenceName# ,
            # # Keep the following parameters for backward compatibility
            # "GlobalTag": self.general["globaltag"]
            })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = (".oO[workdir]Oo./.oO[reference]Oo."
                                           "ROOTGeometry.root")
        repMap["name"] += "_vs_.oO[reference]Oo."
        return repMap

    def createConfiguration(self, path ):
        # self.__compares
        repMap = self.getRepMap()
        cfgs = {"TkAlCompareToNTuple.%s_cfg.py"%self.alignmentToValidate.name:
                replaceByMap( configTemplates.intoNTuplesTemplate, repMap)}
        if not self.referenceAlignment == "IDEAL":
            referenceRepMap = self.getRepMap( self.referenceAlignment )
            cfgFileName = "TkAlCompareToNTuple.%s.%s_cfg.py"%(
                self.referenceAlignment.name, self.randomWorkdirPart)
            cfgs[ cfgFileName ] = replaceByMap( configTemplates.intoNTuplesTemplate, referenceRepMap)

        cfgSchedule = cfgs.keys()
        for common in self.__compares:
            repMap.update({"common": common,
                           "levels": self.__compares[common][0],
                           "dbOutput": self.__compares[common][1]
                           })
            if self.__compares[common][1].split()[0] == "true":
                repMap["dbOutputService"] = configTemplates.dbOutputTemplate
            else:
                repMap["dbOutputService"] = ""
            cfgName = replaceByMap("TkAlCompareCommon.oO[common]Oo...oO[name]Oo._cfg.py",repMap)
            cfgs[ cfgName ] = replaceByMap(configTemplates.compareTemplate, repMap)
            
            cfgSchedule.append( cfgName )
        GenericValidation.createConfiguration(self, cfgs, path, cfgSchedule)

    def createScript(self, path):    
        repMap = self.getRepMap()    
        repMap["runComparisonScripts"] = ""
        scriptName = replaceByMap("TkAlGeomCompare..oO[name]Oo..sh",repMap)
        for name in self.__compares:
            if  '"DetUnit"' in self.__compares[name][0].split(","):
                repMap["runComparisonScripts"] += "root -b -q 'comparisonScript.C(\".oO[workdir]Oo./.oO[name]Oo..Comparison_common"+name+".root\",\".oO[workdir]Oo./\")'\n"
                if  self.copyImages:
                   repMap["runComparisonScripts"] += "rfmkdir -p .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images\n"
                   repMap["runComparisonScripts"] += "find .oO[workdir]Oo. -maxdepth 1 -name \"plot*.eps\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/\" \n"
                   repMap["runComparisonScripts"] += "rfmkdir -p .oO[workdir]Oo./.oO[name]Oo.."+name+"_ArrowPlots\n"
                   repMap["runComparisonScripts"] += "root -b -q 'makeArrowPlots.C(\".oO[workdir]Oo./.oO[name]Oo..Comparison_common"+name+".root\",\".oO[workdir]Oo./.oO[name]Oo.."+name+"_ArrowPlots\")'\n"
                   repMap["runComparisonScripts"] += "rfmkdir -p .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/ArrowPlots\n"
                   repMap["runComparisonScripts"] += "find .oO[workdir]Oo./.oO[name]Oo.."+name+"_ArrowPlots -maxdepth 1 -name \"*.png\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/ArrowPlots\"\n"
                   
                resultingFile = replaceByMap(".oO[datadir]Oo./compared%s_.oO[name]Oo..root"%name,repMap)
                resultingFile = os.path.expandvars( resultingFile )
                resultingFile = os.path.abspath( resultingFile )
                repMap["runComparisonScripts"] += "rfcp .oO[workdir]Oo./OUTPUT_comparison.root %s\n"%resultingFile
                self.filesToCompare[ name ] = resultingFile
                
        repMap["CommandLine"]=""

        for cfg in self.configFiles:
#find . -maxdepth 1 -name \"LOGFILE_*_.oO[name]Oo..log\" -print | xargs -I {} bash -c 'echo \"*** \";echo \"**   {}\";echo \"***\" ; cat {}' > .oO[workdir]Oo./LOGFILE_GeomComparision_.oO[name]Oo..log
#cd .oO[workdir]Oo.
            postProcess = "rfcp .oO[workdir]Oo./*.db .oO[datadir]Oo.\n"
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                                   "postProcess":postProcess
                                                                   }
        repMap["CommandLine"]+= """# overall postprocessing
cd .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/
.oO[runComparisonScripts]Oo.
cd .oO[workdir]Oo.
"""
        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }  
        return GenericValidation.createScript(self, scripts, path)

        
class OfflineValidation(GenericValidation):
    def __init__(self, valName, alignment,config):
        GenericValidation.__init__(self, valName, alignment, config)
        defaults = {
            "DMRMethod":"median",
            "DMRMinimum":"30",
            "DMROptions":"",
            "offlineModuleLevelHistsTransient":"False",
            "offlineModuleLevelProfiles":"False",
            "OfflineTreeBaseDir":"TrackHitFilter",
            "SurfaceShapes":"none"
            }
        mandatories = [ "dataset", "maxevents"# , "globaltag"
                        , "trackcollection" ]
        if not config.has_section( "offline:"+self.name ):
            offline = config.getResultingSection( "general",
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        else:
            offline = config.getResultingSection( "offline:"+self.name, 
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        self.general.update( offline )
        try:
            self.jobmode = config.getResultingSection( "offline:"+self.name)["jobmode"]
        except KeyError:
            pass
    
    def createConfiguration(self, path,
                            configBaseName = "TkAlOfflineValidation" ):
        cfgName = "%s.%s_cfg.py"%( configBaseName,
                                   self.alignmentToValidate.name )
        repMap = self.getRepMap()
          
        cfgs = {cfgName:replaceByMap( configTemplates.offlineTemplate, repMap)}
        self.filesToCompare[
            GenericValidation.defaultReferenceName ] = repMap["resultFile"]
        GenericValidation.createConfiguration(self, cfgs, path)
        
    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        scriptName = "%s.%s.sh"%(scriptBaseName, self.alignmentToValidate.name )
        repMap = GenericValidation.getRepMap(self)
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate,
                                             repMap ) }
        return GenericValidation.createScript(self, scripts, path)

    def getRepMap(self, alignment = None):
        repMap = GenericValidation.getRepMap(self, alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "outputFile": replaceByMap( (".oO[workdir]Oo./AlignmentValidation_"
                                         ".oO[name]Oo..root"), repMap ),
            "resultFile": replaceByMap( (".oO[datadir]Oo./AlignmentValidation_"
                                         ".oO[name]Oo..root"), repMap ),
            "TrackSelectionTemplate": configTemplates.TrackSelectionTemplate,
            "LorentzAngleTemplate": configTemplates.LorentzAngleTemplate,
            "offlineValidationMode": "Standalone",
            "offlineValidationFileOutput":
            configTemplates.offlineStandaloneFileOutputTemplate,
            "nJobs": self.general["parallelJobs"],
            # Keep the following parameters for backward compatibility
            "TrackCollection": self.general["trackcollection"]# ,
            # "GlobalTag": self.general["globaltag"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        repMap["resultFile"] = os.path.abspath( repMap["resultFile"] )

        # In case maxevents==-1, set number of parallel jobs to 1
        # since we cannot calculate number of events for each
        # parallel job
        if str(self.general["maxevents"]) == "-1":
            repMap.update({ "nJobs": "1" })
        return repMap

    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        if validationsSoFar == "":
            validationsSoFar = ('PlotAlignmentValidation p("%(resultFile)s",'
                                '"%(name)s", %(color)s, %(style)s);\n')%repMap
        else:
            validationsSoFar += ('p.loadFileList("%(resultFile)s", "%(name)s",'
                                 '%(color)s, %(style)s);\n')%repMap
        return validationsSoFar


class OfflineValidationParallel(OfflineValidation):
    def __init__(self, valName, alignment,config):
        OfflineValidation.__init__(self, valName, alignment, config)
        defaults = {
            "parallelJobs":"1"
            }
        if not config.has_section( "offline:"+self.name ):
            offline = config.getResultingSection( "general",
                                                  defaultDict = defaults )
        else:
            offline = config.getResultingSection( "offline:"+self.name, 
                                                  defaultDict = defaults )
        self.general.update( offline )
        self.__NJobs = self.general["parallelJobs"]

    def createConfiguration(self, path, configBaseName = "TkAlOfflineValidation" ):
        # if offline validation uses N parallel jobs, we create here N cfg files
        numberParallelJobs = int( self.general["parallelJobs"] )
        # limit maximum number of parallel jobs to 40
        # (each output file is approximately 20MB)
        maximumNumberJobs = 40
        if numberParallelJobs > maximumNumberJobs:
            raise StandardError, "Maximum allowed number of parallel jobs "+str(maximumNumberJobs)+" exceeded!!!"
        # if maxevents is not specified, cannot calculate number of events for each
        # parallel job, and therefore running only a single job
        if int( self.general["maxevents"] ) == -1:
            raise StandardError, "Maximum number of events (maxevents) not specified: cannot use parallel jobs in offline validation"
        if numberParallelJobs > 1:    
            if self.general["offlineModuleLevelHistsTransient"] == "True":
                raise StandardError, "To be able to merge results when running parallel jobs, set offlineModuleLevelHistsTransient to false."
        for index in range(numberParallelJobs):
            cfgName = "%s.%s_%s_cfg.py"%( configBaseName, self.alignmentToValidate.name, str(index) )
            repMap = self.getRepMap()
            # in this parallel job, skip index*(maxEvents/nJobs) events from the beginning
            # (first index is zero, so no skipping for a single job)
            # and use _index_ in the name of the output file
            repMap.update({"nIndex": str(index)})
            repMap.update({
                "outputFile": replaceByMap( ".oO[workdir]Oo./AlignmentValidation_.oO[name]Oo._.oO[nIndex]Oo..root", repMap )
                })
            repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
            repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )

            cfgs = {cfgName:replaceByMap( configTemplates.offlineParallelTemplate, repMap)}
            self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["resultFile"] 
            GenericValidation.createConfiguration(self, cfgs, path)
            # here is a small problem. only the last cfgs is saved
            # it requires a bit ugly solution later
        
    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        # A separate script is created for each parallel jobs.
        # Since only one cfg is saved a bit ugly solution is needed in the loop.
        returnValue = []
        numJobs = int( self.general["parallelJobs"] )
        for index in range(numJobs):
            scriptName = "%s.%s_%s.sh"%(scriptBaseName, self.alignmentToValidate.name, str(index) )
            repMap = GenericValidation.getRepMap(self)
            repMap["nIndex"]=""
            repMap["nIndex"]=str(index)
            repMap["CommandLine"]=""
            for cfg in self.configFiles:
                # The ugly solution here is to change the name for each parallel job 
                cfgtemp = cfg.replace( str(numJobs-1)+"_cfg.py" , str(index)+"_cfg.py" )
                repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfgtemp,
                                                                       "postProcess":""
                                                                       }
                scripts = {scriptName: replaceByMap( configTemplates.parallelScriptTemplate, repMap ) }
                returnValue.extend( GenericValidation.createScript(self, scripts, path) )
        return returnValue

    def getRepMap(self, alignment = None):
        repMap = OfflineValidation.getRepMap(self, alignment) 
        return repMap

    def appendToMergeParJobs( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned, 
        else the validation is appended to the list
        """
        repMap = self.getRepMap()
        if validationsSoFar == "":
            parameters = ""
            fileToAdd = ""
            for index in range(int(self.__NJobs)):
                fileToAdd = '%(resultFile)s'%repMap
                fileToAdd = fileToAdd.replace('.root','_'+str(index)+'.root')
                if index < int( self.general["parallelJobs"] )-1:
                    parameters = parameters+fileToAdd+','
                else:
                    parameters = parameters+fileToAdd                

            validationsSoFar = 'hadd("'+parameters+'");'
        return validationsSoFar


class OfflineValidationDQM(OfflineValidation):
    def __init__(self, valName, alignment, config):
        OfflineValidation.__init__(self, valName, alignment, config)
        if not config.has_section("DQM"):
            raise StandardError, "You need to have a DQM section in your configfile!"
        
        self.__PrimaryDataset = config.get("DQM", "primaryDataset")
        self.__firstRun = int(config.get("DQM", "firstRun"))
        self.__lastRun = int(config.get("DQM", "lastRun"))

    def createConfiguration(self, path):
        OfflineValidation.createConfiguration(self, path, "TkAlOfflineValidationDQM")
        
    def createScript(self, path):
        return OfflineValidation.createScript(self, path, "TkAlOfflineValidationDQM")

    def getRepMap(self, alignment = None):
        repMap = OfflineValidation.getRepMap(self, alignment)
        repMap.update({
                "workdir": os.path.expandvars(repMap["workdir"]),
		"offlineValidationMode": "Dqm",
                "offlineValidationFileOutput": configTemplates.offlineDqmFileOutputTemplate,
                "workflow": "/%s/TkAl%s-.oO[alignmentName]Oo._R%09i_R%09i_ValSkim-v1/ALCARECO"%(self.__PrimaryDataset, datetime.datetime.now().strftime("%y"), self.__firstRun, self.__lastRun),
                "firstRunNumber": "%i"% self.__firstRun
                }
            )
        if "__" in repMap["workflow"]:
            raise StandardError, "the DQM workflow specefication must not contain '__'. it is: %s"%repMap["workflow"]
        return repMap

class MonteCarloValidation(GenericValidation):
    def __init__(self, valName, alignment, config):
        GenericValidation.__init__(self, valName, alignment, config)
        try:
            self.jobmode = config.getResultingSection( "mcValidate:"+self.name)["jobmode"]
        except KeyError:
            pass
        mandatories = [ "relvalsample", "maxevents" ]
        if not config.has_section( "mcValidate:"+self.name ):
            mcValidate = config.getResultingSection( "general",
                                                  demandPars = mandatories )
        else:
            mcValidate = config.getResultingSection( "mcValidate:"+self.name, 
                                                  demandPars = mandatories )
        self.general.update( mcValidate )

    def createConfiguration(self, path ):
        cfgName = "TkAlMcValidation.%s_cfg.py"%( self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap.update({
                "outputFile": replaceByMap( ".oO[workdir]Oo./McValidation_.oO[name]Oo..root", repMap )
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        cfgs = {cfgName:replaceByMap( configTemplates.mcValidateTemplate, repMap)}
        self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["outputFile"]
        GenericValidation.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlMcValidate.%s.sh"%( self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }

        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidation.getRepMap(self, alignment)
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            # Keep the following parameters for backward compatibility
            "RelValSample": self.general["relvalsample"]# ,
            # "GlobalTag": self.general["globaltag"]
            })
        return repMap


class TrackSplittingValidation(GenericValidation):
    def __init__(self, valName, alignment, config):
        GenericValidation.__init__(self, valName, alignment, config)
        try:
            self.jobmode = config.getResultingSection( "split:"+self.name)["jobmode"]
        except KeyError:
            pass
        mandatories = [ "trackcollection", "maxevents" ]
        if not config.has_section( "split:"+self.name ):
            split = config.getResultingSection( "general",
                                                  demandPars = mandatories )
        else:
            split = config.getResultingSection( "split:"+self.name, 
                                                  demandPars = mandatories )
        self.general.update( split )


    def createConfiguration(self, path ):
        cfgName = "TkAlTrackSplitting.%s_cfg.py"%( self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap.update({
                "outputFile": replaceByMap( ".oO[workdir]Oo./TrackSplitting_.oO[name]Oo..root", repMap )
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        cfgs = {cfgName:replaceByMap( configTemplates.TrackSplittingTemplate, repMap)}
        self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["outputFile"]
        GenericValidation.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlTrackSplitting.%s.sh"%( self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }

        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidation.getRepMap(self)
        # repMap = self.getRepMap()
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            # Keep the following parameters for backward compatibility
            "TrackCollection": self.general["trackcollection"]# ,
            # "GlobalTag": self.general["globaltag"]
            })
        return repMap
    
    
class ZMuMuValidation(GenericValidation):
    def __init__(self, valName, alignment,config):
        GenericValidation.__init__(self, valName, alignment, config)
        defaults = {
            "zmumureference": ("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2"
                               "/TMP_EM/ZMuMu/data/MC/BiasCheck_DYToMuMu_Summer"
                               "11_TkAlZMuMu_IDEAL.root")
            }
        mandatories = [ "dataset", "maxevents", "etamax1", "etamin1", "etamax2",
                        "etamin2" ]
        if not config.has_section( "zmumu:"+self.name ):
            zmumu = config.getResultingSection( "general",
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        else:
            zmumu = config.getResultingSection( "zmumu:"+self.name, 
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        self.general.update( zmumu )
        try:
            self.jobmode = config.getResultingSection( "zmumu:"+self.name)["jobmode"]
        except KeyError:
            pass
    
    def createConfiguration(self, path, configBaseName = "TkAlZMuMuValidation" ):
        cfgName = "%s.%s_cfg.py"%( configBaseName, self.alignmentToValidate.name )
        repMap = self.getRepMap()
        cfgs = {cfgName:replaceByMap( configTemplates.ZMuMuValidationTemplate, repMap)}
        GenericValidation.createConfiguration(self, cfgs, path)
        
    def createScript(self, path, scriptBaseName = "TkAlZMuMuValidation"):
        scriptName = "%s.%s.sh"%(scriptBaseName, self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: replaceByMap( configTemplates.zMuMuScriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)

    def getRepMap(self, alignment = None):
        repMap = GenericValidation.getRepMap(self, alignment) 
        repMap.update({
            "nEvents": self.general["maxevents"] # ,
            # "GlobalTag": self.general["globaltag"]
                })
        return repMap

class ValidationJob:
    def __init__( self, validation, config, options ):
        if validation[1] == "":
            # new intermediate syntax
            valString = validation[0].split( "->" )[0]
            alignments = validation[0].split( "->" )[1]
        else:
            # old syntax
            valString = validation[0]
            alignments = validation[1]
        valString = valString.split()
        self.__valType = valString[0]
        self.__valName = valString[1]
        self.__commandLineOptions = options
        # workaround for intermediate parallel version
        if self.__valType == "offlineParallel":
            section = "offline" + ":" + self.__valName
        else:
            section = self.__valType + ":" + self.__valName
        if not config.has_section( section ):
            raise StandardError, ("Validation '%s' of type '%s' is requested in"
                                  " '[validation]' section, but is not defined."
                                  "\nYou have to add a '[%s]' section."
                                  %( self.__valName, self.__valType, section ))
        self.validation = self.__getValidation( self.__valType, self.__valName,
                                                alignments, config, options )
        self.__config = config

    def __getValidation( self, valType, name, alignments, config, options ):
        if valType == "compare":
            # validation = GeometryComparison( name, alignments, config, options )
            alignmentsList = alignments.split( "," )
            firstAlignList = alignmentsList[0].split()
            firstAlignName = firstAlignList[0].strip()
            if firstAlignName == "IDEAL":
                raise StandardError, ("'IDEAL' has to be the second (reference)"
                                      " alignment in 'compare <val_name>: "
                                      "<alignment> <reference>'.")
            if len( firstAlignList ) > 1:
                firstRun = firstAlignList[1]
            else:
                firstRun = "1"
            firstAlign = Alignment( firstAlignName, config, firstRun )
            secondAlignList = alignmentsList[1].split()
            secondAlignName = secondAlignList[0].strip()
            if len( secondAlignList ) > 1:
                secondRun = secondAlignList[1]
            else:
                secondRun = "1"
            if secondAlignName == "IDEAL":
                secondAlign = secondAlignName
            else:
                secondAlign = Alignment( secondAlignName, config, secondRun )
            validation = GeometryComparison( name, firstAlign, secondAlign,
                                             config,
                                             self.__commandLineOptions.getImages )
        elif valType == "offline":
            # validation = OfflineValidation( name, alignments, config, options )
            validation = OfflineValidation( name, 
                Alignment( alignments.strip(), config ), config )
        elif valType == "offlineDQM":
            validation = OfflineValidationDQM( name, 
                Alignment( alignments.strip(), config ), config )
        elif valType == "offlineParallel":
            validation = OfflineValidationParallel( name, 
                Alignment( alignments.strip(), config ), config )
        elif valType == "mcValidate":
            validation = MonteCarloValidation( name, 
                Alignment( alignments.strip(), config ), config )
        elif valType == "split":
            validation = TrackSplittingValidation( name, 
                Alignment( alignments.strip(), config ), config )
        elif valType == "zmumu":
            validation = ZMuMuValidation( name, 
                Alignment( alignments.strip(), config ), config )
        else:
            raise StandardError, "Unknown validation mode '%s'"%valType
        return validation

    def __createJob( self, jobMode, outpath ):
        # Create here scripts, cfg.py-files (, crab.cfg-files)
        self.validation.createConfiguration( outpath )
        self.__scripts = self.validation.createScript( outpath )
        # self.validation.createCrabConfig()
        return None

    def createJob(self):
        self.__createJob( self.validation.jobmode, os.path.abspath( self.__commandLineOptions.Name) )

    def runJob( self ):
        # Check here for valid job mode: lxBatch, interactive (, crab)
        general = self.__config.getGeneral()
#         self.__createJob( self.validation.jobmode, os.path.abspath( self.__commandLineOptions.Name) )
        # Now run the job 
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
            if self.validation.jobmode.split(",")[0] == "lxBatch":
                repMap = { 
                    "commands": self.validation.jobmode.split(",")[1],
                    "logDir": general["logdir"],
                    "jobName": name,
                    "script": script,
                    "bsub": "/afs/cern.ch/cms/caf/scripts/cmsbsub"
                    }
            log+=getCommandOutput2("%(bsub)s %(commands)s -J %(jobName)s "
                                   "-o %(logDir)s/%(jobName)s.stdout -e "
                                   "%(logDir)s/%(jobName)s.stderr %(script)s"%repMap)
        return log

    def getValidation( self ):
        return self.validation


def runJob(jobName, script, config):
    general = config.getGeneral()
    log = ">             Validating "+jobName
    print ">             Validating "+jobName
    if general["jobmode"] == "interactive":
        log += getCommandOutput2( script )
    if general["jobmode"].split(",")[0] == "lxBatch":
        repMap = { 
            "commands": general["jobmode"].split(",")[1],
            "logDir": general["logdir"],
            "jobName": jobName,
            "script": script,
            "bsub": "/afs/cern.ch/cms/caf/scripts/cmsbsub"
            }
        
        log+=getCommandOutput2("%(bsub)s %(commands)s -J %(jobName)s -o %(logDir)s/%(jobName)s.stdout -e %(logDir)s/%(jobName)s.stderr %(script)s"%repMap)
    return log

def createOfflineJobsMergeScript(offlineValidationList, outFilePath):
    repMap = offlineValidationList[0].getRepMap() # bit ugly since some special features are filled
    repMap[ "mergeOfflinParJobsInstantiation" ] = "" #give it a "" at first in order to get the initialisation back

    for validation in offlineValidationList:
        repMap[ "mergeOfflinParJobsInstantiation" ] = validation.appendToMergeParJobs( repMap[ "mergeOfflinParJobsInstantiation" ] )
#                    validationsSoFar = 'PlotAlignmentValidation p("%(resultFile)s", "%(name)s", %(color)s, %(style)s);\n'%repMap
    
    theFile = open( outFilePath, "w" )
    theFile.write( replaceByMap( configTemplates.mergeOfflineParJobsTemplate ,repMap ) )
    theFile.close()

def createExtendedValidationScript(offlineValidationList, outFilePath):
    repMap = offlineValidationList[0].getRepMap() # bit ugly since some special features are filled
    repMap[ "extendedInstantiation" ] = "" #give it a "" at first in order to get the initialisation back

    for validation in offlineValidationList:
        repMap[ "extendedInstantiation" ] = validation.appendToExtendedValidation( repMap[ "extendedInstantiation" ] )
    
    theFile = open( outFilePath, "w" )
    theFile.write( replaceByMap( configTemplates.extendedValidationTemplate ,repMap ) )
    theFile.close()
    
def createMergeScript( path, validations ):
    if( len(validations) == 0 ):
        raise StandardError, "cowardly refusing to merge nothing!"

    repMap = validations[0].getRepMap() #FIXME - not nice this way
    repMap.update({
            "DownloadData":"",
            "CompareAllignments":"",
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
        repMap["extendeValScriptPath"] = os.path.join(path, "TkAlExtendedOfflineValidation.C")
        createExtendedValidationScript( comparisonLists["OfflineValidation"], repMap["extendeValScriptPath"] )
        repMap["RunExtendedOfflineValidation"] = replaceByMap(configTemplates.extendedValidationExecution, repMap)

    repMap["CompareAllignments"] = "#run comparisons"
    for validationId in comparisonLists:
        compareStrings = [ val.getCompareStrings(validationId) for val in comparisonLists[validationId] ]
            
        repMap.update({"validationId": validationId,
                       "compareStrings": " , ".join(compareStrings) })
        
        repMap["CompareAllignments"] += replaceByMap( configTemplates.compareAlignmentsExecution, repMap )
      
    filePath = os.path.join(path, "TkAlMerge.sh")
    theFile = open( filePath, "w" )
    theFile.write( replaceByMap( configTemplates.mergeTemplate, repMap ) )
    theFile.close()
    os.chmod(filePath,0755)
    
    return filePath
    
def createParallelMergeScript( path, validations ):
    if( len(validations) == 0 ):
        raise StandardError, "cowardly refusing to merge nothing!"

    repMap = validations[0].getRepMap() #FIXME - not nice this way
    repMap.update({
            "DownloadData":"",
            "CompareAllignments":"",
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
        # add a script which merges output files from parallel jobs
        repMap["mergeOfflineParJobsScriptPath"] = os.path.join(path, "TkAlOfflineJobsMerge.C")
        createOfflineJobsMergeScript( comparisonLists["OfflineValidationParallel"], repMap["mergeOfflineParJobsScriptPath"] )
        repMap["RunExtendedOfflineValidation"] = replaceByMap(configTemplates.extendedValidationExecution, repMap)
        # DownloadData is used to merge output files from parallel jobs
        # it uses the file TkAlOfflineJobsMerge.C
        repMap["DownloadData"] += replaceByMap( configTemplates.mergeOfflineParallelResults, repMap )

    repMap["CompareAllignments"] = "#run comparisons"
    for validationId in comparisonLists:
        compareStrings = [ val.getCompareStrings(validationId) for val in comparisonLists[validationId] ]
            
        repMap.update({"validationId": validationId,
                       "compareStrings": " , ".join(compareStrings) })
        
        repMap["CompareAllignments"] += replaceByMap( configTemplates.compareAlignmentsExecution, repMap )
      
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
    optParser.add_option("-c", "--config", dest="config",
                         help="configuration to use (default compareConfig.ini) this can be a comma-seperated list of all .ini file you want to merge", metavar="CONFIG")
    optParser.add_option("-N", "--Name", dest="Name",
                         help="Name of this validation (default: alignmentValidation_DATE_TIME)", metavar="NAME")
    optParser.add_option("-r", "--restrictTo", dest="restrictTo",
                         help="restrict validations to given modes (comma seperated) (default: no restriction)", metavar="RESTRICTTO")

    (options, args) = optParser.parse_args(argv)

    if not options.restrictTo == None:
        options.restrictTo = options.restrictTo.split(",")
    if options.config == None:
        options.config = "compareConfig.ini"
    else:
        options.config = options.config.split(",")
        result = []
        for iniFile in options.config:
            result.append( os.path.abspath(iniFile) )
        options.config = result
    config = BetterConfigParser()
    config.read( options.config )

    if options.Name == None:
        options.Name = "alignmentValidation_%s"%(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

    outPath = os.path.abspath( options.Name )
    general = config.getGeneral()
    config.set("general","workdir",os.path.join(general["workdir"],options.Name) )
    config.set("general","datadir",os.path.join(general["datadir"],options.Name) )
    config.set("general","logdir",os.path.join(general["logdir"],options.Name) )
    if not os.path.exists( outPath ):
        os.makedirs( outPath )
    elif not os.path.isdir( outPath ):
        raise StandardError,"the file %s is in the way rename the Job or move it away"%outPath

    #replace default templates by the once specified in the "alternateTemplates" section
    loadTemplates( config )

    # alignments = config.getAlignments()
    # validations = []
    # for alignment in alignments:
    #     alignment.restrictTo( options.restrictTo )
    #     validations.extend( alignment.createValidations( config, options, alignments ) )

    #save backup configuration file
    backupConfigFile = open( os.path.join( outPath, "usedConfiguration.ini" ) , "w"  )
    config.write( backupConfigFile )

    jobs = [ ValidationJob( validation, config, options) \
                 for validation in config.items( "validation" ) ]
    map( lambda job: job.createJob(), jobs )
    validations = [ job.getValidation() for job in jobs ]

#     if general["parallelJobs"] == "1":
    if "OfflineValidationParallel" not in [val.__class__.__name__ for val in validations]:
        createMergeScript( outPath, validations )
    else:
        createParallelMergeScript( outPath, validations )

    map( lambda job: job.runJob(), jobs )
    

if __name__ == "__main__":        
   # main(["-n","-N","test","-c","defaultCRAFTValidation.ini,latestObjects.ini","--getImages"])
   main()

