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

class Alignment:
    def __init__(self, name, config):
        section = "alignment:%s"%name
        self.name = name
        if not config.has_section( section ):
            raise StandardError, "section %s not found. Please define the alignment!"%section
        self.mode = config.get(section, "mode").split()
        self.dbpath = config.get(section, "dbpath")
        self.__testDbExist( self.dbpath )
 
        self.errordbpath = "frontier://FrontierProd/CMS_COND_31X_FROM21X"
        self.errortag = "TrackerIdealGeometryErrors210_mc"
        if config.has_option(section,"errordbpath") and config.has_option(section,"errortag"):
            self.errordbpath = config.get(section, "errordbpath")
            self.__testDbExist( self.errordbpath )
            self.errortag = config.get(section,"errortag")
        else:
            if config.has_option(section,"errordbpath") or config.has_option(section,"errortag"):
                raise StandardError, "in alignment:%s you have to provide either both errordbpath _and_ errortag or none of both."%name
        self.tag = config.get(section,"tag")
        
        self.color = config.get(section,"color")
        self.style = config.get(section,"style")
        if "compare" in self.mode:
            if config.has_option(section,"rungeomcomp"):
                self.runGeomComp = config.get(section,"rungeomcomp")
            else:
                raise StandardError, "in alignment:%s you have to provide rungeomcomp in compare mode."%(name)
        else:
            self.runGeomComp = "1"

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

        if config.has_option(section,"primeplot"):
            self.primePlot = config.get(section,"primeplot")
        else:
            self.primePlot = "true"
            
        if config.has_option(section,"splitplots"):
            self.splitPlots = config.get(section,"splitplots")
        else:
            self.splitPlots = "false"
            
        self.compareTo = {}
        for option in config.options( section ):
            if option.startswith("compare|"):
                alignmentName = option.split("compare|")[1]
                comparisonList = config.get( section, option ).split()
                if len(comparisonList) == 0:
                    raise StandardError, "empty comaprison list '%s' given for %s"%(config.get( section, option ), alignmentName )
                self.compareTo[ alignmentName ] = comparisonList
        if self.compareTo == {}:
            self.compareTo = {
                "IDEAL":["Tracker","SubDets"]
                }       

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
            "primePlot": self.primePlot,
            "splitPlots": self.splitPlots
            }
        return result  

    def getLoadTemplate(self):
        return replaceByMap( configTemplates.dbLoadTemplate, self.getRepMap() )

    def createValidations(self, config, options, allAlignments=[]):
        """
config is the overall configuration
options are the options as paresed from command line
allAlignemts is a list of Alignment objects the is used to generate Alignment_vs_Alignemnt jobs
"""
        result = []
        for validationName in self.mode:
            if validationName == "compare":
                randomWorkdirPart = None #use only one ranom numberr for all geom. comparisions for one alignment
                #test if all needed alignments are defined
                for alignmentName in self.compareTo:
                    referenceAlignment = 'IDEAL'
                    if not alignmentName == "IDEAL":
                        foundAlignment = False
                        for alignment in allAlignments:
                            if alignment.name == alignmentName:
                                referenceAlignment = alignment
                                foundAlignment = True
                        if not foundAlignment:
                            raise StandardError, " could not find alignment called '%s'"%alignmentName
                    result.append( GeometryComparison( self, referenceAlignment, config, options.getImages, randomWorkdirPart ) )
                    if randomWorkdirPart == None:
                        randomWorkdirPart = result[-1].randomWorkdirPart
            elif validationName == "offline":
                result.append( OfflineValidation( self, config ) )
            elif validationName == "offlineDQM":
                result.append( OfflineValidationDQM( self, config ) )
            elif validationName == "mcValidate":
                result.append( MonteCarloValidation( self, config ) )
            elif validationName == "split":
                result.append( TrackSplittingValidation( self, config ) )
            elif validationName == "zmumu":
                result.append( ZMuMuValidation( self, config ) )
            else:
                raise StandardError, "unknown validation mode '%s'"%validationName
        return result

class GenericValidation:
    defaultReferenceName = "DEFAULT"

    def __init__(self, alignment, config):
        import random
        self.alignmentToValidate = alignment
        self.__general = readGeneral( config )
        self.randomWorkdirPart = "%0i"%random.randint(1,10e9)
        self.configFiles = []
        self.filesToCompare = {}
#        self.configFileSchedule = None

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        result = alignment.getRepMap()
        result.update({
                "nEvents": str(self.__general["maxevents"]),
                "dataset": str(self.__general["dataset"]),
                "superPointingDataset": str(self.__general["superPointingDataset"]),
                "RelValSample": self.__general["relvalsample"],
                "TrackCollection": str(self.__general["trackcollection"]),
                "workdir": str(os.path.join(self.__general["workdir"],self.randomWorkdirPart)),
                "datadir": str(self.__general["datadir"]),
                "logdir": str(self.__general["logdir"]),
                "dbLoad": alignment.getLoadTemplate(),
                "CommandLineTemplate": """#run configfile and post-proccess it
cmsRun %(cfgFile)s
%(postProcess)s """,
                "GlobalTag": self.__general["globaltag"],
                "CMSSW_BASE": os.environ['CMSSW_BASE'],
                "SCRAM_ARCH": os.environ['SCRAM_ARCH'],
                "alignmentName": alignment.name,
                "offlineModuleLevelHistsTransient":  self.__general["offlineModuleLevelHistsTransient"]
                })
        #TODO catch missing delcalration of i.e. dataset or relvalsample here and rethrow to be catched by 
        #     individual validation.
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
    def __init__(self, alignment, referenceAlignment, config, copyImages = True, randomWorkdirPart = None):
        GenericValidation.__init__(self, alignment, config)
        if not randomWorkdirPart == None:
            self.randomWorkdirPart = randomWorkdirPart
        self.referenceAlignment = referenceAlignment
        self.__compares = {}
        allCompares = readCompare(config)
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name

        #test if all compare sections are present
        for compareName in self.alignmentToValidate.compareTo[ referenceName ]:
            if compareName in allCompares:
                self.__compares[compareName] = allCompares[compareName]
            else:
                raise StandardError, "could not find compare section '%s' in '%s'"%(compareName, allCompares)                  
        self.copyImages = copyImages
    
    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = GenericValidation.getRepMap( self, alignment )
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name
        
        repMap.update({"comparedGeometry": ".oO[workdir]Oo./.oO[alignmentName]Oo.ROOTGeometry.root",
                       "referenceGeometry": "IDEAL",#will be replaced later if not compared to IDEAL
                       "reference": referenceName,
                       "APE": configTemplates.APETemplate
                       })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = ".oO[workdir]Oo./.oO[reference]Oo.ROOTGeometry.root"
        repMap["name"] += "_vs_.oO[reference]Oo."
        return repMap

    def createConfiguration(self, path ):
        # self.__compares
        repMap = self.getRepMap()
        cfgs = {"TkAlCompareToNTuple.%s_cfg.py"%self.alignmentToValidate.name:
                    replaceByMap( configTemplates.intoNTuplesTemplate, repMap)}
        if not self.referenceAlignment == "IDEAL":
            referenceRepMap = self.getRepMap( self.referenceAlignment )
            cfgFileName = "TkAlCompareToNTuple.%s.%s_cfg.py"%(self.referenceAlignment.name, self.randomWorkdirPart)
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
    def __init__(self, alignment,config):
        GenericValidation.__init__(self, alignment, config)
        general = readGeneral( config )
        self.__DMRMethod = general["DMRMethod"]
        self.__DMRMinimum = general["DMRMinimum"]
        self.__OfflineTreeBaseDir = general["OfflineTreeBaseDir"]
    
    def createConfiguration(self, path, configBaseName = "TkAlOfflineValidation" ):
        cfgName = "%s.%s_cfg.py"%( configBaseName, self.alignmentToValidate.name )
        repMap = self.getRepMap()
          
        cfgs = {cfgName:replaceByMap( configTemplates.offlineTemplate, repMap)}
        self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["resultFile"] 
        GenericValidation.createConfiguration(self, cfgs, path)
        
    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        scriptName = "%s.%s.sh"%(scriptBaseName, self.alignmentToValidate.name )
        repMap = GenericValidation.getRepMap(self)
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)

    def getRepMap(self, alignment = None):
        repMap = GenericValidation.getRepMap(self, alignment) 
        repMap.update({
                "OfflineTreeBaseDir": self.__OfflineTreeBaseDir,
                "DMRMethod":self.__DMRMethod,
                "DMRMinimum":self.__DMRMinimum,
                "APE": configTemplates.APETemplate,
                "outputFile": replaceByMap( ".oO[workdir]Oo./AlignmentValidation_.oO[name]Oo..root", repMap ),
                "resultFile": replaceByMap( ".oO[datadir]Oo./AlignmentValidation_.oO[name]Oo..root", repMap ),
                "TrackSelectionTemplate": configTemplates.TrackSelectionTemplate,
                "LorentzAngleTemplate": configTemplates.LorentzAngleTemplate,
                "offlineValidationMode": "Standalone",
                "offlineValidationFileOutput": configTemplates.offlineStandaloneFileOutputTemplate
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        repMap["resultFile"] = os.path.abspath( repMap["resultFile"] )
        
        return repMap

    
    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned, 
        else the validation is appended to the list
        """
        repMap = self.getRepMap()
        if validationsSoFar == "":
            validationsSoFar = 'PlotAlignmentValidation p("%(resultFile)s", "%(name)s", %(color)s, %(style)s);\n'%repMap
        else:
            validationsSoFar +='p.loadFileList("%(resultFile)s", "%(name)s", %(color)s, %(style)s);\n'%repMap

#          PlotAlignmentValidation p(".oO[firstFile]Oo.",".oO[firstLegendEntry]Oo.");
#  p.loadFileList("rfio:/castor/cern.ch/user/j/jdraeger/Validation/MCfromCRAFT/new/Validation_MC_Adun1_CosmicTF.root","Brot ist lecker2");
        return validationsSoFar

class OfflineValidationDQM(OfflineValidation):
    def __init__(self, alignment, config):
        OfflineValidation.__init__(self, alignment, config)
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
    def __init__(self, alignment, config):
        GenericValidation.__init__(self, alignment, config)

    def createConfiguration(self, path ):
        cfgName = "TkAlMcValidation.%s_cfg.py"%( self.alignmentToValidate.name )
        repMap = GenericValidation.getRepMap(self)
        repMap.update({
                "APE": configTemplates.APETemplate,
                "outputFile": replaceByMap( ".oO[workdir]Oo./McValidation_.oO[name]Oo..root", repMap )
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        cfgs = {cfgName:replaceByMap( configTemplates.mcValidateTemplate, repMap)}
        self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["outputFile"]
        GenericValidation.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlMcValidate.%s.sh"%( self.alignmentToValidate.name )
        repMap = GenericValidation.getRepMap(self)
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }

        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)

class TrackSplittingValidation(GenericValidation):
    def __init__(self, alignment, config):
        GenericValidation.__init__(self, alignment, config)

    def createConfiguration(self, path ):
        cfgName = "TkAlTrackSplitting.%s_cfg.py"%( self.alignmentToValidate.name )
        repMap = GenericValidation.getRepMap(self)
        repMap.update({
                "APE": configTemplates.APETemplate,
                "outputFile": replaceByMap( ".oO[workdir]Oo./TrackSplitting_.oO[name]Oo..root", repMap )
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        cfgs = {cfgName:replaceByMap( configTemplates.TrackSplittingTemplate, repMap)}
        self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["outputFile"]
        GenericValidation.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlTrackSplitting.%s.sh"%( self.alignmentToValidate.name )
        repMap = GenericValidation.getRepMap(self)
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }

        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)
    
class ZMuMuValidation(GenericValidation):
    def __init__(self, alignment,config):
        GenericValidation.__init__(self, alignment, config)
        general = readGeneral( config )
        for key in ("zmumureference","etamax1","etamin1","etamax2","etamin2"):
            if not general.has_key(key):
                raise StandardError, "missing parameter '%s' in general section. This parameter is mandatory for the ZMuMuValidation."%(key)
        
        self.__zmumureference = general["zmumureference"]
        self.__etamax1 = general["etamax1"]
        self.__etamin1 = general["etamin1"]
        self.__etamax2 = general["etamax2"]
        self.__etamin2 = general["etamin2"]
    
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
                "APE": configTemplates.APETemplate,
                "zmumureference":self.__zmumureference,
                "etamax1":self.__etamax1,
                "etamin1":self.__etamin1,
                "etamax2":self.__etamax2,
                "etamin2":self.__etamin2
                })
       
        return repMap


####################--- Read Configfiles ---############################
def readAlignments( config ):
    result = []
    for section in config.sections():
        if "alignment:" in section:
            result.append( Alignment( section.split("alignment:")[1], config ) )
    return result

def readCompare( config ):
    result = {}
    for section in config.sections():
        if "compare:" in section:
            levels =  config.get(section, "levels")
            dbOutput = config.get(section, "dbOutput")
            result[section.split(":")[1]] =(levels,dbOutput)
    return result

def readGeneral( config ):
    result = {
        "jobmode":"interactive",
        "superPointingDataset":"",# see remark about rethroughung missing entries in general. this is needed to be backward compatible
        "workdir":os.getcwd(),
        "datadir":os.getcwd(),
        "logdir":os.getcwd(),
        "offlineModuleLevelHistsTransient":"False",
        "OfflineTreeBaseDir":"TrackHitFilter",
        "DMRMethod":"medianX",
        "DMRMinimum":"30"
        }
    try:
       for option in config.options("general"):
          result[option] = config.get("general",option)
           
       if "localGeneral" in config.sections():
          for option in result:
             if option in [item[0] for item in config.items("localGeneral")]:
                result[ option ] = config.get("localGeneral", option)
    except ConfigParser.NoSectionError, section:
       raise StandardError, "missing section '%s' in configuration files. This section is mandatory."%section
    return result

def runJob(jobName, script, config):
    general = readGeneral( config )
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
    general = readGeneral( config )
    config.set("general","workdir",os.path.join(general["workdir"],options.Name) )
    config.set("general","datadir",os.path.join(general["datadir"],options.Name) )
    config.set("general","logdir",os.path.join(general["logdir"],options.Name) )
    if not os.path.exists( outPath ):
        os.makedirs( outPath )
    elif not os.path.isdir( outPath ):
        raise StandardError,"the file %s is in the way rename the Job or move it away"%outPath

    #replace default templates by the once specified in the "alternateTemplates" section
    loadTemplates( config )

    log = ""
    alignments = readAlignments( config )
    validations = []
    for alignment in alignments:
        alignment.restrictTo( options.restrictTo )
        validations.extend( alignment.createValidations( config, options, alignments ) )

    scripts = []
    for validation in validations:
        validation.createConfiguration( outPath )
        scripts.extend( validation.createScript( outPath ) )
    
    createMergeScript( outPath, validations )

    #save backup configuration file
    backupConfigFile = open( os.path.join( outPath, "usedConfiguration.ini" ) , "w"  )
    config.write( backupConfigFile )

    for script in scripts:
        name = os.path.splitext( os.path.basename( script ) )[0]
        if options.dryRun:
            print "%s would run: %s"%( name, script)
        else:
            runJob( name, script, config)

if __name__ == "__main__":        
   # main(["-n","-N","test","-c","defaultCRAFTValidation.ini,latestObjects.ini","--getImages"])
   main()

