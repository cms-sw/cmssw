import os
import ConfigParser # needed for exceptions in this module
import configTemplates
from genericValidation import GenericValidation
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class GeometryComparison(GenericValidation):
    """
    Object representing a geometry comparison job.
    """
    def __init__( self, valName, alignment, referenceAlignment,
                  config, copyImages = True, randomWorkdirPart = None):
        """
        Constructor of the GeometryComparison class.

        Arguments:
        - `valName`: String which identifies individual validation instances
        - `alignment`: `Alignment` instance to validate
        - `referenceAlignment`: `Alignment` instance which is compared
                                with `alignment`
        - `config`: `BetterConfigParser` instance which includes the
                    configuration of the validations
        - `copyImages`: Boolean which indicates whether png- and pdf-files 
                        should be copied back from the batch farm
        - `randomWorkDirPart`: If this option is ommitted a random number is
                               generated to create unique path names for the
                               individual validation instances.
        """
        GenericValidation.__init__(self, valName, alignment, config)
        if not randomWorkdirPart == None:
            self.randomWorkdirPart = randomWorkdirPart
        self.referenceAlignment = referenceAlignment
        try:  # try to override 'jobmode' from [general] section
            self.jobmode = config.get( "compare:"+self.name, "jobmode" )
        except ConfigParser.NoOptionError:
            pass
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name

        allCompares = config.getCompares()
        self.__compares = {}
        if valName in allCompares:
            self.__compares[valName] = allCompares[valName]
        else:
            msg = ("Could not find compare section '%s' in '%s'"
                   %(valName, allCompares))
            raise AllInOneError(msg)
        self.copyImages = copyImages
    
    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = GenericValidation.getRepMap( self, alignment )
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name
        
        repMap.update({
            "comparedGeometry": (".oO[alignmentName]Oo."
                                 "ROOTGeometry.root"),
            "referenceGeometry": "IDEAL", # will be replaced later
                                          #  if not compared to IDEAL
            "reference": referenceName
            })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = (".oO[reference]Oo."
                                           "ROOTGeometry.root")
        repMap["name"] += "_vs_.oO[reference]Oo."
        return repMap

    def createConfiguration(self, path ):
        # self.__compares
        repMap = self.getRepMap()
        cfgs = { "TkAlCompareToNTuple.%s.%s_cfg.py"%(
            self.alignmentToValidate.name, self.randomWorkdirPart ):
                replaceByMap( configTemplates.intoNTuplesTemplate, repMap)}
        if not self.referenceAlignment == "IDEAL":
            referenceRepMap = self.getRepMap( self.referenceAlignment )
            cfgFileName = "TkAlCompareToNTuple.%s.%s_cfg.py"%(
                self.referenceAlignment.name, self.randomWorkdirPart )
            cfgs[cfgFileName] = replaceByMap(configTemplates.intoNTuplesTemplate,
                                             referenceRepMap)

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
            cfgName = replaceByMap(("TkAlCompareCommon.oO[common]Oo.."
                                    ".oO[name]Oo._cfg.py"),repMap)
            cfgs[cfgName] = replaceByMap(configTemplates.compareTemplate, repMap)
            
            cfgSchedule.append( cfgName )
        GenericValidation.createConfiguration(self, cfgs, path, cfgSchedule)

    def createScript(self, path):    
        repMap = self.getRepMap()    
        repMap["runComparisonScripts"] = ""
        scriptName = replaceByMap(("TkAlGeomCompare.%s..oO[name]Oo..sh"
                                   %self.name), repMap)
        for name in self.__compares:
            if  '"DetUnit"' in self.__compares[name][0].split(","):
                repMap["runComparisonScripts"] += \
                    ("rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation"
                     "/scripts/comparisonScript.C .\n"
                     "rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation"
                     "/scripts/comparisonPlots.h .\n"
                     "rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation"
                     "/scripts/comparisonPlots.cc .\n"
                     "root -b -q 'comparisonScript.C(\""
                     ".oO[name]Oo..Comparison_common"+name+".root\",\""
                     "./\")'\n")
                if  self.copyImages:
                   repMap["runComparisonScripts"] += \
                       ("rfmkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images\n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"plot*.eps\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"plot*.pdf\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name "
                        "\"TkMap_SurfDeform*.pdf\" -print | xargs -I {} bash -c"
                        " \"rfcp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name "
                        "\"TkMap_SurfDeform*.png\" -print | xargs -I {} bash -c"
                        " \"rfcp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]]\n"
                        "then\n"
                        "    rfmkdir -p .oO[workdir]Oo./.oO[name]Oo.."+name
                        +"_ArrowPlots\n"
                        "else\n"
                        "    mkdir -p $CWD/TkAllInOneTool/.oO[name]Oo.."+name
                        +"_ArrowPlots\n"
                        "fi\n")
                   repMap["runComparisonScripts"] += \
                       ("rfcp .oO[CMSSW_BASE]Oo./src/Alignment"
                        "/OfflineValidation/scripts/makeArrowPlots "
                        "$CWD/TkAllInOneTool\n"
                        "root -b -q 'makeArrowPlots.C(\""
                        ".oO[name]Oo..Comparison_common"+name
                        +".root\",\".oO[name]Oo.."
                        +name+"_ArrowPlots\")'\n")
                   repMap["runComparisonScripts"] += \
                       ("rfmkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/ArrowPlots\n")
                   repMap["runComparisonScripts"] += \
                       ("find .oO[name]Oo.."+name+"_ArrowPlots "
                        "-maxdepth 1 -name \"*.png\" -print | xargs -I {} bash "
                        "-c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/ArrowPlots\"\n")

                resultingFile = replaceByMap(("/store/caf/user/$USER/.oO[eosdir]Oo./compared%s_"
                                              ".oO[name]Oo..root"%name), repMap)
                resultingFile = os.path.expandvars( resultingFile )
                resultingFile = os.path.abspath( resultingFile )
                repMap["runComparisonScripts"] += \
                    ("cmsStage -f OUTPUT_comparison.root %s\n"
                     %resultingFile)
                self.filesToCompare[ name ] = resultingFile
                
        repMap["CommandLine"]=""

        for cfg in self.configFiles:
            # FIXME: produce this line only for enabled dbOutput
            # postProcess = "rfcp .oO[workdir]Oo./*.db .oO[datadir]Oo.\n"
            postProcess = "rfcp *.db .oO[datadir]Oo.\n"
            repMap["CommandLine"]+= \
                repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                               "postProcess":postProcess}
        repMap["CommandLine"]+= ("# overall postprocessing\n"
                                 ".oO[runComparisonScripts]Oo.\n"
                                 )

        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }  
        return GenericValidation.createScript(self, scripts, path)

    def createCrabCfg(self, path):
        msg = ("Parallelization not supported for geometry comparison. Please "
               "choose another 'jobmode'.")
        raise AllInOneError(msg)
