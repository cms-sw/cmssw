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
	defaults = {
	    "3DSubdetector1":"1",
	    "3DSubdetector2":"2",
	    "3DTranslationalScaleFactor":"50"
            }
        mandatories = ["levels", "dbOutput"]
        GenericValidation.__init__(self, valName, alignment, config, 
				   "compare", addDefaults=defaults, 
				   addMandatories = mandatories)
        if not randomWorkdirPart == None:
            self.randomWorkdirPart = randomWorkdirPart
        self.referenceAlignment = referenceAlignment
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
            "reference": referenceName,
            "referenceTitle": self.referenceAlignment.title,
	    "alignmentTitle": self.alignmentToValidate.title
            })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = (".oO[reference]Oo."
                                           "ROOTGeometry.root")
        repMap["name"] += "_vs_.oO[reference]Oo."
        return repMap

    def createConfiguration(self, path ):
        # self.__compares
        repMap = self.getRepMap()
        cfgFileName = "TkAlCompareToNTuple.%s.%s_cfg.py"%(
            self.alignmentToValidate.name, self.randomWorkdirPart)
        cfgs = {cfgFileName: configTemplates.intoNTuplesTemplate}
        repMaps = {cfgFileName: repMap}
        if not self.referenceAlignment == "IDEAL":
            referenceRepMap = self.getRepMap( self.referenceAlignment )
            cfgFileName = "TkAlCompareToNTuple.%s.%s_cfg.py"%(
                self.referenceAlignment.name, self.randomWorkdirPart )
            cfgs[cfgFileName] = configTemplates.intoNTuplesTemplate
            repMaps[cfgFileName] = referenceRepMap

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
            cfgs[cfgName] = configTemplates.compareTemplate
            repMaps[cfgName] = repMap

            cfgSchedule.append( cfgName )  
        GenericValidation.createConfiguration(self, cfgs, path, cfgSchedule, repMaps = repMaps)

    def createScript(self, path):
        repMap = self.getRepMap()
        repMap["runComparisonScripts"] = ""
        scriptName = replaceByMap(("TkAlGeomCompare.%s..oO[name]Oo..sh"
                                   %self.name), repMap)
        for name in self.__compares:
            if  '"DetUnit"' in self.__compares[name][0].split(","):
                repMap["outputFile"] = (".oO[name]Oo..Comparison_common"+name+".root")
                repMap["nIndex"] = ("")
                repMap["runComparisonScripts"] += \
                    ("rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation"
                     "/scripts/comparisonScript.C .\n"
                     "rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation"
                     "/scripts/GeometryComparisonPlotter.h .\n"
                     "rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation"
                     "/scripts/GeometryComparisonPlotter.cc .\n"
                     "root -b -q 'comparisonScript.C+(\""
                     ".oO[name]Oo..Comparison_common"+name+".root\",\""
                     "./\")'\n"
		     "rfcp "+path+"/TkAl3DVisualization_.oO[name]Oo..C .\n"
		     "root -l -b -q TkAl3DVisualization_.oO[name]Oo..C+\n")
                if  self.copyImages:
                   repMap["runComparisonScripts"] += \
                       ("rfmkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images\n")
                   repMap["runComparisonScripts"] += \
                       ("rfmkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/Translations\n")
                   repMap["runComparisonScripts"] += \
                       ("rfmkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/Rotations\n")
                   repMap["runComparisonScripts"] += \
                       ("rfmkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/CrossTalk\n")


                   ### At the moment translations are immages with suffix _1 and _2, rotations _3 and _4, and cross talk _5 and _6
                   ### The numeration depends on the order of the MakePlots(x, y) commands in comparisonScript.C
                   ### If comparisonScript.C is changed, check if the following lines need to be changed as well
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"*_1*\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/Translations/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"*_2*\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/Translations/\" \n")
                   
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"*_3*\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/Rotations/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"*_4*\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/Rotations/\" \n")
                   
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"*_5*\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/CrossTalk/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name \"*_6*\" "
                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
                        "/.oO[name]Oo..Comparison_common"+name+"_Images/CrossTalk/\" \n")
                   
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
                        "/OfflineValidation/scripts/makeArrowPlots.C "
                        ".\n"
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
		   repMap["runComparisonScripts"] += \
                       ("find . "
                        "-maxdepth 1 -name \".oO[name]Oo..Visualization_rotated.gif\" -print | xargs -I {} bash "
                        "-c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/.oO[name]Oo..Visualization.gif\"\n")

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
            # postProcess = "rfcp *.db .oO[datadir]Oo.\n"
            postProcess = ""
            repMap["CommandLine"]+= \
                repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                               "postProcess":postProcess}
        repMap["CommandLine"]+= ("# overall postprocessing\n"
                                 ".oO[runComparisonScripts]Oo.\n"
                                 )

        #~ print configTemplates.scriptTemplate
        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap )}
	files = {replaceByMap("TkAl3DVisualization_.oO[name]Oo..C", repMap ): replaceByMap(configTemplates.visualizationTrackerTemplate, repMap )}
	self.createFiles(files, path)
        return GenericValidation.createScript(self, scripts, path)

    def createCrabCfg(self, path):
        msg = ("Parallelization not supported for geometry comparison. Please "
               "choose another 'jobmode'.")
        raise AllInOneError(msg)
