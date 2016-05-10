import os
import ConfigParser # needed for exceptions in this module
import configTemplates
from genericValidation import GenericValidation
from helperFunctions import replaceByMap, getCommandOutput2
from TkAlExceptions import AllInOneError


class GeometryComparison(GenericValidation):
    """
    Object representing a geometry comparison job.
    """
    def __init__( self, valName, alignment, referenceAlignment,
                  config, copyImages = True):
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
        """
	defaults = {
	    "3DSubdetector1":"1",
	    "3DSubdetector2":"2",
	    "3DTranslationalScaleFactor":"50",
	    "modulesToPlot":"all",
	    "moduleList": "/store/caf/user/cschomak/emptyModuleList.txt",
	    "useDefaultRange":"false",
	    "plotOnlyGlobal":"false",
	    "plotPng":"true",
	    "dx_min":"-99999",
	    "dx_max":"-99999",
	    "dy_min":"-99999",
	    "dy_max":"-99999",
	    "dz_min":"-99999",
	    "dz_max":"-99999",
	    "dr_min":"-99999",
	    "dr_max":"-99999",
	    "rdphi_min":"-99999",
	    "rdphi_max":"-99999",
	    "dalpha_min":"-99999",
	    "dalpha_max":"-99999",
	    "dbeta_min":"-99999",
	    "dbeta_max":"-99999",
	    "dgamma_min":"-99999",
	    "dgamma_max":"-99999",
            }
        mandatories = ["levels", "dbOutput"]
        GenericValidation.__init__(self, valName, alignment, config, 
				   "compare", addDefaults=defaults, 
				   addMandatories = mandatories)
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
        referenceName  = "IDEAL"
        referenceTitle = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName  = self.referenceAlignment.name
            referenceTitle = self.referenceAlignment.title

        assert len(self.__compares) == 1 #? not sure how it can be anything else, but just in case
        common = self.__compares.keys()[0]

        repMap.update({
            "common": common,
            "comparedGeometry": (".oO[alignmentName]Oo."
                                 "ROOTGeometry.root"),
            "referenceGeometry": "IDEAL", # will be replaced later
                                          #  if not compared to IDEAL
            "reference": referenceName,
            "referenceTitle": referenceTitle,
	    "alignmentTitle": self.alignmentToValidate.title,
            "moduleListBase": os.path.basename(repMap["moduleList"]),
            })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = (".oO[reference]Oo."
                                           "ROOTGeometry.root")
        repMap["name"] += "_vs_.oO[reference]Oo."
        return repMap

    def createConfiguration(self, path ):
        # self.__compares
        repMap = self.getRepMap()
        cfgFileName = "TkAlCompareToNTuple.%s_cfg.py"%(
            self.alignmentToValidate.name)
        cfgs = {cfgFileName: configTemplates.intoNTuplesTemplate}
        repMaps = {cfgFileName: repMap}
        if not self.referenceAlignment == "IDEAL":
            referenceRepMap = self.getRepMap( self.referenceAlignment )
            cfgFileName = "TkAlCompareToNTuple.%s_cfg.py"%(
                self.referenceAlignment.name )
            cfgs[cfgFileName] = configTemplates.intoNTuplesTemplate
            repMaps[cfgFileName] = referenceRepMap

        cfgSchedule = cfgs.keys()
        for common in self.__compares:
            repMap.update({
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
        
        y_ranges = ""
        plottedDifferences = ["dx","dy","dz","dr","rdphi","dalpha","dbeta","dgamma"]
        for diff in plottedDifferences:
			y_ranges += ","+repMap["%s_min"%diff]
			y_ranges += ","+repMap["%s_max"%diff]
			
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
                     "./\",\".oO[modulesToPlot]Oo.\",\".oO[alignmentName]Oo.\",\".oO[reference]Oo.\",\".oO[useDefaultRange]Oo.\",\".oO[plotOnlyGlobal]Oo.\",\".oO[plotPng]Oo.\""+y_ranges+")'\n"
		     "rfcp "+path+"/TkAl3DVisualization_.oO[common]Oo._.oO[name]Oo..C .\n"
		     "root -l -b -q TkAl3DVisualization_.oO[common]Oo._.oO[name]Oo..C+\n")
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


                   ### At the moment translations are images with suffix _1 and _2, rotations _3 and _4
                   ### The numeration depends on the order of the MakePlots(x, y) commands in comparisonScript.C
                   ### If comparisonScript.C is changed, check if the following lines need to be changed as well
                   
                   if repMap["plotPng"] == "true":
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
	                        
                   else:
	                   repMap["runComparisonScripts"] += \
	                       ("find . -maxdepth 1 -name \"*_1*\" "
	                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
	                        "/.oO[name]Oo..Comparison_common"+name+"_Images/Translations/\" \n")
	                   
	                   repMap["runComparisonScripts"] += \
	                       ("find . -maxdepth 1 -name \"*_2*\" "
	                        "-print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo."
	                        "/.oO[name]Oo..Comparison_common"+name+"_Images/Rotations/\" \n")
                   
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
                        "-maxdepth 1 -name \".oO[common]Oo._.oO[name]Oo..Visualization_rotated.gif\" -print | xargs -I {} bash "
                        "-c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/.oO[common]Oo._.oO[name]Oo..Visualization.gif\"\n")

                resultingFile = replaceByMap(("/store/caf/user/$USER/.oO[eosdir]Oo./compared%s_"
                                              ".oO[name]Oo..root"%name), repMap)
                resultingFile = os.path.expandvars( resultingFile )
                resultingFile = os.path.abspath( resultingFile )
                resultingFile = "root://eoscms//eos/cms" + resultingFile   #needs to be AFTER abspath so that it doesn't eat the //
                repMap["runComparisonScripts"] += \
                    ("xrdcp -f OUTPUT_comparison.root %s\n"
                     %resultingFile)
                self.filesToCompare[ name ] = resultingFile

            else:
                raise AllInOneError("Need to have DetUnit in levels!")

        repMap["CommandLine"]=""
        repMap["CommandLine"]+= \
                 "# copy module list required for comparison script \n"
        if repMap["moduleList"].startswith("/store"):
            repMap["CommandLine"]+= \
                 "xrdcp root://eoscms//eos/cms.oO[moduleList]Oo. .\n"
        elif repMap["moduleList"].startswith("root://"):
            repMap["CommandLine"]+= \
                 "xrdcp .oO[moduleList]Oo. .\n"
        else:
            repMap["CommandLine"]+= \
                     "rfcp .oO[moduleList]Oo. .\n"

        try:
            getCommandOutput2(replaceByMap("cd $(mktemp -d)\n.oO[CommandLine]Oo.\ncat .oO[moduleListBase]Oo.", repMap))
        except RuntimeError:
            raise AllInOneError(replaceByMap(".oO[moduleList]Oo. does not exist!", repMap))

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
	files = {replaceByMap("TkAl3DVisualization_.oO[common]Oo._.oO[name]Oo..C", repMap ): replaceByMap(configTemplates.visualizationTrackerTemplate, repMap )}
	self.createFiles(files, path)
        return GenericValidation.createScript(self, scripts, path)

    def createCrabCfg(self, path):
        msg = ("Parallelization not supported for geometry comparison. Please "
               "choose another 'jobmode'.")
        raise AllInOneError(msg)
