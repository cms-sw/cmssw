from __future__ import absolute_import
import os
import configparser as ConfigParser # needed for exceptions in this module
from . import configTemplates
from .genericValidation import GenericValidation
from .helperFunctions import replaceByMap, getCommandOutput2, cppboolstring, pythonboolstring, clean_name
from .TkAlExceptions import AllInOneError


class GeometryComparison(GenericValidation):
    """
    Object representing a geometry comparison job.
    """
    defaults = {
        "3DSubdetector1":"1",
        "3DSubdetector2":"2",
        "3DTranslationalScaleFactor":"50",
        "modulesToPlot":"all",
        "moduleList": "./CREATE_NEW/emptyModuleList.txt",
        "useDefaultRange":"false",
        "plotOnlyGlobal":"true",
        "plotPng":"true",
        "makeProfilePlots":"true",
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
        "multiIOV":"False",
        }
    mandatories = {"levels", "dbOutput"}
    valType = "compare"
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
        super(GeometryComparison, self).__init__(valName, alignment, config)
        self.referenceAlignment = referenceAlignment
        referenceName = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName = self.referenceAlignment.name

        allCompares = config.getCompares()
        self.__compares = {}
        self.__filesToCompare = {}
        if valName in allCompares:
            self.__compares[valName] = allCompares[valName]
        else:
            msg = ("Could not find compare section '%s' in '%s'"
                   %(valName, allCompares))
            raise AllInOneError(msg)
        self.copyImages = copyImages

        for name in "useDefaultRange", "plotOnlyGlobal", "plotPng":
            self.general[name] = cppboolstring(self.general[name], name)


    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = super(GeometryComparison, self).getRepMap( alignment )
        referenceName  = "IDEAL"
        referenceTitle = "IDEAL"
        if not self.referenceAlignment == "IDEAL":
            referenceName  = self.referenceAlignment.name
            referenceTitle = self.referenceAlignment.title

        assert len(self.__compares) == 1 #? not sure how it can be anything else, but just in case
        common = list(self.__compares.keys())[0]

        repMap.update({
            "common": clean_name(common),
            "comparedGeometry": (".oO[alignmentName]Oo."
                                 "ROOTGeometry.root"),
            "referenceGeometry": "IDEAL", # will be replaced later
                                          #  if not compared to IDEAL
            "reference": clean_name(referenceName),
            "referenceTitle": referenceTitle,
            "alignmentTitle": self.alignmentToValidate.title,
            "moduleListBase": os.path.basename(repMap["moduleList"]),
            })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = (".oO[reference]Oo."
                                           "ROOTGeometry.root")
        repMap["name"] += "_vs_.oO[reference]Oo."
        return repMap

    @property
    def filesToCompare(self):
        return self.__filesToCompare

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

        cfgSchedule = list(cfgs.keys())
        for common in self.__compares:
            repMap.update({
                           "levels": self.__compares[common][0],
                           "dbOutput": pythonboolstring(self.__compares[common][1], "dbOutput")
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
        super(GeometryComparison, self).createConfiguration(cfgs, path, cfgSchedule, repMaps = repMaps)

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
                    ("cp .oO[Alignment/OfflineValidation]Oo."
                     "/scripts/comparisonScript.C .\n"
                     "cp .oO[Alignment/OfflineValidation]Oo."
                     "/scripts/GeometryComparisonPlotter.h .\n"
                     "cp .oO[Alignment/OfflineValidation]Oo."
                     "/scripts/GeometryComparisonPlotter.cc .\n"
                     "root -b -q 'comparisonScript.C+(\""
                     ".oO[name]Oo..Comparison_common"+name+".root\",\""
                     "./\",\".oO[modulesToPlot]Oo.\",\".oO[alignmentName]Oo.\",\".oO[reference]Oo.\",.oO[useDefaultRange]Oo.,.oO[plotOnlyGlobal]Oo.,.oO[plotPng]Oo.,.oO[makeProfilePlots]Oo."+y_ranges+")'\n"
                     "cp "+path+"/TkAl3DVisualization_.oO[common]Oo._.oO[name]Oo..C .\n"
                     "root -l -b -q TkAl3DVisualization_.oO[common]Oo._.oO[name]Oo..C+\n")
                if  self.copyImages:
                   repMap["runComparisonScripts"] += \
                       ("mkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/Translations\n")
                   repMap["runComparisonScripts"] += \
                       ("mkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/Rotations\n")


                   ### At the moment translations are images with suffix _1 and _2, rotations _3 and _4
                   ### The numeration depends on the order of the MakePlots(x, y) commands in comparisonScript.C
                   ### If comparisonScript.C is changed, check if the following lines need to be changed as well

                   if repMap["plotPng"] == "true":
                           repMap["runComparisonScripts"] += \
                               ("find . -maxdepth 1 -name \"*_1*\" "
                                "-print | xargs -I {} bash -c \"cp {} .oO[datadir]Oo."
                                "/.oO[name]Oo..Comparison_common"+name+"_Images/Translations/\" \n")
                           repMap["runComparisonScripts"] += \
                               ("find . -maxdepth 1 -name \"*_2*\" "
                                "-print | xargs -I {} bash -c \"cp {} .oO[datadir]Oo."
                                "/.oO[name]Oo..Comparison_common"+name+"_Images/Translations/\" \n")

                           repMap["runComparisonScripts"] += \
                               ("find . -maxdepth 1 -name \"*_3*\" "
                                "-print | xargs -I {} bash -c \"cp {} .oO[datadir]Oo."
                                "/.oO[name]Oo..Comparison_common"+name+"_Images/Rotations/\" \n")
                           repMap["runComparisonScripts"] += \
                               ("find . -maxdepth 1 -name \"*_4*\" "
                                "-print | xargs -I {} bash -c \"cp {} .oO[datadir]Oo."
                                "/.oO[name]Oo..Comparison_common"+name+"_Images/Rotations/\" \n")

                   else:
                           repMap["runComparisonScripts"] += \
                               ("find . -maxdepth 1 -name \"*_1*\" "
                                "-print | xargs -I {} bash -c \"cp {} .oO[datadir]Oo."
                                "/.oO[name]Oo..Comparison_common"+name+"_Images/Translations/\" \n")

                           repMap["runComparisonScripts"] += \
                               ("find . -maxdepth 1 -name \"*_2*\" "
                                "-print | xargs -I {} bash -c \"cp {} .oO[datadir]Oo."
                                "/.oO[name]Oo..Comparison_common"+name+"_Images/Rotations/\" \n")

                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name "
                        "\"*.tex\" -print | xargs -I {} bash -c"
                        " \"cp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name "
                        "\"TkMap_SurfDeform*.pdf\" -print | xargs -I {} bash -c"
                        " \"cp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("find . -maxdepth 1 -name "
                        "\"TkMap_SurfDeform*.png\" -print | xargs -I {} bash -c"
                        " \"cp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/\" \n")
                   repMap["runComparisonScripts"] += \
                       ("cp .oO[Alignment/OfflineValidation]Oo."
                        "/macros/makeArrowPlots.C "
                        ".\n"
                        "root -b -q 'makeArrowPlots.C(\""
                        ".oO[name]Oo..Comparison_common"+name
                        +".root\",\".oO[name]Oo.."
                        +name+"_ArrowPlots\")'\n")
                   repMap["runComparisonScripts"] += \
                       ("mkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/ArrowPlots\n")
                   repMap["runComparisonScripts"] += \
                       ("find .oO[name]Oo.."+name+"_ArrowPlots "
                        "-maxdepth 1 -name \"*.png\" -print | xargs -I {} bash "
                        "-c \"cp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/ArrowPlots\"\n")
                   repMap["runComparisonScripts"] += \
                       ("find .oO[name]Oo.."+name+"_ArrowPlots "
                        "-maxdepth 1 -name \"*.pdf\" -print | xargs -I {} bash "
                        "-c \"cp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/ArrowPlots\"\n")
                   repMap["runComparisonScripts"] += \
                       ("find . "
                        "-maxdepth 1 -name \".oO[common]Oo._.oO[name]Oo..Visualization_rotated.gif\" -print | xargs -I {} bash "
                        "-c \"cp {} .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/.oO[common]Oo._.oO[name]Oo..Visualization.gif\"\n")

                   # TkAlMap inFile=tree.root compAl=UL2018 refAl=StartGeom savePNG=True TkVersion=phase1 outDir=./test_plots/tanh colPal=2
                   range_str = ''
                   plottedDifferences = ["dx","dy","dz","dr","rdphi","dalpha","dbeta","dgamma"]
                   for diff in plottedDifferences:
                       range_str += diff+'_range=['+str(repMap[diff+'_min'])+','+str(repMap[diff+'_max'])+'];'
                   repMap["runComparisonScripts"] += \
                       ("mkdir -p .oO[datadir]Oo./.oO[name]Oo."
                        ".Comparison_common"+name+"_Images/TkAlMapPlots\n")
                   repMap["runComparisonScripts"] += \
                       ("python .oO[Alignment/OfflineValidation]Oo./python/runGCPTkAlMap.py -b " 
                        "inFile=.oO[name]Oo..Comparison_common"+name+".root "
                        "refAl=\".oO[reference]Oo.\" " 
                        "compAl=\".oO[alignmentName]Oo.\" "
                        "savePNG=True "
                        "TkVersion=\"phase0\" "
                        "colPal=2 "
                        "defRanges=\""+range_str+"\" "
                        "outDir=.oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/TkAlMapPlots\n")
                        #"outDir=.oO[name]Oo.."+name+"_TkMapPlots "
                        #"useDefaultRanges=.oO[useDefaultRange]Oo. "+range_str+"\n")

                   # Copy root file for check
                   repMap["runComparisonScripts"] += \
                       ("cp .oO[name]Oo..Comparison_common"+name+".root "
                        ".oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/TkAlMapPlots/GCP.root\n")
                   #repMap["runComparisonScripts"] += \
                   #    ("cp .oO[alignmentName]Oo.ROOTGeometry.root "
                   #     ".oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/TkAlMapPlots/comparedGeometry.root\n")

                resultingFile = replaceByMap(("/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./compared%s_"
                                              ".oO[name]Oo..root"%name), repMap)
                resultingFile = os.path.expandvars( resultingFile )
                resultingFile = os.path.abspath( resultingFile )
                resultingFile = "root://eoscms//eos/cms" + resultingFile   #needs to be AFTER abspath so that it doesn't eat the //
                self.__filesToCompare[ name ] = resultingFile

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
        elif repMap["moduleList"].startswith("./CREATE_NEW/"):
            repMap["CommandLine"]+= \
                 "touch .oO[moduleListBase]Oo.\n"
        else:
            repMap["CommandLine"]+= \
                     "cp .oO[moduleList]Oo. .\n"

        try:
            getCommandOutput2(replaceByMap("cd $(mktemp -d)\n.oO[CommandLine]Oo.\ncat .oO[moduleListBase]Oo.", repMap))
        except RuntimeError:
            raise AllInOneError(replaceByMap(".oO[moduleList]Oo. does not exist!", repMap))

        for cfg in self.configFiles:
            # FIXME: produce this line only for enabled dbOutput
            # postProcess = "cp .oO[workdir]Oo./*.db .oO[datadir]Oo.\n"
            # postProcess = "cp *.db .oO[datadir]Oo.\n"
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
        return super(GeometryComparison, self).createScript(scripts, path)

    def createCrabCfg(self, path):
        msg = ("Parallelization not supported for geometry comparison. Please "
               "choose another 'jobmode'.")
        raise AllInOneError(msg)
