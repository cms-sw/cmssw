import os
import ConfigParser # needed for exceptions in this module
import configTemplates
from genericValidation import GenericValidation
from helperFunctions import replaceByMap


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
        try:
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
            raise AllInOneError, "Could not find compare section '%s' in '%s'"%(valName, allCompares)
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
            "reference": referenceName
            })
        if not referenceName == "IDEAL":
            repMap["referenceGeometry"] = (".oO[workdir]Oo./.oO[reference]Oo."
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
        scriptName = replaceByMap( "TkAlGeomCompare.%s..oO[name]Oo..sh"%( self.name ),
                                   repMap)
        for name in self.__compares:
            if  '"DetUnit"' in self.__compares[name][0].split(","):
                repMap["runComparisonScripts"] += "root -b -q 'comparisonScript.C(\".oO[workdir]Oo./.oO[name]Oo..Comparison_common"+name+".root\",\".oO[workdir]Oo./\")'\n"
                if  self.copyImages:
                   repMap["runComparisonScripts"] += "rfmkdir -p .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images\n"
                   repMap["runComparisonScripts"] += "find .oO[workdir]Oo. -maxdepth 1 -name \"plot*.eps\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/\" \n"
                   repMap["runComparisonScripts"] += "find .oO[workdir]Oo. -maxdepth 1 -name \"plot*.pdf\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/\" \n"
                   repMap["runComparisonScripts"] += "find .oO[workdir]Oo. -maxdepth 1 -name \"TkMap_SurfDeform*.pdf\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/\" \n"
                   repMap["runComparisonScripts"] += "find .oO[workdir]Oo. -maxdepth 1 -name \"TkMap_SurfDeform*.png\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo..Comparison_common"+name+"_Images/\" \n"
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

    def createCrabCfg( self ):
        raise AllInOneError, ("Parallelization not supported for geometry "
                              "comparison. Please choose another 'jobmode'.")
