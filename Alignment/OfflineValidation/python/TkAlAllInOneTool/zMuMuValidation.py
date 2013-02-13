import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class ZMuMuValidation(GenericValidationData):
    def __init__(self, valName, alignment,config):
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "zmumu")
        defaults = {
            "zmumureference": ("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2"
                               "/TMP_EM/ZMuMu/data/MC/BiasCheck_DYToMuMu_Summer"
                               "11_TkAlZMuMu_IDEAL.root")
            }
        mandatories = [ "dataset", "maxevents",
                        "etamaxneg", "etaminneg", "etamaxpos", "etaminpos" ]
        if not config.has_section( "zmumu:"+self.name ):
            zmumu = config.getResultingSection( "general",
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        else:
            zmumu = config.getResultingSection( "zmumu:"+self.name, 
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        self.general.update( zmumu )
    
    def createConfiguration(self, path, configBaseName = "TkAlZMuMuValidation" ):
        cfgName = "%s.%s.%s_cfg.py"%( configBaseName, self.name,
                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
        cfgs = {cfgName:replaceByMap( configTemplates.ZMuMuValidationTemplate, repMap)}
        GenericValidationData.createConfiguration(self, cfgs, path)
        
    def createScript(self, path, scriptBaseName = "TkAlZMuMuValidation"):
        scriptName = "%s.%s.%s.sh"%(scriptBaseName, self.name,
                                    self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: replaceByMap( configTemplates.zMuMuScriptTemplate, repMap ) }
        return GenericValidationData.createScript(self, scripts, path)

    def createCrabCfg( self, path,
                       crabCfgBaseName = "TkAlZMuMuValidation"  ):
        crabCfgName = "crab.%s.%s.%s.cfg"%( crabCfgBaseName, self.name,
                                            self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["script"] = "dummy_script.sh"
        repMap["crabOutputDir"] = os.path.basename( path )
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
        crabCfg = {crabCfgName: replaceByMap( configTemplates.crabCfgTemplate,
                                              repMap ) }
        return GenericValidationData.createCrabCfg( self, crabCfg, path )

    def getRepMap(self, alignment = None):
        repMap = GenericValidationData.getRepMap(self, alignment) 
        repMap.update({
            "nEvents": self.general["maxevents"],
#             "outputFile": "zmumuHisto.root"
            "outputFile": ("0_zmumuHisto.root"
                           ",genSimRecoPlots.root"
                           ",FitParameters.txt")
                })
        return repMap
