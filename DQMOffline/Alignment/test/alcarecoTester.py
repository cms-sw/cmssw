#!/usr/bin/env python3

from __future__ import print_function
import os
import sys
import configparser as ConfigParser
import optparse
import datetime

scriptTemplate = """
#!/bin/bash
#init
export STAGE_SVCCLASS=cmscaf
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd %(CMSSW_BASE)s
eval `scramv1 ru -sh`
rfmkdir -p %(workDir)s
rm -f %(workDir)s/*
cd %(workDir)s

#run
pwd
df -h .
#run configfile and post-proccess it
%(commandLine)s &> cmsRunOut.%(name)s.log

echo "----"
echo "List of files in $(pwd):"
ls -ltr
echo "----"
echo ""

#retrive
rfmkdir %(outDir)s
rfmkdir %(outDir)s/%(name)s
gzip cmsRunOut.%(name)s.log
rfmkdir %(logDir)s
rfcp cmsRunOut.%(name)s%(isn)s.log.gz %(logDir)s
for i in *.root
do
   rfcp $i %(outDir)s/%(name)s/${i/.root}%(isn)s.root
done
   
#cleanup
#rm -rf %(workDir)s
echo "done."
"""

extractDQMtemplate = """
import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(%(AlCARecos)s)
)

process.p1 = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/ConverterTester/Test/RECO'
"""

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

class Sample:
   def __init__(self, name, config):
      self.name = name
      self.files = config.get("sample:%s"%name, "files").split("\n")
      self.globalTag = config.get("sample:%s"%name, "globalTag")
      self.maxEvents = config.get("sample:%s"%name, "maxEvents")
      self.AlCaRecos = config.get("sample:%s"%name, "AlCaRecos")
      self.AlcaRecoScripts = None
      self.DQMScript = None
      self.DQMConfig = None
      self.__config = config
   
   def createScript(self, path, style, commandLine, isn = None):
      repMap = {
                   "name": self.name,
                   "workDir": self.__config.get("general", "workDir"),
                   "outDir": self.__config.get("general", "outDir"),
                   "logDir": self.__config.get("general", "logDir"),
                   "CMSSW_BASE": os.environ["CMSSW_BASE"],
                   "commandLine":commandLine
                   }
      fileName = "%s_%s"%(style,self.name)
      if not isn == None:
         repMap["isn"] = "_%03i"%isn
         fileName += repMap["isn"]
      else:
         repMap["isn"] = ""
         
      fileName += ".sh"
      scriptPath = os.path.join(path, fileName)
      scriptFile = open( scriptPath, "w" )
      scriptFile.write( scriptTemplate%repMap )
      scriptFile.close()
      os.chmod(scriptPath, 0o755)
      return scriptPath
   
   def createDQMExtract(self, path):
      filePrefix = "ALCARECO%s"%self.AlCaRecos.split("+")[0]

      alcaFiles = []
      for i in range( 0,len(self.files) ):
         alcaFilePath = os.path.join( self.__config.get("general","outDir"), self.name,"%s_%03i.root"%(filePrefix,i+1) )
         alcaFilePath = os.path.expandvars(alcaFilePath)
         alcaFilePath = os.path.abspath(alcaFilePath)   
         if alcaFilePath.startswith("/castor"):
            alcaFilePath = "'rfio:"+alcaFilePath+"'"
         else:
            alcaFilePath = "'file:"+alcaFilePath+"'"
         alcaFiles.append(alcaFilePath)
      repMap = { "AlCARecos":", ".join(alcaFiles)}   
      fileName = "extractDQM_%s_cfg.py"%self.name
      cfgPath = os.path.join(path, fileName)
      cfgFile = open( cfgPath, "w" )
      cfgFile.write( extractDQMtemplate%repMap )
      cfgFile.close()
      return cfgPath
   
   def createScripts(self, path):
      result = []
      repMap = {"maxEvents":self.maxEvents,
                "globalTag":self.globalTag,
                "AlCaRecos":self.AlCaRecos,
                "sampleFile":"" }
      i = 1
      for sampleFile in self.files:
         repMap["sampleFile"] = sampleFile
         result.append(self.createScript(path, "AlCaReco", """
cmsDriver.py step3 -s ALCA:%(AlCaRecos)s+DQM \\
    -n %(maxEvents)s \\
    --filein %(sampleFile)s\\
    --conditions FrontierConditions_GlobalTag,%(globalTag)s \\
    --eventcontent RECO\\
    --mc"""%repMap, i))
         i += 1
      self.AlcaRecoScripts = result
      self.DQMConfig = self.createDQMExtract(path)
      self.DQMScript = self.createScript(path, "extractDQM", "cmsRun %s"%self.DQMConfig)
      return result

######################### Helpers ####################
#excute [command] and return output
def getCommandOutput2(command):
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError('%s failed w/ exit code %d' % (command, err))
    return data
 
def runJob(jobName, script, config):
   jobMode = config.get("general","jobMode")
   print("> Testing "+jobName)
   if jobMode == "interactive":
       getCommandOutput2( script )
   if jobMode.split(",")[0] == "lxBatch":
       commands = jobMode.split(",")[1]
       getCommandOutput2("bsub "+commands+" -J "+jobName+" "+ script)     

def readSamples( config):
   result = []
   for section in config.sections():
      if "sample:" in section:
         name = section.split("sample:")[1]
         result.append( Sample(name, config) )
   return result

def main(argv = None):
    if argv == None:
       argv = sys.argv[1:]
    optParser = optparse.OptionParser()
    optParser.description = "test the tracker AlCaReco production and DQM"
    optParser.add_option("-n", "--dryRun", dest="dryRun", action="store_true", default=False,
                         help="create all scripts and cfg File but do not start jobs (default=False)")
    optParser.add_option("-c", "--config", dest="config",
                         help="configuration to use (default testConfig.ini) this can be a comma-seperated list of all .ini file you want to merge", metavar="CONFIG")
    optParser.add_option("-N", "--Name", dest="Name",
                         help="Name of this test (default: test_DATE_TIME)", metavar="NAME")

    (options, args) = optParser.parse_args(argv)

    if options.config == None:
        options.config = "testConfig.ini"
    else:
        options.config = options.config.split(",")
        result = []
        for iniFile in options.config:
            result.append( os.path.abspath(iniFile) )
        options.config = result
    
    config = BetterConfigParser()
    config.read( options.config )

    if options.Name == None:
        options.Name = "test_%s"%(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

    outPath = os.path.abspath( options.Name )
    for dir in [ "workDir","dataDir","logDir" ]:
       if not config.has_option("general", dir):
          config.set("general", dir, os.path.join( os.getcwd(), options.Name ))
       else:
          config.set("general", dir, os.path.join( config.get("general",dir), options.Name ))

    if not os.path.exists( outPath ):
        os.makedirs( outPath )
    elif not os.path.isdir( outPath ):
        raise "the file %s is in the way rename the Job or move it away"%outPath
     
    samples = readSamples( config )
    for sample in samples:
       sample.createScripts( outPath )
       for scriptPath in sample.AlcaRecoScripts:
          if not options.dryRun:
             runJob( sample.name, scriptPath, config )
     
if __name__ == "__main__":        
    main()
