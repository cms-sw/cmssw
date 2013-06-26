#!/usr/bin/env python
# Original Author: James Jackson
# $Id: HltComparatorCreateWorkflow.py,v 1.2 2009/07/19 14:34:19 wittich Exp $
# $Log: HltComparatorCreateWorkflow.py,v $
# Revision 1.2  2009/07/19 14:34:19  wittich
# Add HltComparator file back in for 3.1.
# Tweaks to scripts and cfg files for HLT re-running.
#

# MakeValidationConfig.py
#   Makes CMSSW config for prompt validation of a given run number,
#   HLT Key or HLT Config

from optparse import OptionParser
import os, time, re

jobHash = "%s_%s_%s" % (os.getuid(), os.getpid(), int(time.time()))

usage = '%prog [options]. \n\t-a and -k required, -h for help.'
parser = OptionParser(usage)
parser.add_option("-a", "--analysis", dest="analysis", 
                  help="analysis configuration file")
# not yet implemented
#parser.add_option("-r", "--run", dest="run", 
#                  help="construct config for runnumber RUN", metavar="RUN")
parser.add_option("-k", "--hltkey", dest="hltkey", 
                  help="ignore RunRegistry and force the use of HLT key KEY", 
                  metavar="KEY")
parser.add_option("-c", "--hltcff", dest="hltcff", 
                  help="use the config fragment CFF to define HLT configuration", 
                  metavar="CFF")
parser.add_option("-f", "--frontier", dest="frontier", 
                  help="frontier connection string to use, defaults to frontier://FrontierProd/CMS_COND_21X_GLOBALTAG", 
                  default="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG")

# Parse options and perform sanity checks
(options, args) = parser.parse_args()
#if options.run == None and options.hltkey == None and options.hltcff == None:
if options.hltkey == None and options.hltcff == None:
   parser.error("I don't have all the required options.")
   raise SystemExit(
      "Please specify one of --hltkey (-k) or --hltcff (-s)")
if options.hltkey != None and options.hltcff != None:
   raise SystemExit("Please only specify --hltkey (-k) or --hltcff (-c)")
# if options.run != None and options.hltkey != None:
#    raise SystemExit("Please only specify --run (-r) or --hltkey (-k)")
# if options.run != None and options.hltcff != None: 
#    raise SystemExit("Please only specify --run (-r) or --hltcff (-c)")
if options.analysis == None:
   raise SystemExit(
      "Please specify an analysis configuration: -a or --analysis")

# Checks the runtime environment is suitable
def CheckEnvironment():
   cwd = os.getcwd()
   t = cwd.split("/")
   if t[-1] != "python":
      raise SystemExit("Must run from a Module/Package/python directory")

# Converts an online HLT configuration into offline suitable format
def ConvertHltOnlineToOffline(config, frontierString):
   # Replace the frontier string
   onlineFrontier = re.search('"(frontier:.*)"', config)
   if not onlineFrontier:
      print "WARNING: Could not find Frontier string in HLT configuration. Will ignore."
   else:
      config = config.replace(onlineFrontier.group(1), frontierString)

   # Replace the global tag
   config = config.replace("H::All", "P::All")

   # print config -- debugging
   # Remove unwanted PSets
   config = RemovePSet(config, "MessageLogger")
#   config = RemovePSet(config, "DQMStore")
   config = RemovePSet(config, "DQM")
   config = RemovePSet(config, "FUShmDQMOutputService")

   return config

def RemovePSet(config, pset):
   startLoc = config.find(pset)
   started = False
   count = 0
   curLoc = startLoc
   endLoc = 0

   # Find starting parenthesis
   while not started:
      if config[curLoc] == "(":
         started = True
         count = 1
      curLoc += 1

   # Find end parenthesis
   while endLoc == 0:
      if config[curLoc] == "(":
         count += 1
      elif config[curLoc] == ")":
         count -= 1
      if count == 0:
         endLoc = curLoc
      curLoc += 1

   config = config.replace(config[startLoc:endLoc + 1], "")

   return config

# Fetches the HLT configuration from ConfDB
def GetHltConfiguration(hltKey, frontierString):
   # Format the config path for include
   cwd = os.getcwd()
   t = cwd.split("/")
   module = t[-3]
   package = t[-2]

   # Get the HLT config
   config = os.popen2('wget "http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/get.jsp?dbName=ORCOFF&configName=%s&cff=&nooutput=&format=Python" -O- -o /dev/null' % hltKey)[1].read()
   #config = os.popen2("edmConfigFromDB --debug --configName %s --orcoff --format Python --nooutput --cff" % hltKey)[1].read() 
   configName = "JobHLTConfig_%s_cff.py" % jobHash

   # Perform online --> offline conversion
   config = ConvertHltOnlineToOffline(config, frontierString)

   # Write the config
   f = open(configName, "w")
   f.write(config)
   f.close()

   return 'process.load("%s.%s.%s")' % (module, package, configName.split(".")[0])

# Fetches the HLT key from ConfDB - turned off in options for now
def GetHltKeyForRun(run):
   raise SystemExit("Not implemented yet")

# Formats an HLT CFF path into a suitable python include statement
def FormatHltCff(cffPath):
   pathParts = cffPath.split(".")
   if not re.match("^[_A-Za-z0-9]*\.[_A-Za-z0-9]*\.[_A-Za-z0-9]*$", cffPath):
      raise SystemExit("Expected cff in form Package.Module.configName_cff")
   return 'process.load("%s")' % cffPath

# Returns python code to compile an HLT configuration
def GetHltCompileCode(subsystem, package, hltConfig):
   tmpCode = compileCode.replace("SUBSYSTEM", subsystem)
   tmpCode = tmpCode.replace("PACKAGE", package)
   tmpCode = tmpCode.replace("CONFIG", hltConfig + "c")
   return tmpCode

# Prepares the analysis configuration
def CreateAnalysisConfig(analysis, hltInclude):
   anaName = "JobAnalysisConfig_%s_cfg.py" % jobHash
   f = open(analysis)
   g = open(anaName, "w")
   g.write(f.read())
   g.write("\n")
   g.write(hltInclude)
   g.close()
   f.close()
   return anaName

# Quick sanity check of the environment
CheckEnvironment()

# Get the required HLT config snippet
hltConfig = None
if options.hltkey != None:
   hltConfig = GetHltConfiguration(options.hltkey, options.frontier)
elif options.hltcff != None:
   hltConfig = FormatHltCff(options.hltcff)
else:
   hltKey = GetHltKeyForRun(0)
   hltConfig = GetHltConfiguration(hltKey)

# Prepare the analysis configuration
anaConfig = CreateAnalysisConfig(options.analysis, hltConfig)

if options.hltcff:
    print "Using HLT configuration:           %s" % hltConfig
else:
    print "Created HLT configuration:         %s" % hltConfig
print "Created analysis configuration:    %s" % anaConfig

