#REMOVE-THIS-SAFEGUARD## #!/usr/bin/env python

import distutils
import distutils.fancy_getopt
import os
import sys
import string
import re
import glob
import commands


#---Settings----#000000#FFFFFF--------------------------------------------------

class Color:
  "ANSI escape display sequences"
  info          = "\033[1;34m"
  hilight       = "\033[31m"
  alternate     = "\033[32m"
  extra         = "\033[33m"
  backlight     = "\033[43m"
  underline     = "\033[4m"
  lessemphasis  = "\033[30m"
  deemphasis    = "\033[1;30m"
  none          = "\033[0m"

class Error:
  "Exit codes"
  none        = 0
  usage       = 1
  argument    = 2
  execution   = 3

class Settings:
  "Settings for execution etc."

  blurb         = """
Usage: %program% [options] <config-file>

Dumps all the included configuration files (and fragments) using a deep search
starting from the given top-level file.

Command-line options can be:
"""
  variables     = distutils.fancy_getopt.FancyGetopt([
                    ("help",                  "h",  "Print this informational message."),
                    ("search-paths",          "s",  os.path.pathsep + " separated list of paths to try and render all those relative include paths from -- they will be tried in the given order until a match is found. Default %search-paths% (i.e. current directory plus your CMSSW_SEARCH_PATH if defined)."),
                    ("dump-contents",         "c",  "Dump also the contents of each file, not just the list of include files. Default %dump-contents%."),
                    ("search-mask",           "s",  "Semi-colon separated list of glob expressions for include files to follow. This controls the search and therefore also the display (if you don't find it, you can't show it). Default %searched-extensions%."),
                    ("dump-mask",             "d",  "Semi-colon separated list of glob expressions for files to display. This only controls display and not the search. Default %displayed-extensions%."),
                    ("veto-dump-mask",        "v",  "Semi-colon separated list of glob expressions for files to NOT dump the contents of, if relevant (if you're not dumping anything...). Note that these are just the relative to CMSSW/src paths, as they appear in the config files. Default %veto-dump-mask%."),
                    ("min-dump-depth",        "M",  "Minimum depth of tree to dump contents of, if relevant (they will still be traversed in order to get to the lower echelons, of course). Default %min-dump-depth%."),
                    ("max-depth",             "m",  "Maximum depth of tree to traverse. Default %max-depth%, use negative to get 'em all."),
                    ("no-color",              "n",  "Print without color, but why would you want to do that? Default %no-color%. WARNING: not yet implemented, so there."),
                  ])
  options       = { "help"                    : False,
                    "search-paths"            : "." + os.path.pathsep + os.environ.get("CMSSW_SEARCH_PATH", ""),
                    "dump-contents"           : False,
                    "search-mask"             : "*_cfi.py;*_cff.py;*_cfg.py",
                    "dump-mask"               : "*",
                    "veto-dump-mask"          : "",
                    "min-dump-depth"          : 0,
                    "max-depth"               : -1,
                    "no-color"                : False,
                  }
  usage         = None
  
#  
  matchIncludes = [ re.compile("""^\s*include\s+(['"])(.+)\\1""", re.IGNORECASE | re.MULTILINE)
#                  , re.compile("""^\s*module\s+[^=]+=.*\s+from\s*(['"])(.+)\\1""", re.IGNORECASE | re.MULTILINE)
                   , re.compile("""process\.load\(['"](.+)\\1""", re.IGNORECASE | re.MULTILINE)
                  ]
  matchImports =  [ re.compile("""^\s*import\s+(.+)""", re.IGNORECASE | re.MULTILINE)
                   , re.compile("""^\s*from\s+(.+)+\s+import\s""", re.IGNORECASE | re.MULTILINE)
                   , re.compile("""process\.load\(['"](.+)['"]\)""", re.IGNORECASE | re.MULTILINE)
                  ]




#---Functions---#000000#FFFFFF--------------------------------------------------

def convertSetting(setting, convertor, errorMessage):
  """
  Convert the given setting using the form produced by convertor. ValueError's
  are caught and errorMessage is printed, then the application exits. errorMessage
  is a format string taking one string argument, the problematic value.
  """

  try:
    Settings.options[setting] = convertor(Settings.options[setting])
  except ValueError:
    print Color.hilight + (errorMessage % str(Settings.options[setting])) + Color.none
    sys.exit(Error.argument)

  return

def toList(streeng, delimiter = ";"):
  streeng = streeng.strip()
  if streeng == "":   return []
  return string.split(streeng, delimiter)

def fileName(pyModule):
  result = ''
  l = pyModule.split('.')
  if len(l) == 3:
    result = l[0]+'/'+l[1]+'/python/'+l[2]+'.py'
  return result

def isRelevant(filePath, patterns):
  """
  Yes if matched and not vetoed, no otherwise.
  """
  
  for match in patterns:
    if glob.fnmatch.fnmatch(filePath, match):
      return True
  return False

def goGetIt(relativePath):
  """
  Find the first instance of relativePath in the search-paths list.
  If not found, returns None.
  """
  if os.path.isabs(relativePath): return relativePath
  for basePath in Settings.options["search-paths"]:
    fullPath    = os.path.join(basePath, relativePath)
    if os.path.isfile(fullPath):
      return fullPath
  return None



#---Main Execution Point---#D50000#FFFF80---------------------------------------
class TreeCrawler:
  def __init__(self, doer):
    self.__levels = 0
    self.__alreadyDone = []
    self.__doer = doer
  def parseArgs(self,args):
    "Parses command-line arguments into options and variables"

    Settings.scriptDir  = os.path.dirname(os.path.abspath(args[0]))
    Settings.blurb      = Settings.blurb.replace("%program%", os.path.basename(args[0]))
    Settings.usage      = string.join(Settings.variables.generate_help(Settings.blurb), "\n")
    for (variable, default) in Settings.options.iteritems():
      if type(default) == type(""):
        Settings.options[variable]  = default.replace("%script-dir%", Settings.scriptDir)
    for (variable, default) in Settings.options.iteritems():
      Settings.usage    = re.compile("%" + variable.replace("-", "-\s*") + "%") \
                            .sub(str(default), Settings.usage)

    try:
      (arguments, options)  = Settings.variables.getopt(args[1:])
    except distutils.errors.DistutilsArgError, details:
      print "Error in command-line:", details
      print Settings.usage
      sys.exit(Error.usage)

    # Get the dictionary of option -> value
    Settings.options.update(dict(Settings.variables.get_option_order()))

    # Special options
    convertSetting("search-mask",   toList,   "%s must be a semi-colon separated list.")
    convertSetting("dump-mask",   toList,   "%s must be a semi-colon separated list.")
    convertSetting("veto-dump-mask",toList,   "%s must be a semi-colon separated list.")
    convertSetting("search-paths",  lambda x: toList(x, os.path.pathsep), "%s must be a " + os.path.pathsep + " separated list.")

    #if Settings.options["help"] or len(arguments) != 1:
    #  print Settings.usage
    #  print
    #  sys.exit(Error.none)
 
    return arguments

  def crawlInto(self, filePath):
    maxDepth      = Settings.options["max-depth"]

    if (maxDepth >= 0 and self.__levels > maxDepth):  return
    fullPath  = goGetIt(filePath)
    self.__doer.do(filePath, self.__levels)
    
    if fullPath in self.__alreadyDone: return
    aFile     = open(fullPath)
    content   = aFile.read()
    aFile.close()

    # Content 
    if self.__levels >= Settings.options["min-dump-depth"]                \
      and isRelevant(filePath, Settings.options["dump-mask"])         \
      and not isRelevant(filePath, Settings.options["veto-dump-mask"]):
      self.__doer.doContent(content)

    self.__alreadyDone.append(fullPath)

    # Search for includes
    includes  = []
    for matchImports in Settings.matchImports:
      includes.extend(matchImports.findall(content))

    for include in includes:
      anotherFile = fileName(include)
      if isRelevant(anotherFile, Settings.options["search-mask"]):
        self.__levels += 1
        self.crawlInto(anotherFile)
        self.__levels -= 1


  
if __name__ == '__main__':
  pass
