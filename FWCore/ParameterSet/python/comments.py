#!/bin/env python

# tool to add comments to python configuration files
# 
# this tool makes the following assumptions
#  - for each file Subproject/Package/data/config.[cff,cfg,cfi] 
#    there is a file Subproject/Package/python/config_[cff,cfg,cfi].py
#  - comments are either in the line just above a command or just behind in the same line


# should hold the comment ID
# which is Comment.value and Comment.type = "trailing" or "inline"
# 

from FWCore.ParameterSet import cfgName2py

class Comment:
    pass


def prepareReplaceDict(line, comment, replaceDict):
    """take a line and a corresponding comment and prepare the replaceDict such that it can be found again in the python version """
    allKeywords = ['module','int32','vint32','uint32', 'vuint32','double','vdouble','InputTag', 'VInputTag', 'PSet', 'VPSet', 'string', 'vstring', 'bool', 'vbool', 'path', 'sequence', 'schedule', 'endpath', 'es_source', 'es_module', 'block', 'FileInPath']
    unnamedKeywords = ['es_source', 'es_module']


    words = line.lstrip().split()
    # at least <keyword> <label> = "
    if len(words) > 1:
        firstWord = words[0]
        if firstWord in allKeywords and len(words) > 2 and words[2] == '=':
           tokenInPython = words[1] + " = "
           replaceDict[tokenInPython] = comment
        elif firstWord == 'untracked' and len(words) > 3 and words[1] in allKeywords and words[3] == '=':
           tokenInPython = words[2] + " = cms.untracked"
           replaceDict[tokenInPython] = comment
        elif firstWord.startswith('include'): # handling of include statements
           pythonModule = cfgName2py.cfgName2py(line.split('"')[1])
           pythonModule = pythonModule.replace("/",".").replace("python.","").replace(".py","")
           tokenInPython = "from "+pythonModule
           tokenInPython = tokenInPython.replace("/",".").replace("python.","").replace(".py","")
           replaceDict[tokenInPython] = comment
           # if a cfg
           tokenInPython = "process.load(\""+pythonModule
           replaceDict[tokenInPython] = comment
        elif firstWord == 'source' and len(words) > 1 and words[1] == '=':
           replaceDict['source = '] = comment
        elif firstWord in unnamedKeywords and len(words) > 2 and words[1] == '=':
           tokenInPython = words[2] + ' = cms.ES'
           replaceDict[tokenInPython] = comment
        elif firstWord == 'replace' and len(words) > 2 and words[2] == '=':
           tokenInPython= words[1] + " = "
           replaceDict[tokenInPython] = comment
        elif firstWord == 'replace' and len(words) > 2 and words[2] == '=':
           tokenInPython= words[1] + " = "
           replaceDict[tokenInPython] = comment
           # if it's a cfg
           tokenInPython = 'process.'+tokenInPython
           replaceDict[tokenInPython] = comment
        elif firstWord == 'using' and len(words) == 2:
           tokenInPython= words[1]
           replaceDict[tokenInPython] = comment
       # if it's a significant line, we're not in a comment any more
        else:
          replaceDict["@beginning"] +="\n"+comment.value
    


def identifyComments(configString):
    
    replaceDict = {}

    replaceDict["@beginning"] = "# The following comments couldn't be translated into the new config version:\n"
    allKeywords = ['module','int32','vint32','uint32', 'vuint32','double','vdouble','InputTag', 'VInputTag', 'PSet', 'VPSet', 'string', 'bool', 'vbool', 'path', 'sequence', 'schedule', 'endpath', 'es_source', 'es_module', 'block', 'FileInPath']
    unnamedKeywords = ['es_source', 'es_module']
    # the ugly parsing part
    inComment = False # are we currently in a comment?
    inSlashStarComment = False

    # TODO - include inline comments
    # for now only modules work
    for line in configString.splitlines():
        if line.lstrip().startswith("/*"):
            comment = Comment()
            comment.type = "slashstar"
            splitStarSlash = line.lstrip().lstrip("/*").split("*/")
            comment.value = "# "+splitStarSlash[0]+"\n"
            # has it ended yet?  Might ignore legitimate commands after the end.
            inSlashStarComment = (line.find('*/') == -1)
            inComment = True
        elif inSlashStarComment:
            splitStarSlash = line.lstrip().lstrip("/*").split("*/")
            comment.value += "# "+splitStarSlash[0]+"\n"
            inSlashStarComment = (line.find('*/') == -1)
        elif line.lstrip().startswith("#") or line.lstrip().startswith("//"):
          if inComment:
            comment.value += "#"+line.lstrip().lstrip("//").lstrip("#") + "\n"
          else:
            comment = Comment()
            comment.type = "trailing"
            comment.value = "#"+line.lstrip().lstrip("//").lstrip("#") + "\n"
            inComment = True
        elif inComment:  # we are now in a line just below a comment
          words = line.lstrip().split()
          # at least <keyword> <label> = "
          if len(words) > 1:
             prepareReplaceDict(line,comment,replaceDict)             
          if len(words) > 0:
             inComment = False
        else:
             # now to comments in the same line
             if len(line.split("#")) > 1 or len(line.split("//")) > 1:
               comment = Comment()
               if len(line.split("#")) > 1:
                 comment.value = '#'+line.split("#")[1]
               else:
                 comment.value = '#'+line.split("//")[1]
               comment.type = "inline"
               # prepare the replaceDict
               prepareReplaceDict(line, comment, replaceDict)
      

    return replaceDict
          
        


def modifyPythonVersion(configString, replaceDict):

    # first put all comments at the beginning of the file which could not be assigned to any other token
    if replaceDict["@beginning"] != "# The following comments couldn't be translated into the new config version:\n":
      configString = replaceDict["@beginning"]+"\n"+configString 

    replaceDict.pop("@beginning")

    # identify the lines to replace and prepare the replacing line
    actualReplacements = {}
    for keyword, comment in replaceDict.iteritems():
        for line in configString.splitlines():
            if line.lstrip().startswith(keyword):
                indentation = line[0:line.find(keyword)]
                if len([1 for l in configString.splitlines() if l.lstrip().startswith(keyword)]) !=1:
                    print "WARNING. Following keyword not unique:", keyword
                    continue 
                if comment.type == "inline":
                  newLine = line + " #"+comment.value+"\n"
                else:
                  newLine = line.replace(line,comment.value+line)  # lacking the trailing whitespace support
                  newLine = newLine.replace('#', indentation+'#')
                actualReplacements[line] = newLine


    # now do the actual replacement
    for original, new in actualReplacements.iteritems():        
        configString = configString.replace(original, new)

    return configString


def loopfile(cfgFileName):
   cfgFile = file(cfgFileName)
   cfgString = cfgFile.read()
   
   pyFileName = cfgName2py.cfgName2py(cfgFileName)

   try: 
     pyFile = file(pyFileName)
     pyString = pyFile.read()
   except IOError:
     print pyFileName, "does not exist"
     return
   comments = identifyComments(cfgString)
   print "Opening", pyFileName
   newPyString = modifyPythonVersion(pyString, comments)
   pyFile.close()

   pyOutFile = file(pyFileName,"w")
   pyOutFile.write(newPyString)
   print "Wrote", pyFileName
   pyOutFile.close()
 
#out = file("foo_cfi.py","w")
#out.write(configString)
#print configString



# 
import os
from sys import argv

if len(argv) != 2:
    print "Please give either a filename, or 'local'"
elif argv[1] == 'local':
  for subsystem in os.listdir("."):
    try:
      #print subsystem
       for package in os.listdir("./"+subsystem):
         #print "  ",subsystem+"/"+package
          try:
            for name in os.listdir(subsystem+"/"+package+"/data"):
              if len(name.split("."))==2 and name.split(".")[1] in ["cfg","cfi","cff"]:
                print subsystem+"/"+package+"/data/"+name
                loopfile(subsystem+"/"+package+"/data/"+name)
          except OSError:
            pass
    except OSError:
      pass
    

else:
  loopfile(argv[1])


