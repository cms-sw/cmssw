#! /usr/bin/env python3

from __future__ import print_function
import optparse
import os
import sys
import re
import shutil

ccRE = re.compile (r'(\w+)\.cc')

def extractBuildFilePiece (buildfile, copy, target = 'dummy'):
    """Extracts necessary piece of the buildfile.  Returns empty
    string if not found."""    
    try:
        build = open (buildfile, 'r')
    except:        
        raise RuntimeError("Could not open BuildFile '%s' for reading. Aboring." \
              % buildfile)
    # make my regex
    startBinRE = re.compile (r'<\s*bin\s+name=(\S+)')
    endBinRE   = re.compile (r'<\s*/bin>')
    exeNameRE  = re.compile (r'(\w+)\.exe')
    ccName = os.path.basename (copy)
    match = ccRE.match (ccName)
    if match:
        ccName = match.group (1)
    retval = ''
    foundBin = False
    for line in build:
        # Are we in the middle of copying what we need?
        if foundBin:
            retval += line
        # Are we copying what we need and reach the end?
        if foundBin and endBinRE.search (line):
            # all done
            break
        # Is this the start of a bin line with the right name?
        match = startBinRE.search (line)
        if match:
            # strip of .exe if it's there
            exeName = match.group (1)
            exeMatch = exeNameRE.search (exeName)
            if exeMatch:
                exeName = exeMatch.group (1)
            if exeName == ccName:
                foundBin = True
                line = re.sub (exeName, target, line)
                retval = line
    build.close()
    return retval


def createBuildFile (buildfile):
    """Creates a BuildFile if one doesn't already exist"""
    if os.path.exists (buildfile):
        return
    build = open (buildfile, 'w')
    build.write ("<!-- -*- XML -*- -->\n\n<environment>\n\n</environment>\n")
    build.close()


def addBuildPiece (targetBuild, buildPiece):
    """Adds needed piece for new executable.  Returns true upon
    success."""
    backup = targetBuild + ".bak"
    shutil.copyfile (targetBuild, backup)
    oldBuild = open (backup, 'r')
    newBuild = open (targetBuild, 'w')
    environRE = re.compile (r'<\s*environment')
    success = False
    for line in oldBuild:
        newBuild.write (line)
        if environRE.search (line):
            newBuild.write ('\n\n')
            newBuild.write (buildPiece)
            success = True
    oldBuild.close()
    newBuild.close()
    return success
        


########################
## ################## ##
## ## Main Program ## ##
## ################## ##
########################

if __name__ == '__main__':
    # We need CMSSW to already be setup
    base         = os.environ.get ('CMSSW_BASE')
    release_base = os.environ.get ('CMSSW_RELEASE_BASE')
    if not base or not release_base:
        print("Error: You must have already setup a CMSSW release.  Aborting.")
        sys.exit()
    # setup the options
    parser = optparse.OptionParser('usage: %prog [options] '
                                   'Package/SubPackage/name\n'
								   'Creates new analysis code')
    parser.add_option ('--copy', dest='copy', type='string', default = 'blank',
                       help='Copies example. COPY should either be a file'
                       ' _or_ an example in PhysicsTools/FWLite/examples')
    parser.add_option ('--newPackage', dest='newPackage', action='store_true',
                       help='Will create Package/Subpackage folders if necessary')
    parser.add_option ('--toTest', dest='toTest', action='store_true',
                       help='Will create files in test/ instead of bin/')
    (options, args) = parser.parse_args()
    # get the name of the copy file and make sure we can find everything
    copy = options.copy
    if not re.search ('\.cc$', copy):
        copy += '.cc'
    buildCopy = os.path.dirname (copy)
    if len (buildCopy):
        buildCopy += '/BuildFile'
    else:
        buildCopy = 'BuildFile'
    found = False
    searchList = ['',
                  base + "/src/PhysicsTools/FWLite/examples/",
                  release_base+ "/src/PhysicsTools/FWLite/examples/"]
    fullName  = ''
    fullBuild = ''
    for where in searchList:        
        name  = where + copy
        build = where + buildCopy
        # Is the copy file here
        if os.path.exists (name):
            # Yes.
            found = True
            # Is there a Buildfile too?
            if not os.path.exists (build):
                print("Error: Found '%s', but no accompying " % name, \
                      "Buildfile '%s'. Aborting" % build)
                sys.exit()
            fullName = name
            fullBuild = build
            break
    if not found:
        print("Error: Did not find '%s' to copy.  Aborting." % copy)
        sys.exit()
    if len (args) != 1:
        parser.print_usage()
        sys.exit()
    pieces = args[0].split('/')
    if len (pieces) != 3:
        print("Error: Need to provide 'Package/SubPackage/name'")
        sys.exit()    
    target = pieces[2]
    match = ccRE.match (target)
    if match:
        target = match.group (1)    
    buildPiece = extractBuildFilePiece (fullBuild, copy, target)
    if not buildPiece:
        print("Error: Could not extract necessary info from Buildfile. Aborting.")
        sys.exit()
    # print buildPiece
    firstDir  = base + '/src/' + pieces[0]
    secondDir = firstDir + '/' + pieces[1]
    if options.toTest:
        bin = '/test'
    else:
        bin = '/bin'
    if not os.path.exists (secondDir) and \
       not options.newPackage:
        raise RuntimeError("%s does not exist.  Use '--newPackage' to create" \
              % ("/".join (pieces[0:2]) ))
    dirList = [firstDir, secondDir, secondDir + bin]
    targetCC = dirList[2] + '/' + target + '.cc'
    targetBuild = dirList[2] + '/BuildFile'
    if os.path.exists (targetCC):
        print('Error: "%s" already exists.  Aborting.' % targetCC)
        sys.exit()
    # Start making directory structure
    for currDir in dirList:
        if not os.path.exists (currDir):
            os.mkdir (currDir)
    # copy the file over
    shutil.copyfile (fullName, targetCC)
    print("Copied:\n   %s\nto:\n   %s.\n" % (fullName, targetCC))
    createBuildFile (targetBuild)
    if extractBuildFilePiece (targetBuild, target):
        print("Buildfile already has '%s'.  Skipping" % target)
    else :
        # we don't already have a piece here
        if addBuildPiece (targetBuild, buildPiece):
            print("Added info to:\n   %s." % targetBuild)
        else:
            print("Unable to modify Buildfile.  Sorry.")
