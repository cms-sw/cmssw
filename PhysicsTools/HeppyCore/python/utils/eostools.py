#!/usr/bin/env python
"""
A module to manipulate files on EOS or on the local file system. Intended to have the same interface as castortools.py.
"""
import sys
import os
import re
import pprint
import shutil

def setCAFPath():
    """Hack to get the CAF scripts on the PYTHONPATH"""
    caf = '/afs/cern.ch/cms/caf/python'

    if caf not in sys.path:
        sys.path.append(caf)
setCAFPath()
import cmsIO

def runXRDCommand(path, cmd, *args):
    """Run an xrd command.

    !!! Will, what is happening in case of problem?
    ??? At some point, should return a list of lines instead of a string."""
    
    lfn = eosToLFN(path)
    #print "lfn:", lfn, cmd
    tokens = cmsIO.splitPFN(lfnToPFN(lfn))
    
    command = ['xrd', tokens[1], cmd, tokens[2]]
    command.extend(args)
    runner = cmsIO.cmsFileManip()
    # print ' '.join(command)
    return runner.runCommand(command)

def runEOSCommand(path, cmd, *args):
    """Run an eos command.

    !!! Will, when the EOS command fails, it passes silently...
    I think we should really try and raise an exception in case of problems.
    should be possible as the return code is provided in the tuple returned by runner."""
    
    lfn = eosToLFN(path)
    pfn = lfnToPFN(lfn)
    tokens = cmsIO.splitPFN(pfn)
    
    #obviously, this is not nice
    command = ['/afs/cern.ch/project/eos/installation/pro/bin/eos.select', cmd]
    command.extend(args)
    command.append(tokens[2])
    runner = cmsIO.cmsFileManip()
    return runner.runCommand(command)

def isLFN( path ):
    """Tests whether this path is a CMS LFN (name starts with /store...)"""
    # return re.match('^/store.*', path ) is not None
    return path.startswith('/store')

def isEOS( path ):
    """Tests whether this path is a CMS EOS (name starts with /eos...)"""
    return path.startswith('/eos') or path.startswith('root://eoscms.cern.ch//eos/cms')

def eosToLFN( path ):
    """Converts a EOS PFN to an LFN.

    Just strip out /eos/cms from path.
    If this string is not found, return path.
    ??? Shouldn't we raise an exception instead?"""
    return path.replace('root://eoscms.cern.ch/', '').replace('/eos/cms','')

#also define an alias for backwards compatibility
castorToLFN = eosToLFN

def lfnToPFN( path, tfcProt = 'rfio'):
    """Converts an LFN to a PFN. For example:
    /store/cmst3/user/cbern/CMG/TauPlusX/Run2011A-03Oct2011-v1/AOD/V2/PAT_CMG_V2_4_0/H2TAUTAU_Nov21
    ->
    root://eoscms//eos/cms/store/cmst3/user/cbern/CMG/TauPlusX/Run2011A-03Oct2011-v1/AOD/V2/PAT_CMG_V2_4_0/H2TAUTAU_Nov21?svcClass=cmst3&stageHost=castorcms

    This function only checks path, and does not access the storage system.
    If the path is in /store/cmst3, it assumes that the CMST3 svcClass is to be used.
    Otherwise, is uses the default one. 
    
    ??? what is tfcprot? """

    if path.startswith("/store/"):
        path = path.replace("/store/","root://eoscms.cern.ch//eos/cms/store/")
    if path.startswith("/pnfs/psi.ch/cms/trivcat/"):
        path = path.replace("/pnfs/psi.ch/cms/trivcat/","root://t3se01.psi.ch//")
    #print "path to cmsFile():", path
    entity = cmsIO.cmsFile( path, tfcProt )
#    tokens = cmsIO.splitPFN(entity.pfn)
    pfn = '%s://%s//%s/' % (entity.protocol,entity.host,entity.path)
    
    pfn = entity.pfn
    if tfcProt == 'rfio' and \
        entity.path.startswith("/eos/cms/") and \
                str(entity.stat()).startswith("Error 3011: Unable to stat"):

            pfn.replace("/eos/cms","/castor/cern.ch/cms")
            pfn.replace("eoscms","castorcms")
    return pfn


def lfnToEOS( path ):
    """Converts LFN to EOS.

    If path is not an LFN in the first place, return path.
    ??? shouldn't we raise an exception?"""
    if isLFN(path):
        pfn = 'root://eoscms.cern.ch//eos/cms/' + path
        return pfn.replace('//store','/store') 
    else:
        return path

#also define an alias for backwards compatibility
lfnToCastor = lfnToEOS

def isEOSDir( path ):
    """Returns True if path is either:
    /store/...
    or
    /eos/cms/store/...
    or
    root://eoscms.cern.ch//eos/cms/

    Otherwise, returns False.

    WARNING!! This function does not check for path existence,
    and returns true also for plain files.
    !!! Will, is my summary correct? 
    """
    if os.path.exists( path ):
        # path does not exist
        # COLIN: I think this condition could be removed,
        # as it duplicates the following one. 
        return False
    if not path.startswith('/eos') and not path.startswith('/store') and not path.startswith('root://eoscms.cern.ch//eos/cms/'):
        # neither an EOS PFN or a LFN.
        return False
    # at this stage, we must have an EOS PFN or an LFN
    pfn = lfnToPFN(eosToLFN(path))
    tokens = cmsIO.splitPFN(pfn)
    return tokens and tokens[1].lower().startswith('eos')

#also define an alias for backwards compatibility
isCastorDir = isEOSDir


def isEOSFile( path, tfcProt = 'rfio'):
    """Returns True if path is a file or directory stored on EOS (checks for path existence)
    ??? This function does not behave well if passed a non EOS path...
    returns lots of error messages like:
>>> eostools.isEOSFile('/store/asdfasfd')
Command (['ls', '/', 's', 't', 'o', 'r', 'e', '/', 'a', 's', 'd', 'f', 'a', 's', 'f', 'd', '/store']) failed with return code: 2
ls: s: No such file or directory
ls: t: No such file or directory
ls: o: No such file or directory
ls: r: No such file or directory
ls: e: No such file or directory
ls: a: No such file or directory
ls: s: No such file or directory
ls: d: No such file or directory
ls: f: No such file or directory
ls: a: No such file or directory
ls: s: No such file or directory
ls: f: No such file or directory
ls: d: No such file or directory
ls: /store: No such file or directory

ls: s: No such file or directory
ls: t: No such file or directory
ls: o: No such file or directory
ls: r: No such file or directory
ls: e: No such file or directory
ls: a: No such file or directory
ls: s: No such file or directory
ls: d: No such file or directory
ls: f: No such file or directory
ls: a: No such file or directory
ls: s: No such file or directory
ls: f: No such file or directory
ls: d: No such file or directory
ls: /store: No such file or directory

False
    """
    _, _, ret = runEOSCommand( path, 'ls')
    return ret == 0

#also define an alias for backwards compatibility
isCastorFile = isEOSFile


def fileExists( path ):
    """Returns true if path is a file or directory stored locally, or on EOS.

    This function checks for the file or directory existence."""

    eos = isEOSDir(path)
    result = False
    if eos:
        # print 'eos', path
        result = isEOSFile(path)
    else:
        # print 'not eos', path
        #check locally
        result = os.path.exists(path)
    # print result
    return result


def eosDirSize(path):
    '''Returns the size of a directory on EOS in GB.'''
    lfn = eosToLFN(path)
    res = runEOSCommand(lfn, 'find', '--size')
    output = res[0].split('\n')
    size = 0
    for file in output:
        try:
            size += float(file.split('=')[2])
        except IndexError:
            pass
    return size/1024/1024/1024


def createEOSDir( path ):
    """Makes a directory in EOS

    ???Will, I'm quite worried by the fact that if this path already exists, and is
    a file, everything will 'work'. But then we have a file, and not a directory,
    while we expect a dir..."""
    lfn = eosToLFN(path)
    if not isEOSFile(lfn):
    # if not isDirectory(lfn):
        runEOSCommand(lfn,'mkdir','-p')
        #        entity = cmsIO.cmsFile( lfn,"stageout")
        #        entity.mkdir([])
        #        # print 'created ', path
    if isDirectory(path):
        return path
    else:
        raise OSError('cannot create directory '+ path)

#also define an alias for backwards compatibility
createCastorDir = createEOSDir

def mkdir(path):
    """Create a directory, either on EOS or locally"""
    # print 'mkdir', path
    if isEOS( path ) or isLFN(path):
        createEOSDir(path)
    else:
        # recursive directory creation (like mkdir -p)
        os.makedirs(path)
    return path

def isDirectory(path):
    """Returns True if path is a directory on EOS.

    Tests for file existence. 
    This function returns False for EOS files, and crashes with local paths

    ???Will, this function also seems to work for paths like:
    /eos/cms/...
    ??? I think that it should work also for local files, see isFile."""

    out, _, _ = runXRDCommand(path,'existdir')
    return 'The directory exists' in out

def isFile(path):
    """Returns True if a path is a file.

    Tests for file existence.
    Returns False for directories.
    Works on EOS and local paths.
    
    ???This function works with local files, so not the same as isDirectory...
    isFile and isDirectory should behave the same.
    """

    if not path.startswith('/eos') and not path.startswith('/store'):
        if( os.path.isfile(path) ):
            return True
        else:
            return False
    else: 
        out, _, _ = runXRDCommand(path,'existfile')
        return 'The file exists' in out

def chmod(path, mode):
    """Does chmod on a file or directory"""
    #
    return runEOSCommand(path, 'chmod', '-r', str(mode))


def listFiles(path, rec = False, full_info = False):
    """Provides a list of the specified directory
    """
    # -- listing on the local filesystem --
    if os.path.isdir( path ):
        if not rec:
            # not recursive
            return [ '/'.join([path,file]) for file in os.listdir( path )]
        else:
            # recursive, directories are put in the list first,
            # followed by the list of all files in the directory tree
            result = []
            allFiles = []
            for root,dirs,files in os.walk(path):
                result.extend( [ '/'.join([root,dir]) for dir in dirs] )
                allFiles.extend( [ '/'.join([root,file]) for file in files] )
            result.extend(allFiles)
            return result
    # -- listing on EOS --
    cmd = 'dirlist'
    if rec:
        cmd = 'dirlistrec'
    files, _, _ = runXRDCommand(path, cmd)
    result = []
    for line in files.split('\n'):
        tokens = [t for t in line.split() if t]
        if tokens:
            #convert to an LFN
            # result.append(tuple(tokens))
            #COLIN need same interface for eos and local fs
            if full_info:
                result.append( tokens)
            else:
                result.append( tokens[4] )
    return result

def which(cmd):
    command = ['which', cmd]
    runner = cmsIO.cmsFileManip()
    out, _, _ = runner.runCommand(command)
    
    lines = [line for line in out.split('\n') if line]
    if len(lines) == 1:
        return lines[0]
    elif len(lines) == 2:
        return lines[1]
    else:
        return lines

def ls(path, rec = False):
    """Provides a simple list of the specified directory, works on EOS and locally"""
    return [eosToLFN(t) for t in listFiles(path, rec)]

def ls_EOS(path, rec = False):
    """Provides a simple list of the specified directory, works on EOS only, but is faster than the xrd version"""
    if rec:
        stdout, _, ret = runEOSCommand(path,'find','-f')
        return [eosToLFN(line) for line in stdout.split('\n') if line]
    else:
        stdout, _, ret = runEOSCommand(path,'ls')
        lfn = eosToLFN(path)
        return [os.path.join(lfn,line) for line in stdout.split('\n') if line]

def rm(path, rec=False):
    """rm, works on EOS and locally.

    Colin: should implement a -f mode and a confirmation when deleting dirs recursively."""
    # print 'rm ', path
    path = lfnToEOS(path)
    if isEOS(path):
        if rec:
            runEOSCommand(path, 'rm', '-r')
        else: 
            runEOSCommand(path,'rm')
    elif os.path.exists(path):
        if not rec:
            os.remove( path )
        else:
            shutil.rmtree(path)
    else:
        raise ValueError(path + ' is not EOS and not local... should not happen!')

def remove( files, rec = False):
    """Remove a list of files and directories, possibly recursively

    Colin: Is that obsolete? why not use rm?"""
    for path in files:
        lfn = eosToLFN(path)
        if not rec:
            rm(path)
        else:
            #this should be used with care
            file_list = ls(path, rec = True)
            file_list.append(lfn)
            
            #order the files in depth order - i.e. remove the deepest files first
            files_rec = sorted([(len([ff for ff in f.split('/') if ff]), f) for f in file_list if f and f.startswith(lfn)], reverse = True)
            
            for f in files_rec:
                rm(f[1])
                
def cat(path):
    """cat, works on EOS and locally"""
    path = lfnToEOS(path)
    if isEOS(path):
        #print "the file to cat is:", path
        out, err, _ = runXRDCommand(path,'cat') 
        lines = []
        if out:
            pattern = re.compile('cat returned [0-9]+')
            for line in out.split('\n'):
                match = pattern.search(line)
                if line and match is not None:
                    lines.append(line.replace(match.group(0),''))
                    break
                else:
                    lines.append(line)
        if err:
            print >> sys.stderr, out
            print >> sys.stderr, err
        allLines = '\n'.join(lines)
        if allLines and not allLines.endswith('\n'):
            allLines += '\n'
        return allLines
    else:
        content = file(path).read()
        if content and not content.endswith('\n'):
            content += '\n'
        return content
    
def xrdcp(src, dest):
    """Does a copy of files using xrd.

    Colin: implement a generic cp interface as done for rm, ls, etc?"""
    
    recursive = False
    
    #first the src file
    pfn_src = src
    if os.path.exists(src):
        #local
        pfn_src = src
        if os.path.isdir(src):
            recursive = True
    elif fileExists(src):
        src = eosToLFN(src)
        pfn_src = lfnToPFN(src)
        if isDirectory(src):
            recursive = True
    else:
        raise ValueError(src + ' does not exist.')
            
    #now the dest
    pfn_dest = dest
    if isEOSDir(dest):
        dest = eosToLFN(dest)
        pfn_dest = lfnToPFN(dest)
        if isDirectory(dest):
            tokens = cmsIO.splitPFN(pfn_dest)
            pfn_dest = '%s://%s//%s/' % (tokens[0],tokens[1],tokens[2])
    elif os.path.exists(dest):
        pfn_dest = dest

    command = ['xrdcp']
    if recursive:
        # print 'recursive'
        topDir = src.rstrip('/').split('/')[-1]
        if topDir != '.':
            dest = '/'.join([dest, topDir])
            # print 'mkdir ' + dest
            mkdir( dest )
        files = listFiles(src, rec=True)
        # pprint.pprint( [file[4] for file in files] )
        for srcFile in files:
            # srcFile = file[4]
            pfnSrcFile = srcFile
            if isEOSDir(srcFile):
                srcFile = eosToLFN(srcFile)
                pfnSrcFile = lfnToPFN(srcFile)
            destFile = srcFile.replace( src, '' )
            destFile = '/'.join([dest,destFile])
            pfnDestFile = destFile
            if isEOSDir(destFile):
                lfnDestFile = eosToLFN(destFile)
                pfnDestFile = lfnToPFN(lfnDestFile)
            # print 'srcFile', pfnSrcFile
            # print 'destFile', pfnDestFile
            if isFile(srcFile):
                _xrdcpSingleFile(  pfnSrcFile, pfnDestFile )
            else:
                mkdir(destFile)
    else:
        _xrdcpSingleFile( pfn_src, pfn_dest )


def _xrdcpSingleFile( pfn_src, pfn_dest):
    """Copies a single file using xrd."""
    
    command = ['xrdcp']
    command.append(pfn_src)
    command.append(pfn_dest)
    # print ' '.join(command)
    run = True
    if run: 
        runner = cmsIO.cmsFileManip()
        out, err, ret = runner.runCommand(command)
        if err:
            print >> sys.stderr, out
            print >> sys.stderr, err
        return ret

def move(src, dest):
    """Move filename1 to filename2 locally to the same server"""
    
    src = eosToLFN(src)
    dest = eosToLFN(dest)

    runXRDCommand(src,'mv', lfnToEOS(dest))
                
def matchingFiles( path, regexp):
    """Return a list of files matching a regexp"""

    # print path, regexp
    pattern = re.compile( regexp )
    #files = ls_EOS(path)
    files = ls(path)
    # print files
    return [f for f in files if pattern.match(os.path.basename(f)) is not None]

def datasetNotEmpty( path, regexp ):
    pattern = re.compile( regexp )
    files = ls_EOS(path)
    
    for f in files:
        if pattern.match( os.path.basename(f) ) is not None:
            return 1
    return 0
    
def cmsStage( absDestDir, files, force):
    """Runs cmsStage with LFNs if possible"""

    destIsEOSDir = isEOSDir(absDestDir)
    if destIsEOSDir: 
        createEOSDir( absDestDir )

    for fname in files:
        command = ['cmsStage']
        if force:
            command.append('-f')
        command.append(eosToLFN(fname))
        command.append(eosToLFN(absDestDir))
        print ' '.join(command)
        runner = cmsIO.cmsFileManip()
        runner.runCommand(command)
