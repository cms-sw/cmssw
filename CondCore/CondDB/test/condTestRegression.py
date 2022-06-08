#!/usr/bin/env python3

from __future__ import print_function
import sys, os
import glob
import time
import subprocess
import shutil
import re
import json

# Boost 1.51 [no writers available - a back-port is required!]
#readers = { 'CMSSW_7_1_0'        : ['slc6_amd64_gcc490'], 
#          }

# Boost 1.57
#readers = { 'CMSSW_7_5_0'        : ['slc6_amd64_gcc491'], 
#          }

# Boost 1.63
#readers = { 'CMSSW_9_0_0'        : ['slc6_amd64_gcc530'], 
#          }

# Boost 1.67 [No reference release yet...]
readers = {
          }

writers = { 'CMSSW_9_0_1'        : [ ('slc6_amd64_gcc630', 'ref901-s6630.db')],
            'CMSSW_8_1_0'        : [ ('slc6_amd64_gcc530', 'ref750-s6530.db'),('slc6_amd64_gcc600', 'ref750-s600.db')],
            'CMSSW_7_6_6'        : [ ('slc6_amd64_gcc493', 'ref750-s6493.db')]
          }

os2image_overrides = {"slc7": "cc7"}

def check_output(*popenargs, **kwargs):
    '''Mimics subprocess.check_output() in Python 2.6
    '''

    process = subprocess.Popen(*popenargs, **kwargs)
    stdout, stderr = process.communicate()
    returnCode = process.returncode

    if returnCode:
        msg = '\nERROR from process (ret=%s): \n' %(str(returnCode),)
        msg += '      stderr: %s\n' % (str(stderr),)
        msg += '      stdout: %s\n' % (str(stdout),)
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(returnCode, cmd+msg)

    return stdout


# nice one from:
# https://www.daniweb.com/software-development/python/code/216610/timing-a-function-python
def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print('\n%s(%s) took %0.3f ms\n' % (func.__name__, ','.join([str(x) for x in arg[1:]]), (t2-t1)*1000.0))
        return res
    return wrapper


class CondRegressionTester(object):

    @print_timing
    def __init__(self):

        tmpBase = '/tmp'
        if 'CMSSW_BASE' in os.environ:
            tmpBase = os.path.join(os.environ['CMSSW_BASE'],'tmp')
        self.topDir = os.path.join( tmpBase, 'cmsCondRegTst-'+time.strftime('%Y-%m-%d-%H-%M'))
        if not os.path.exists(self.topDir): os.makedirs(self.topDir)

        self.dbDir = os.path.join( self.topDir, 'dbDir' )
        if not os.path.exists(self.dbDir): os.makedirs(self.dbDir)

        self.logDir = os.path.join( self.topDir, 'logs' )
        if not os.path.exists(self.logDir): 
            os.makedirs(self.logDir)
        else:  # if it exists, remove the logDir and re-create it
            shutil.rmtree(self.logDir, ignore_errors=True)
            os.makedirs(self.logDir)

        # add the IB/release itself:
        self.regTestSrcDir = os.path.join( os.environ['LOCALRT'], 'src', 'CondCore', 'CondDB', 'test' )
        self.rel = os.environ['CMSSW_VERSION']
        self.arch = os.environ['SCRAM_ARCH']
        self.dbName = 'self-%s-%s.db' % (self.rel, self.arch)

        self.dbList = {}

        self.status = {}

        return

    def summary(self, verbose=False, jsonOut=False):
        if verbose: 
            allReaders = dict(readers)
            allReaders['SELF']=['%s' %(self.arch)]
            dbNames = []
            header = ( 'Write', )
            reslen = len(header[0])
            for result in sorted(self.status.keys()):
                if len(result)>reslen:
                    reslen = len(result)

            fmt = ' %' + '%s' %(reslen) + 's '
            for reader in sorted(allReaders.keys()):
                for arch in allReaders[reader]:
                    if reader == 'SELF':
                        readerArch = 'Read: %s [%s]' %(self.rel,self.arch)
                    else:
                        readerArch = 'Read: %s [%s]' %(reader,arch)
                    fmt += '| %' + '%s' %len(readerArch) + 's '
                    header += tuple([readerArch])
            fmt += '|'
            print('SELF: %s [%s]\n' %(self.rel,self.arch))
            print(fmt %header)
            for result in sorted(self.status.keys()):
                params = (result,)+tuple([self.status[result][key] for key in sorted(self.status[result].keys())])
                print(fmt %params)

        if jsonOut:
            print(json.dumps( self.status, sort_keys=True, indent=4 ))

        overall = True
        for result in sorted(self.status.keys()):
            for result_arch in sorted(self.status[result].keys()):
                if not self.status[result][result_arch]: overall = False

        return overall

    @print_timing
    def run(self, rel, arch, readOrWrite, dbName):

        if readOrWrite == 'write':
            self.dbList['%s [%s]'%(rel,arch)] = dbName

        cmd ="scram -a %s list -c %s | grep '\\b%s\\b' | head -1 | sed 's|.* ||'" %(arch,rel,rel)
        out =check_output(cmd, shell=True, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        ind = out.find( arch )
        if ind == -1:
            raise Exception('Could not locate the reference release %s with "%s" [ got %s ]' %(rel,cmd,out))

        cmsPath = out[:ind-1]
        # using wildcard to support the path for normal ( BASE/ARCH/cms/cmssw/RELEASE ) and patch releases ( BASE-PATCH/ARCH/cms/cmssw-patch/RELEASE )
        releaseDir = '%s/%s/cms/*/%s' %(cmsPath,arch,rel)

        cmd =  'source %s/cmsset_default.sh; export SCRAM_ARCH=%s; cd %s/src ; eval `scram runtime -sh`; cd - ; ' %(cmsPath,arch,releaseDir)
        cmd += "echo 'CMSSW_BASE='$CMSSW_BASE; echo 'RELEASE_BASE='$RELEASE_BASE; echo 'PATH='$PATH; echo 'LD_LIBRARY_PATH='$LD_LIBRARY_PATH;"
        cmd += '$LOCALRT/test/%s/testReadWritePayloads %s sqlite_file:///%s/%s ' % (arch, readOrWrite, self.dbDir, dbName)

        cur_os = os.environ['SCRAM_ARCH'].split("_")[0]
        rel_os = arch.split("_")[0]
        if cur_os in os2image_overrides: cur_os = os2image_overrides[cur_os]
        if rel_os in os2image_overrides: rel_os = os2image_overrides[rel_os]
        if rel_os != cur_os:
          run_script = "%s/run_condTestRegression.sh" % self.topDir
          check_output("echo '%s' > %s; chmod +x %s" % (cmd, run_script, run_script), shell=True, universal_newlines=True, env={}, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
          cmd = "%s/common/cmssw-env --cmsos %s -- %s" % (cmsPath, rel_os, run_script)
        print("Running:",cmd)
        try:
            #opening a process with a clean environment ( to avoid to inherit scram variables )
            res = check_output(cmd, shell=True, universal_newlines=True, env={}, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as e:
            self.log( rel, arch, readOrWrite, str(e) )
            raise e

        self.log( rel, arch, readOrWrite, ''.join(res) )

    @print_timing
    def runSelf(self, readOrWrite, dbNameIn=None):

        dbName = self.dbName # set the default
        if dbNameIn : dbName = dbNameIn

        if readOrWrite == 'write':
            self.dbList['%s [%s]'%(self.rel,self.arch)] = dbName

        execName = 'test/%s/testReadWritePayloads' %self.arch
        executable = '%s/%s' %(os.environ['LOCALRT'],execName)
        if not os.path.exists(executable):
            print('Executable %s not found in local release.')
            executable = None
            for rel_base_env in ['CMSSW_BASE', 'CMSSW_RELEASE_BASE', 'CMSSW_FULL_RELEASE_BASE' ]:
                if os.getenv(rel_base_env) and os.path.exists(str(os.environ[rel_base_env])+'/%s' %execName):
                    executable = str(os.environ[rel_base_env])+'/%s' %execName
                    break

        if executable is None:
            raise Exception("Can't find the %s executable." %execName )

        # we run in the local environment, but need to make sure that we start "top-level" of the devel area
        # and we assume that the test was already built 
        cmd = 'export SCRAM_ARCH=%s; cd %s/src; eval `scram runtime -sh 2>/dev/null` ; ' % (os.environ['SCRAM_ARCH'],os.environ['CMSSW_BASE'], )
        cmd += "echo 'CMSSW_BASE='$CMSSW_BASE; echo 'RELEASE_BASE='$RELEASE_BASE; echo 'PATH='$PATH; echo 'LD_LIBRARY_PATH='$LD_LIBRARY_PATH; echo 'LOCALRT='$LOCALRT;"
        cmd += '%s %s sqlite_file:///%s/%s ' % (executable, readOrWrite, self.dbDir, dbName)

        try:
            res = check_output(cmd, shell=True, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as e:
            self.log( self.rel, self.arch, readOrWrite, str(e) )
            raise e

        self.log( self.rel, self.arch, readOrWrite, ''.join(res) )

    def log(self, rel, arch, readOrWrite, msg):

        with open(os.path.join(self.logDir, readOrWrite+'-'+rel+'-'+arch+'.log'), 'a') as logFile:
            logFile.write( '\n'+'='*80+'\n' )
            logFile.write( str(msg) )

    @print_timing
    def runAll(self):

        # write all DBs (including the one from this IB/devArea)
        print('='*80)
        print("going to write DBs ...")
        self.runSelf('write')
        for rel in writers.keys():
            for arch,dbName in writers[rel]:
                self.run(rel, arch, 'write', dbName)

        # now try to read back with all reference releases all the DBs written before ...
        print('='*80)
        print("going to read back DBs ...")
        for rel in readers.keys():
            for arch in readers[rel]:
                for writer in self.dbList.keys(): # for any given rel/arch we check all written DBs
                    dbName = self.dbList[writer]
                    try:
                        self.run(rel, arch, 'read', dbName)
                        status = True
                        print("rel %s reading %s was OK." % (rel, writer))
                    except:
                        status = False
                        print("rel %s reading %s FAILED." % (rel, writer))
                    if writer not in self.status.keys():
                        key = '%s [%s]' %(rel,arch)
                        self.status[writer] = { key : status }
                    else:
                        self.status[writer]['%s [%s]' %(rel,arch)] = status 

        # ... and also with this IB/devArea
        for writer in self.dbList.keys(): # for any given rel/arch we check all written DBs
            dbName = self.dbList[writer]
            try:
                self.runSelf('read', dbName)
                status = True
                print("rel %s reading %s was OK." % (self.rel, writer))
            except:
                status = False
                print("rel %s reading %s FAILED." % (self.rel, writer))
            if writer not in self.status.keys():
                self.status[writer] = { 'SELF': status }
            else:
                self.status[writer]['SELF'] = status 
        print('='*80)

crt = CondRegressionTester()
crt.runAll()
status = crt.summary(verbose=True)
print("\n==> overall status: ", status)

# return the overall result to the caller:
if status: 
    sys.exit(0)
else:
    sys.exit(-1)

