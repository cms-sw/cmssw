#!/usr/bin/env python

import sys, os
import glob
import time
import subprocess
import shutil
import re
import json

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
        print '\n%s(%s) took %0.3f ms\n' % (func.func_name, ','.join([str(x) for x in arg[1:]]), (t2-t1)*1000.0)
        return res
    return wrapper


class CondRegressionTester(object):

      @print_timing
      def __init__(self):

          self.topDir = '/tmp/cmsCondRegTst-2015-08-13-16-14' # +time.strftime('%Y-%m-%d-%H-%M')
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

          self.dbNameList = [ self.dbName ]

	  self.status = {}
          
          return

      def summary(self, verbose=False, jsonOut=False):
          if verbose: 
             radRe = re.compile('^(CMSSW_.*?)-(slc6_amd64_gcc\d\d\d)-(.*)$')
             relArches = []
             dbNames = []
             for rad in self.status.keys():
                 radMatch = radRe.match(rad)
                 if not radMatch: print "NO match found for ", rad
                 ra = radMatch.groups()[0]+'-'+radMatch.groups()[1] 
                 if ra not in relArches: relArches.append( ra )
                 if radMatch.groups()[2] not in dbNames : dbNames.append( radMatch.groups()[2] )

             fmt =  ' %35s ' + '| %10s '*len(dbNames) + ' | '
             print fmt % tuple([' rel ']+[x[:10] for x in dbNames])
             for ra in sorted(relArches):
                 res = []
                 for db in dbNames:
                    res.append( self.status[ra+'-'+db] )
                 print fmt % tuple([ra.replace('slc6_amd64_', '')]+res)

          if jsonOut:
              print json.dumps( self.status, sort_keys=True, indent=4 )

          overall = True
          for k,v in self.status.items():
              # do not take results from 7.1.X into the overall status as this release can not 
	      # read more recent data until the corresponding boost 1.57 version is backported:
              if 'CMSSW_7_1_' in k: continue

              if not v : overall = False

          return overall


      @print_timing
      def setup(self, rel, arch):

      	  # check if we need to do anything at all:
      	  if os.path.exists( os.path.join( self.topDir, rel, 'test', arch, 'testReadWritePayloads' ) ): 
             print "area for %s/%s already setup: found %s " % (rel, arch, os.path.join( self.topDir, rel, 'test', arch, 'testReadWritePayloads' ))
          #    return

          # prepare the devel area, if it does not yet exist:
          if not os.path.exists( os.path.join( self.topDir, rel, 'src') ):
             print "going to create devel area for %s/%s " % (rel, arch)
             cmd = 'cd %s; export SCRAM_ARCH=%s; scram p CMSSW %s' % (self.topDir, arch, rel)
             res = check_output(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
          
          # check out the package and build the tests
          print "going to build the test for %s/%s " % (rel, arch)
          cmd = 'cd %s/%s/src; eval `scram run -sh`; ' % (self.topDir, rel, )
          cmd += 'git cms-addpkg CondCore/CondDB 2>&1; cd CondCore/CondDB/test; '
	  if not os.path.exists( os.path.join( self.topDir, rel, 'src', 'CondCore/CondDB/test/testReadWritePayloads.cpp') ):
             print "copying over test source and BuildFile from devArea/IB ... "
	     cmd += 'cp %s/BuildFile.xml .;' % (self.regTestSrcDir,)
             cmd += 'cp %s/MyTestData.h .;' % (self.regTestSrcDir,)
             cmd += 'cp %s/testReadWritePayloads.cpp .;' % (self.regTestSrcDir,)
          cmd += 'scram b -j 10 2>&1 ;'
          res = check_output(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

          with open(os.path.join(self.logDir, rel+arch+'-build.log'), 'w') as logFile:
             logFile.write( ''.join(res) )	

      @print_timing
      def run(self, rel, arch, readOrWrite, dbName):

          if readOrWrite == 'write':
              self.dbNameList.append( dbName )

          cmd = 'cd %s/%s/src; eval `scram run -sh 2>/dev/null` ; ' % (self.topDir, rel, )
          cmd += '../test/%s/testReadWritePayloads %s sqlite_file:///%s/%s ' % (arch, readOrWrite, self.dbDir, dbName)
          
	  try:
             res = check_output(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
          except Exception, e:
             self.log( rel, arch, readOrWrite, str(e) )
             raise e

          self.log( rel, arch, readOrWrite, ''.join(res) )

      @print_timing
      def runSelf(self, readOrWrite, dbNameIn=None):

          dbName = self.dbName # set the default
          if dbNameIn : dbName = dbNameIn

          # we run in the local environment, but need to make sure that we start "top-level" of the devel area
          # and we assume that the test was already built 
          cmd = 'cd %s/src; eval `scram run -sh 2>/dev/null` ; ' % (os.environ['CMSSW_BASE'], )
          cmd += '../test/%s/testReadWritePayloads %s sqlite_file:///%s/%s ' % (self.arch, readOrWrite, self.dbDir, dbName)
          
          try:
             res = check_output(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
          except Exception, e:
             self.log( self.rel, self.arch, readOrWrite, str(e) )
             raise e

          self.log( self.rel, self.arch, readOrWrite, ''.join(res) )

      def log(self, rel, arch, readOrWrite, msg):

          with open(os.path.join(self.logDir, readOrWrite+'-'+rel+'-'+arch+'.log'), 'a') as logFile:
             logFile.write( '\n'+'='*80+'\n' )
             logFile.write( str(msg) )

      @print_timing
      def runAll(self):

          map = { 'CMSSW_7_6_0_pre2'   : [ 'slc6_amd64_gcc493', 'ref760pre2-s6493.db'],
                  'CMSSW_7_5_1'        : [ 'slc6_amd64_gcc491', 'ref751-s6491.db'],
                  'CMSSW_7_4_9'        : [ 'slc6_amd64_gcc491', 'ref749-s6491.db'],
                  'CMSSW_7_3_6_patch1' : [ 'slc6_amd64_gcc491', 'ref736p1-s6491.db'],
		  'CMSSW_7_2_5'       : [ 'slc6_amd64_gcc481', 'ref729-s6481.db'],
		  'CMSSW_7_1_19'       : [ 'slc6_amd64_gcc481', 'ref7119-s6481.db'],
          }

          # set up the devel areas for the various reference releases
          print '='*80
          print "going to set up areas ..."
          for rel, info in map.items():
             arch, dbName = info
             self.setup(rel, arch)
          
          # write all DBs (including the one from this IB/devArea)
          print '='*80
          print "going to write DBs ..."
          self.runSelf('write')
          for rel, info in map.items():
             arch, dbName = info
             self.run(rel, arch, 'write', dbName)
          
          # now try to read back with all reference releases all the DBs written before ...
          print '='*80
          print "going to read back DBs ..."
          for rel, info in map.items():
             arch, dbName = info
             for item in self.dbNameList: # for any given rel/arch we check all written DBs
                 try:
                    self.run(rel, arch, 'read', item)
                    self.status['%s-%s-%s' % (rel,arch,item)] = True
                    print "rel %s reading %s was OK." % (rel, item)
                 except:
                    self.status['%s-%s-%s' % (rel,arch,item)] = False
                    print "rel %s reading %s FAILED." % (rel, item)

          # ... and also with this IB/devArea
          for item in self.dbNameList: # for any given rel/arch we check all written DBs
             try:
                self.runSelf('read', item)
                self.status['%s-%s-%s' % (self.rel,self.arch,item)] = True
             except:
                self.status['%s-%s-%s' % (self.rel,self.arch,item)] = False

          print '='*80

crt = CondRegressionTester()
crt.runAll()
status = crt.summary(verbose=True)
print "\n==> overall status (ignoring results from 7.1.X): ", status

# return the overall result to the caller:
if status: 
   sys.exit(0)
else:
   sys.exit(-1)

