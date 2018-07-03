#!/usr/bin/env python
from __future__ import absolute_import
import os
import sys
import commands
import time
import optparse
from . import Config

usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-r', '--run'        ,    dest='runNumber'          , help='Run number to process (-1 --> Submit all)', default='-1')
parser.add_option('-f', '--files'      ,    dest='files'              , help='files to process'                         , default='-1')
parser.add_option('-n', '--firstFile'  ,    dest='firstFile'          , help='first File to process'                    , default='-1')
parser.add_option('-a', '--aag'        ,    dest='aag'                , help='Dataset type (is Aag)'                    , default=False, action="store_true")
parser.add_option('-c', '--corrupted'  ,    dest='corrupted'          , help='Check for corrupted runs'                 , default=False, action="store_true")
parser.add_option('-d', '--dataset'    ,    dest='dataset'            , help='dataset'                                  , default='')
parser.add_option('-s', '--stageout'   ,    dest='stageout'           , help='stageout produced files'                  , default="True")

(opt, args) = parser.parse_args()


print "Running calib tree production..."

files     = opt.files
nFiles    = len(files.split(","))-1
firstFile = int(opt.firstFile)
checkCorrupted = opt.corrupted
AAG      = opt.aag
stageout = (opt.stageout=="True")
print "After the abort gap : %s"%AAG
conf           = Config.configuration(opt.aag)
run            = int(opt.runNumber)
dataset        = opt.dataset
print conf.checkIntegrity()
print conf
print files

PWDDIR  =os.getcwd() #Current Dir
print "Running on %s"%PWDDIR
os.chdir(conf.RUNDIR);
if firstFile<0:firstFile=0

print "Processing files %i to %i of run %i" % (firstFile,firstFile+nFiles,run)


if(firstFile==0):outfile = 'calibTree_%i.root' % (run)
else:
   outfile = 'calibTree_%i_%i.root' % (run, firstFile)
#reinitialize the afs token, to make sure that the job isn't kill after a few hours of running
os.system('/usr/sue/bin/kinit -R')


#BUILD CMSSW CONFIG, START CMSRUN, COPY THE OUTPUT AND CLEAN THE PROJECT
cmd='cmsRun produceCalibrationTree_template_cfg.py'
cmd+=' outputFile="'+PWDDIR+'/'+outfile+'"'
cmd+=' conditionGT="'+conf.globalTag+'"'
cmd+=' inputCollection="'+conf.collection+'"'
if files[-1]==",":files=files[:-1]
cmd+=' inputFiles="'+files.replace("'","")+'"'
cmd+=' runNumber=%s'%run
print cmd

exit_code = os.system(conf.initEnv+cmd)
stageOutCode = True
if(int(exit_code)!=0):
   print("Job Failed with ExitCode "+str(exit_code))
   os.system('echo %i %i %i >> FailledRun%s.txt' % (run, firstFile, firstFile+nFiles,'_Aag' if AAG else ''))
else:
   FileSizeInKBytes =commands.getstatusoutput('ls  -lth --block-size=1024 '+PWDDIR+'/'+outfile)[1].split()[4]
   if(int(FileSizeInKBytes)>10 and stageout):
      print("Preparing for stageout of " + PWDDIR+'/'+outfile + ' on ' + conf.CASTORDIR+'/'+outfile + '.  The file size is %d KB' % int(FileSizeInKBytes))
      cpCmd = "eos cp %s/%s "%(PWDDIR,outfile)
      cpCmd+= "root://eoscms.cern.ch//eos/cms/%s/%s"%(conf.CASTORDIR,outfile)
      stageOutCode&= not os.system(conf.initEnv+" "+cpCmd)
      print conf.eosLs + conf.CASTORDIR+'/'+outfile
      stageOutCode&= not os.system("eos ls " + conf.CASTORDIR+'/'+outfile)
   else:
      print('File size is %d KB, this is under the threshold --> the file will not be transfered on EOS' % int(FileSizeInKBytes))
      print "Stageout status = %s"%stageout
if not stageOutCode:
   print "WARNING WARNING WARNING STAGE OUT FAILED BUT NOT RELAUNCHED"

os.system('ls -lth '+PWDDIR+'/'+outfile)
if stageout:
   os.system('rm -f '+PWDDIR+'/'+outfile)
   os.system('rm -f ConfigFile_'+str(run)+'_'+str(firstFile)+'_cfg.py')
   os.system('cd ' + conf.RUNDIR)
   os.system('rm -rf LSFJOB_${LSB_JOBID}')
