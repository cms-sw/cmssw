#!/usr/bin/env python

import sys
import imp
import copy
import os
import shutil
import pickle
import json
import math
from PhysicsTools.HeppyCore.utils.batchmanager import BatchManager
from PhysicsTools.HeppyCore.framework.config import split

import PhysicsTools.HeppyCore.framework.looper as looper

def batchScriptPADOVA( index, jobDir='./'):
   '''prepare the LSF version of the batch script, to run on LSF'''
   script = """#!/bin/bash
#BSUB -q local
#BSUB -J test
#BSUB -o test.log
cd {jdir}
echo 'PWD:'
pwd
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
echo 'environment:'
echo
env > local.env
env
# ulimit -v 3000000 # NO
echo 'copying job dir to worker'
eval `scram runtime -sh`
ls
echo 'running'
python {looper} pycfg.py config.pck --options=options.json >& local.output
exit $? 
#echo
#echo 'sending the job directory back'
#echo cp -r Loop/* $LS_SUBCWD 
""".format(looper=looper.__file__, jdir=jobDir)

   return script

def batchScriptPISA( index, remoteDir=''):
   '''prepare the LSF version of the batch script, to run on LSF'''
   script = """#!/bin/bash
#BSUB -q cms
echo 'PWD:'
pwd
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
echo 'environment:'
echo
env > local.env
env
# ulimit -v 3000000 # NO
echo 'copying job dir to worker'
###cd $CMSSW_BASE/src
eval `scramv1 runtime -sh`
#eval `scramv1 ru -sh`
# cd $LS_SUBCWD
# eval `scramv1 ru -sh`
##cd -
##cp -rf $LS_SUBCWD .
ls
echo `find . -type d | grep /`
echo 'running'
python {looper} pycfg.py config.pck --options=options.json >& local.output
exit $? 
#echo
#echo 'sending the job directory back'
#echo cp -r Loop/* $LS_SUBCWD 
""".format(looper=looper.__file__)
   return script

def batchScriptCERN( jobDir, remoteDir=''):
   '''prepare the LSF version of the batch script, to run on LSF'''

   dirCopy = """echo 'sending the logs back'  # will send also root files if copy failed
rm Loop/cmsswPreProcessing.root
cp -r Loop/* $LS_SUBCWD
if [ $? -ne 0 ]; then
   echo 'ERROR: problem copying job directory back'
else
   echo 'job directory copy succeeded'
fi"""
   if remoteDir=='':
      cpCmd=dirCopy
   elif  remoteDir.startswith("root://eoscms.cern.ch//eos/cms/store/"):
       cpCmd="""echo 'sending root files to remote dir'
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH # 
for f in Loop/*/tree*.root
do
   rm Loop/cmsswPreProcessing.root
   ff=`echo $f | cut -d/ -f2`
   ff="${{ff}}_`basename $f | cut -d . -f 1`"
   echo $f
   echo $ff
   export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
   source $VO_CMS_SW_DIR/cmsset_default.sh
   for try in `seq 1 3`; do
      echo "Stageout try $try"
      echo "/afs/cern.ch/project/eos/installation/pro/bin/eos.select mkdir {srm}"
      /afs/cern.ch/project/eos/installation/pro/bin/eos.select mkdir {srm}
      echo "/afs/cern.ch/project/eos/installation/pro/bin/eos.select cp `pwd`/$f {srm}/${{ff}}_{idx}.root"
      /afs/cern.ch/project/eos/installation/pro/bin/eos.select cp `pwd`/$f {srm}/${{ff}}_{idx}.root
      if [ $? -ne 0 ]; then
         echo "ERROR: remote copy failed for file $ff"
         continue
      fi
      echo "remote copy succeeded"
      remsize=$(/afs/cern.ch/project/eos/installation/pro/bin/eos.select find --size {srm}/${{ff}}_{idx}.root | cut -d= -f3) 
      locsize=$(cat `pwd`/$f | wc -c)
      ok=$(($remsize==$locsize))
      if [ $ok -ne 1 ]; then
         echo "Problem with copy (file sizes don't match), will retry in 30s"
         sleep 30
         continue
      fi
      echo "everything ok"
      rm $f
      echo root://eoscms.cern.ch/{srm}/${{ff}}_{idx}.root > $f.url
      break
   done
done
cp -r Loop/* $LS_SUBCWD
if [ $? -ne 0 ]; then
   echo 'ERROR: problem copying job directory back'
else
   echo 'job directory copy succeeded'
fi
""".format(
          idx = jobDir[jobDir.find("_Chunk")+6:].strip("/") if '_Chunk' in jobDir else 'all',
          srm = (""+remoteDir+jobDir[ jobDir.rfind("/") : (jobDir.find("_Chunk") if '_Chunk' in jobDir else len(jobDir)) ]).replace("root://eoscms.cern.ch/","")
          )
   else:
       print "chosen location not supported yet: ", remoteDir
       print 'path must start with /store/'
       sys.exit(1)

   script = """#!/bin/bash
#BSUB -q 8nm
echo 'environment:'
echo
env | sort
# ulimit -v 3000000 # NO
echo 'copying job dir to worker'
cd $CMSSW_BASE/src
eval `scramv1 ru -sh`
# cd $LS_SUBCWD
# eval `scramv1 ru -sh`
cd -
cp -rf $LS_SUBCWD .
ls
cd `find . -type d | grep /`
echo 'running'
python {looper} pycfg.py config.pck --options=options.json
echo
{copy}
""".format(looper=looper.__file__, copy=cpCmd)

   return script



def batchScriptCERN_FCC( jobDir ):
   '''prepare the LSF version of the batch script, to run on LSF'''

   dirCopy = """echo 'sending the logs back'  # will send also root files if copy failed
cp -r Loop/* $LS_SUBCWD
if [ $? -ne 0 ]; then
   echo 'ERROR: problem copying job directory back'
else
   echo 'job directory copy succeeded'
fi"""
   cpCmd=dirCopy

   script = """#!/bin/bash
#BSUB -q 8nm
# ulimit -v 3000000 # NO
unset LD_LIBRARY_PATH
echo 'copying job dir to worker'
source /afs/cern.ch/exp/fcc/sw/0.7/init_fcc_stack.sh
cd $HEPPY
source ./init.sh
echo 'environment:'
echo
env | sort
echo
which python
cd -
cp -rf $LS_SUBCWD .
ls
cd `find . -type d | grep /`
echo 'running'
python {looper} pycfg.py config.pck
echo
{copy}
""".format(looper=looper.__file__, copy=cpCmd)

   return script


def batchScriptPSI( index, jobDir, remoteDir=''):
   '''prepare the SGE version of the batch script, to run on the PSI tier3 batch system'''

   cmssw_release = os.environ['CMSSW_BASE']
   VO_CMS_SW_DIR = "/swshare/cms"  # $VO_CMS_SW_DIR doesn't seem to work in the new SL6 t3wn

   if remoteDir=='':
       cpCmd="""echo 'sending the job directory back'
rm Loop/cmsswPreProcessing.root
cp -r Loop/* $SUBMISIONDIR"""
   elif remoteDir.startswith("/pnfs/psi.ch"):
       cpCmd="""echo 'sending root files to remote dir'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/dcap/ # Fabio's workaround to fix gfal-tools
for f in Loop/mt2*.root
do
   ff=`basename $f | cut -d . -f 1`
   #d=`echo $f | cut -d / -f 2`
   gfal-mkdir {srm}
   echo "gfal-copy file://`pwd`/Loop/$ff.root {srm}/${{ff}}_{idx}.root"
   gfal-copy file://`pwd`/Loop/$ff.root {srm}/${{ff}}_{idx}.root
   if [ $? -ne 0 ]; then
      echo "ERROR: remote copy failed for file $ff"
   else
      echo "remote copy succeeded"
      rm Loop/$ff.root
   fi
done
rm Loop/cmsswPreProcessing.root
cp -r Loop/* $SUBMISIONDIR""".format(idx=index, srm='srm://t3se01.psi.ch'+remoteDir+jobDir[jobDir.rfind("/"):jobDir.find("_Chunk")])
   else:
      print "remote directory not supported yet: ", remoteDir
      print 'path must start with "/pnfs/psi.ch"'
      sys.exit(1)
      

   script = """#!/bin/bash
shopt expand_aliases
##### MONITORING/DEBUG INFORMATION ###############################
DATE_START=`date +%s`
echo "Job started at " `date`
cat <<EOF
################################################################
## QUEUEING SYSTEM SETTINGS:
HOME=$HOME
USER=$USER
JOB_ID=$JOB_ID
JOB_NAME=$JOB_NAME
HOSTNAME=$HOSTNAME
TASK_ID=$TASK_ID
QUEUE=$QUEUE

EOF
echo "######## Environment Variables ##########"
env
echo "################################################################"
TOPWORKDIR=/scratch/`whoami`
JOBDIR=sgejob-$JOB_ID
WORKDIR=$TOPWORKDIR/$JOBDIR
SUBMISIONDIR={jdir}
if test -e "$WORKDIR"; then
   echo "ERROR: WORKDIR ($WORKDIR) already exists! Aborting..." >&2
   exit 1
fi
mkdir -p $WORKDIR
if test ! -d "$WORKDIR"; then
   echo "ERROR: Failed to create workdir ($WORKDIR)! Aborting..." >&2
   exit 1
fi

#source $VO_CMS_SW_DIR/cmsset_default.sh
source {vo}/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc481
#cd $CMSSW_BASE/src
cd {cmssw}/src
shopt -s expand_aliases
cmsenv
cd $WORKDIR
cp -rf $SUBMISIONDIR .
ls
cd `find . -type d | grep /`
echo 'running'
python {looper} pycfg.py config.pck --options=options.json
echo
{copy}
###########################################################################
DATE_END=`date +%s`
RUNTIME=$((DATE_END-DATE_START))
echo "################################################################"
echo "Job finished at " `date`
echo "Wallclock running time: $RUNTIME s"
exit 0
""".format(jdir=jobDir, vo=VO_CMS_SW_DIR,cmssw=cmssw_release, looper=looper.__file__, copy=cpCmd)

   return script

def batchScriptIC(jobDir):
   '''prepare a IC version of the batch script'''


   cmssw_release = os.environ['CMSSW_BASE']
   script = """#!/bin/bash
export X509_USER_PROXY=/home/hep/$USER/myproxy
source /vols/cms/grid/setup.sh
cd {jobdir}
cd {cmssw}/src
eval `scramv1 ru -sh`
cd -
echo 'running'
python {looper} pycfg.py config.pck --options=options.json
echo
echo 'sending the job directory back'
mv Loop/* ./ && rm -r Loop
""".format(jobdir = jobDir, looper=looper.__file__, cmssw = cmssw_release)
   return script

def batchScriptLocal(  remoteDir, index ):
   '''prepare a local version of the batch script, to run using nohup'''

   script = """#!/bin/bash
echo 'running'
python {looper} pycfg.py config.pck --options=options.json
echo
echo 'sending the job directory back'
mv Loop/* ./
""" 
   return script


class MyBatchManager( BatchManager ):
   '''Batch manager specific to cmsRun processes.''' 
         
   def PrepareJobUser(self, jobDir, value ):
      '''Prepare one job. This function is called by the base class.'''
      print value
      print self.components[value]

      #prepare the batch script
      scriptFileName = jobDir+'/batchScript.sh'
      scriptFile = open(scriptFileName,'w')
      storeDir = self.remoteOutputDir_.replace('/castor/cern.ch/cms','')
      mode = self.RunningMode(self.options_.batch)
      if mode == 'LXPLUS':
         if 'CMSSW_BASE' in os.environ and not 'PODIO' in os.environ:  
            scriptFile.write( batchScriptCERN( jobDir, storeDir) )
         elif 'PODIO' in os.environ:
            #FCC case
            scriptFile.write( batchScriptCERN_FCC( jobDir ) )
         else: 
            assert(False)
      elif mode == 'PSI':
         # storeDir not implemented at the moment
         scriptFile.write( batchScriptPSI ( value, jobDir, storeDir ) ) 
      elif mode == 'LOCAL':
         # watch out arguments are swapped (although not used)         
         scriptFile.write( batchScriptLocal( storeDir, value) ) 
      elif mode == 'PISA' :
         scriptFile.write( batchScriptPISA( storeDir, value) ) 	
      elif mode == 'PADOVA' :
         scriptFile.write( batchScriptPADOVA( value, jobDir) )        
      elif mode == 'IC':
         scriptFile.write( batchScriptIC(jobDir) )
      scriptFile.close()
      os.system('chmod +x %s' % scriptFileName)

      shutil.copyfile(self.cfgFileName, jobDir+'/pycfg.py')
#      jobConfig = copy.deepcopy(config)
#      jobConfig.self.components = [ self.components[value] ]
      cfgFile = open(jobDir+'/config.pck','w')
      pickle.dump(  self.components[value] , cfgFile )
      # pickle.dump( cfo, cfgFile )
      cfgFile.close()
      if hasattr(self,"heppyOptions_"):
         optjsonfile = open(jobDir+'/options.json','w')
         optjsonfile.write(json.dumps(self.heppyOptions_))
         optjsonfile.close()


def create_batch_manager(): 
   batchManager = MyBatchManager()
   batchManager.parser_.usage="""
    %prog [options] <cfgFile>

    Run Colin's python analysis system on the batch.
    Job splitting is determined by your configuration file.
    """
   return batchManager


def main(options, args, batchManager): 
   batchManager.cfgFileName = args[0]

   handle = open(batchManager.cfgFileName, 'r')
   cfo = imp.load_source("pycfg", batchManager.cfgFileName, handle)
   config = cfo.config
   handle.close()

   batchManager.components = split( [comp for comp in config.components \
                                        if len(comp.files)>0] )
   listOfValues = range(0, len(batchManager.components))
   listOfNames = [comp.name for comp in batchManager.components]

   batchManager.PrepareJobs( listOfValues, listOfNames )
   waitingTime = 0.1
   batchManager.SubmitJobs( waitingTime )
   

if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.heppy_loop import _heppyGlobalOptions
    for opt in options.extraOptions:
        if "=" in opt:
            (key,val) = opt.split("=",1)
            _heppyGlobalOptions[key] = val
        else:
            _heppyGlobalOptions[opt] = True
    batchManager.heppyOptions_=_heppyGlobalOptions

   batchManager = create_batch_manager() 
   options, args = batchManager.ParseOptions()
   main(options, args, batchManager)
