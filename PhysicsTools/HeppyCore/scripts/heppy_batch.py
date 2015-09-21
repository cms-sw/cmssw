#!/bin/env python

import sys
import imp
import copy
import os
import shutil
import pickle
import json
import math
from PhysicsTools.HeppyCore.utils.batchmanager import BatchManager

from PhysicsTools.HeppyCore.framework.heppy_loop import split

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
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck options.json >& local.output
exit $? 
#echo
#echo 'sending the job directory back'
#echo cp -r Loop/* $LS_SUBCWD 
""".format(jdir=jobDir)

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
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck options.json >& local.output
exit $? 
#echo
#echo 'sending the job directory back'
#echo cp -r Loop/* $LS_SUBCWD 
"""
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
for f in Loop/tree*.root
do
   ff=`basename $f | cut -d . -f 1`
   echo $f
   echo $ff
   export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
   source $VO_CMS_SW_DIR/cmsset_default.sh
   echo "gfal-copy file://`pwd`/Loop/$ff.root {srm}/${{ff}}_{idx}.root" 
   gfal-copy file://`pwd`/Loop/$ff.root {srm}/${{ff}}_{idx}.root
   echo $idx 
   if [ $? -ne 0 ]; then
      echo "ERROR: remote copy failed for file $ff"
   else
      echo "remote copy succeeded"
      rm Loop/$ff.root
   fi
done
#fi
""".format(idx=jobDir[jobDir.find("_Chunk")+6:].strip("/"),  srm=""+remoteDir+jobDir[jobDir.rfind("/"):jobDir.find("_Chunk")])
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
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck options.json
echo
{copy}
""".format(copy=cpCmd)

   return script


def batchScriptPSI( index, jobDir, remoteDir=''):
   '''prepare the SGE version of the batch script, to run on the PSI tier3 batch system'''

   cmssw_release = os.environ['CMSSW_BASE']
   VO_CMS_SW_DIR = "/swshare/cms"  # $VO_CMS_SW_DIR doesn't seem to work in the new SL6 t3wn

   if remoteDir=='':
       cpCmd="""echo 'sending the job directory back'
cp -r Loop/* $SUBMISIONDIR"""
   elif remoteDir.startswith("/pnfs/psi.ch"):
       cpCmd="""echo 'sending root files to remote dir'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/dcap/ # Fabio's workaround to fix gfal-tools
for f in Loop/treeProducerSusyFullHad/*.root
do
echo $f
ff=`basename $f | cut -d . -f 1`
echo $ff
gfal-mkdir {srm}
echo "gfal-copy file:///`pwd`/Loop/treeProducerSusyFullHad/$file.root {srm}/${{ff}}_{idx}.root"
gfal-copy file:///`pwd`/Loop/treeProducerSusyFullHad/$ff.root {srm}/${{ff}}_{idx}.root
done
rm Loop/treeProducerSusyFullHad/*.root
echo 'sending the logs back'
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
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck options.json
#python $CMSSW_BASE/src/CMGTools/RootTools/python/fwlite/looper.py config.pck
echo
{copy}
###########################################################################
DATE_END=`date +%s`
RUNTIME=$((DATE_END-DATE_START))
echo "################################################################"
echo "Job finished at " `date`
echo "Wallclock running time: $RUNTIME s"
exit 0
""".format(jdir=jobDir, vo=VO_CMS_SW_DIR,cmssw=cmssw_release, copy=cpCmd)

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
python {cmssw}/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck options.json
echo
echo 'sending the job directory back'
mv Loop/* ./ && rm -r Loop
""".format(jobdir = jobDir,cmssw = cmssw_release)
   return script

def batchScriptLocal(  remoteDir, index ):
   '''prepare a local version of the batch script, to run using nohup'''

   script = """#!/bin/bash
echo 'running'
python $CMSSW_BASE/src/PhysicsTools/HeppyCore/python/framework/looper.py pycfg.py config.pck options.json
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
       print components[value]

       #prepare the batch script
       scriptFileName = jobDir+'/batchScript.sh'
       scriptFile = open(scriptFileName,'w')
       storeDir = self.remoteOutputDir_.replace('/castor/cern.ch/cms','')
       mode = self.RunningMode(options.batch)
       if mode == 'LXPLUS':
           scriptFile.write( batchScriptCERN( jobDir, storeDir) ) 
       elif mode == 'PSI':
           scriptFile.write( batchScriptPSI ( value, jobDir, storeDir ) ) # storeDir not implemented at the moment
       elif mode == 'LOCAL':
           scriptFile.write( batchScriptLocal( storeDir, value) )  # watch out arguments are swapped (although not used)
       elif mode == 'PISA' :
	   scriptFile.write( batchScriptPISA( storeDir, value) ) 	
       elif mode == 'PADOVA' :
           scriptFile.write( batchScriptPADOVA( value, jobDir) )        
       elif mode == 'IC':
           scriptFile.write( batchScriptIC(jobDir) )
       scriptFile.close()
       os.system('chmod +x %s' % scriptFileName)
       
       shutil.copyfile(cfgFileName, jobDir+'/pycfg.py')
#      jobConfig = copy.deepcopy(config)
#      jobConfig.components = [ components[value] ]
       cfgFile = open(jobDir+'/config.pck','w')
       pickle.dump(  components[value] , cfgFile )
       # pickle.dump( cfo, cfgFile )
       cfgFile.close()
       if hasattr(self,"heppyOptions_"):
          optjsonfile = open(jobDir+'/options.json','w')
          optjsonfile.write(json.dumps(self.heppyOptions_))
          optjsonfile.close()

if __name__ == '__main__':
    batchManager = MyBatchManager()
    batchManager.parser_.usage="""
    %prog [options] <cfgFile>

    Run Colin's python analysis system on the batch.
    Job splitting is determined by your configuration file.
    """

    options, args = batchManager.ParseOptions()

    from PhysicsTools.HeppyCore.framework.heppy_loop import _heppyGlobalOptions
    for opt in options.extraOptions:
        if "=" in opt:
            (key,val) = opt.split("=",1)
                _heppyGlobalOptions[key] = val
        else:
            _heppyGlobalOptions[opt] = True
    batchManager.heppyOptions_=_heppyGlobalOptions

    cfgFileName = args[0]

    handle = open(cfgFileName, 'r')
    # import pdb; pdb.set_trace()
    cfo = imp.load_source("pycfg", cfgFileName, handle)
    config = cfo.config
    handle.close()

    components = split( [comp for comp in config.components if len(comp.files)>0] )
    listOfValues = range(0, len(components))
    listOfNames = [comp.name for comp in components]

    batchManager.PrepareJobs( listOfValues, listOfNames )
    waitingTime = 0.1
    batchManager.SubmitJobs( waitingTime )

