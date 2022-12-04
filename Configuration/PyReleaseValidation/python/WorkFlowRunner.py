from __future__ import print_function
from threading import Thread
from Configuration.PyReleaseValidation import WorkFlow
import os,time
import shutil
from subprocess import Popen 
from os.path import exists, basename, join
from datetime import datetime

class WorkFlowRunner(Thread):
    def __init__(self, wf, noRun=False,dryRun=False,cafVeto=True,dasOptions="",jobReport=False, nThreads=1, nStreams=0, maxSteps=9999):
        Thread.__init__(self)
        self.wf = wf

        self.status=-1
        self.report=''
        self.nfail=0
        self.npass=0
        self.noRun=noRun
        self.dryRun=dryRun
        self.cafVeto=cafVeto
        self.dasOptions=dasOptions
        self.jobReport=jobReport
        self.nThreads=nThreads
        self.nStreams=nStreams
        self.maxSteps=maxSteps
        
        self.wfDir=str(self.wf.numId)+'_'+self.wf.nameId
        return

    def doCmd(self, cmd):

        msg = "\n# in: " +os.getcwd()
        if self.dryRun: msg += " dryRun for '"
        else:      msg += " going to execute "
        msg += cmd.replace(';','\n')
        print(msg)

        cmdLog = open(self.wfDir+'/cmdLog','a')
        cmdLog.write(msg+'\n')
        cmdLog.close()
        
        ret = 0
        if not self.dryRun:
            p = Popen(cmd, shell=True)
            ret = os.waitpid(p.pid, 0)[1]
            if ret != 0:
                print("ERROR executing ",cmd,'ret=', ret)

        return ret
    
    def run(self):

        startDir = os.getcwd()

        if not os.path.exists(self.wfDir):
            os.makedirs(self.wfDir)
        elif not self.dryRun: # clean up to allow re-running in the same overall devel area, then recreate the dir to make sure it exists
            print("cleaning up ", self.wfDir, ' in ', os.getcwd())
            shutil.rmtree(self.wfDir) 
            os.makedirs(self.wfDir)

        preamble = 'cd '+self.wfDir+'; '
       
        realstarttime = datetime.now()
        startime='date %s' %time.asctime()

        # check where we are running:
        onCAF = False
        if 'cms/caf/cms' in os.environ['CMS_PATH']:
            onCAF = True

        ##needs to set
        #self.report
        self.npass  = []
        self.nfail = []
        self.stat = []
        self.retStep = []

        def closeCmd(i,ID):
            return ' > %s 2>&1; ' % ('step%d_'%(i,)+ID+'.log ',)

        inFile=None
        lumiRangeFile=None
        aborted=False
        for (istepmone,com) in enumerate(self.wf.cmds):
            # isInputOk is used to keep track of the das result. In case this
            # is False we use a different error message to indicate the failed
            # das query.
            isInputOk=True
            istep=istepmone+1
            cmd = preamble
            if aborted:
                self.npass.append(0)
                self.nfail.append(0)
                self.retStep.append(0)
                self.stat.append('NOTRUN')
                continue
            if not isinstance(com,str):
                if self.cafVeto and (com.location == 'CAF' and not onCAF):
                    print("You need to be no CAF to run",self.wf.numId)
                    self.npass.append(0)
                    self.nfail.append(0)
                    self.retStep.append(0)
                    self.stat.append('NOTRUN')
                    aborted=True
                    continue
                #create lumiRange file first so if das fails we get its error code
                cmd2 = com.lumiRanges()
                if cmd2:
                    cmd2 =cmd+cmd2+closeCmd(istep,'lumiRanges')
                    lumiRangeFile='step%d_lumiRanges.log'%(istep,)
                    retStep = self.doCmd(cmd2)
                if (com.dataSetParent):
                    cmd3=cmd+com.das(self.dasOptions,com.dataSetParent)+closeCmd(istep,'dasparentquery')
                    retStep = self.doCmd(cmd3)
                cmd+=com.das(self.dasOptions,com.dataSet)
                cmd+=closeCmd(istep,'dasquery')
                retStep = self.doCmd(cmd)
                #don't use the file list executed, but use the das command of cmsDriver for next step
                # If the das output is not there or it's empty, consider it an
                # issue of this step, not of the next one.
                dasOutputPath = join(self.wfDir, 'step%d_dasquery.log'%(istep,))
                # Check created das output in no-dryRun mode only
                if not self.dryRun:
                    if not exists(dasOutputPath):
                        retStep = 1
                        dasOutput = None
                    else:
                        # We consider only the files which have at least one logical filename
                        # in it. This is because sometimes das fails and still prints out junk.
                        dasOutput = [l for l in open(dasOutputPath).read().split("\n") if l.startswith("/")]
                    if not dasOutput:
                        retStep = 1
                        isInputOk = False
                 
                inFile = 'filelist:' + basename(dasOutputPath)
                print("---")
            else:
                #chaining IO , which should be done in WF object already and not using stepX.root but <stepName>.root
                cmd += com
                if self.noRun:
                    cmd +=' --no_exec'
                # in case previous step used DAS query (either filelist of das:)
                # not to be applied for premixing stage1 to allow combiend stage1+stage2 workflow
                if inFile and not 'premix_stage1' in cmd:
                    cmd += ' --filein '+inFile
                    inFile=None
                if lumiRangeFile: #DAS query can also restrict lumi range
                    cmd += ' --lumiToProcess '+lumiRangeFile
                    lumiRangeFile=None
                # 134 is an existing workflow where harvesting has to operate on AlcaReco and NOT on DQM; hard-coded..    
                if 'HARVESTING' in cmd and not 134==self.wf.numId and not '--filein' in cmd:
                    cmd+=' --filein file:step%d_inDQM.root --fileout file:step%d.root '%(istep-1,istep)
                else:
                    # Disable input for premix stage1 to allow combined stage1+stage2 workflow
                    # Disable input for premix stage2 in FastSim to allow combined stage1+stage2 workflow (in FS, stage2 does also GEN)
                    # Ugly hack but works
                    if istep!=1 and not '--filein' in cmd and not 'premix_stage1' in cmd and not ("--fast" in cmd and "premix_stage2" in cmd):
                        cmd+=' --filein  file:step%s.root '%(istep-1,)
                    if not '--fileout' in com:
                        cmd+=' --fileout file:step%s.root '%(istep,)
                if self.jobReport:
                  cmd += ' --suffix "-j JobReport%s.xml " ' % istep
                if (self.nThreads > 1) and ('HARVESTING' not in cmd) and ('ALCAHARVEST' not in cmd):
                  cmd += ' --nThreads %s' % self.nThreads
                if (self.nStreams > 0) and ('HARVESTING' not in cmd) and ('ALCAHARVEST' not in cmd):
                  cmd += ' --nStreams %s' % self.nStreams
                cmd+=closeCmd(istep,self.wf.nameId)            
                retStep = 0
                if istep>self.maxSteps:
                   wf_stats = open("%s/wf_steps.txt" % self.wfDir,"a")
                   wf_stats.write('step%s:%s\n' % (istep, cmd))
                   wf_stats.close()
                else: retStep = self.doCmd(cmd)
            
            self.retStep.append(retStep)
            if retStep == 32000:
                # A timeout occurred
                self.npass.append(0)
                self.nfail.append(1)
                self.stat.append('TIMEOUT')
                aborted = True
            elif (retStep!=0):
                #error occured
                self.npass.append(0)
                self.nfail.append(1)
                if not isInputOk:
                  self.stat.append("DAS_ERROR")
                else:
                  self.stat.append('FAILED')
                #to skip processing
                aborted=True
            else:
                #things went fine
                self.npass.append(1)
                self.nfail.append(0)
                self.stat.append('PASSED')

        os.chdir(startDir)
        endtime='date %s' %time.asctime()
        tottime='%s-%s'%(endtime,startime)
        

        #### wrap up ####

        logStat=''
        for i,s in enumerate(self.stat):
            logStat+='Step%d-%s '%(i,s)
        #self.report='%s_%s+%s %s - time %s; exit: '%(self.wf.numId,self.wf.nameId,'+'.join(self.wf.stepList),logStat,tottime)+' '.join(map(str,self.retStep))+'\n'
        self.report='%s_%s %s - time %s; exit: '%(self.wf.numId,self.wf.nameId,logStat,tottime)+' '.join(map(str,self.retStep))+'\n'

        return 

