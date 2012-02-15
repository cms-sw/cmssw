
from threading import Thread

from Configuration.PyReleaseValidation import WorkFlow
import os,time
from subprocess import Popen 

class WorkFlowRunner(Thread):
    def __init__(self, wf, noRun=False,dryRun=False):
        Thread.__init__(self)
        self.wf = wf

        self.status=-1
        self.report=''
        self.nfail=0
        self.npass=0
        self.noRun=noRun
        self.dryRun=dryRun
        
        self.wfDir=str(self.wf.numId)+'_'+self.wf.nameId
        return

    def doCmd(self, cmd):

        msg = "\n# in: " +os.getcwd()
        if self.dryRun: msg += " dryRun for '"
        else:      msg += " going to execute "
        msg += cmd.replace(';','\n')
        print msg

        cmdLog = open(self.wfDir+'/cmdLog','a')
        cmdLog.write(msg+'\n')
        cmdLog.close()
        
        ret = 0
        if not self.dryRun:
            p = Popen(cmd, shell=True)
            ret = os.waitpid(p.pid, 0)[1]
            if ret != 0:
                print "ERROR executing ",cmd,'ret=', ret

        return ret
    
    def run(self):

        startDir = os.getcwd()

        if not os.path.exists(self.wfDir):
            os.makedirs(self.wfDir)

        preamble = 'cd '+self.wfDir+'; '
       
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
        aborted=False
        for (istepmone,com) in enumerate(self.wf.cmds):
            istep=istepmone+1
            cmd = preamble
            if aborted:
                self.npass.append(0)
                self.nfail.append(0)
                self.retStep.append(0)
                self.stat.append('NOTRUN')
                continue
            if not isinstance(com,str):
                if com.location == 'CAF' and not onCAF:
                    print "You need to be no CAF to run",self.wf.numId
                    self.npass.append(0)
                    self.nfail.append(0)
                    self.retStep.append(0)
                    self.stat.append('NOTRUN')
                    aborted=True
                    continue
                cmd+=com.dbs()
                cmd+=closeCmd(istep,'dbsquery')
                retStep = self.doCmd(cmd)
                #don't use the file list executed, but use the dbs command of cmsDriver for next step
                inFile='filelist:step%d_dbsquery.log'%(istep,)
                print "---"
            else:
                #chaining IO , which should be done in WF object already and not using stepX.root but <stepName>.root
                cmd += com
                if self.noRun:
                    cmd +=' --no_exec'
                if inFile: #in case previous step used DBS query (either filelist of dbs:)
                    cmd += ' --filein '+inFile
                    inFile=None
                if 'HARVESTING' in cmd and not '134' in str(self.wf.numId) and not '--filein' in cmd:
                    cmd+=' --filein file:step%d_inDQM.root --fileout file:step%d.root '%(istep-1,istep)
                else:
                    if istep!=1 and not '--filein' in cmd:
                        cmd+=' --filein file:step%s.root '%(istep-1,)
                    if not '--fileout' in com:
                        cmd+=' --fileout file:step%s.root '%(istep,)
                    
                                

                cmd+=closeCmd(istep,self.wf.nameId)            
                retStep = self.doCmd(cmd)
            
            self.retStep.append(retStep)
            if (retStep!=0):
                #error occured
                self.npass.append(0)
                self.nfail.append(1)
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
        self.report='%s_%s %s - time %s; exit: '%(self.wf.numId,self.wf.nameId,logStat,tottime)+' '.join(map(str,self.retStep))+'\n'

        return 



