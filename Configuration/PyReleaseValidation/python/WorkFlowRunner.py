
from threading import Thread

from Configuration.PyReleaseValidation import WorkFlow
import os,time
from subprocess import Popen 

class WorkFlowRunner(Thread):
    def __init__(self, wf):
        Thread.__init__(self)
        self.wf = wf

        self.status=-1
        self.report=''
        self.nfail=0
        self.npass=0

        return

    def doCmd(self, cmd, dryRun=False):

        msg = "\n# in: " +os.getcwd()
        if dryRun: msg += " dryRun for '"
        else:      msg += " going to execute "
        msg += cmd.replace(';','\n')
        print msg

        cmdLog = open(self.wf.numId+'_'+self.wf.nameId+'/cmdLog','a')
        cmdLog.write(msg+'\n')
        cmdLog.close()
        
        ret = 0
        if not dryRun:
            p = Popen(cmd, shell=True)
            ret = os.waitpid(p.pid, 0)[1]
            if ret != 0:
                print "ERROR executing ",cmd,'ret=', ret

        return ret
    
    def run(self):

        startDir = os.getcwd()

        wfDir = self.wf.numId+'_'+self.wf.nameId
        if not os.path.exists(wfDir):
            os.makedirs(wfDir)

        preamble = 'cd '+wfDir+'; '
       
        startime='date %s' %time.asctime()

        # check where we are running:
        onCAF = False
        if 'cms/caf/cms' in os.environ['CMS_PATH']:
            onCAF = True

        ##needs to set
        #self.report
        self.npass  = [] #size 4
        self.nfail = [] #size 4
        self.stat = []
        self.retStep = []

        def closeCmd(i,ID):
            return ' > %s 2>&1; ' % ('step%d_'%(i,)+ID+'.log ',)

        inFile=None
        aborted=False
        for (istep,com) in enumerate(self.wf.cmds):
            cmd = preamble
            if aborted:
                self.npass.append(0)
                self.nfail.append(0)
                self.retStep.append(0)
                self.stat.append('NOTRUN')
                continue
            if not isinstance(com,str):
                print "going to run with file input ... "
                #cmd+=self.wf.input.dbs() #should be taken from the com object
                cmd+=scom.dbs()
                cmd+=closeCmd(istep,'dbsquery')
                retStep = self.doCmd(cmd)
                #don't use the file list executed, but use the dbs command of cmsDriver for next step
                inFile='filelist:step%d_dbsquery.log'%(istep,)
                print "---"
            else:
                print "regular cmsRun ..."
                #chaining IO , which should be done in WF object already and not using stepX.root but <stepName>.root
                cmd += com
                if inFile: #in case previous step used DBS query (either filelist of dbs:)
                    cmd += ' --filein '+inFile
                    inFile=None
                if 'HARVESTING' in cmd and not '134' in str(self.wf.numId):
                    cmd+=' --filein file:step%d_inDQM.root --fileout file:step%d.root '%(istep-1,istep)
                else:
                    if istep!=0 and not '--filein' in cmd:
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

        #pad with NOTRUN
        if True:
            logStat=''
            for i,s in enumerate(self.stat):
                logStat+='Step%d-%s '%(i,s)
            self.report='%s_%s %s - time %s; exit: '%(self.wf.numId,self.wf.nameId,logStat,tottime)+' '.join(map(str,self.retStep))+'\n'
        else:
            #backward compatible
            for i in range(len(self.npass),4):
                if i>=len(self.wf.cmds):
                    self.stat.append('NOSTEP')
                else:
                    self.stat.append('NOTRUN')
                self.npass.append(0)
                self.nfail.append(0)
                self.retStep.append(0)

            logStat=''
            for i,s in enumerate(self.stat):
                logStat+='Step%d-%s '%(i,s)
            self.report='%s_%s %s - time %s; exit: %s %s %s %s \n' % (self.wf.numId,
                                                                      self.wf.nameId,
                                                                      logStat,
                                                                      tottime,
                                                                      self.retStep[0],
                                                                      self.retStep[1],
                                                                      self.retStep[2],
                                                                      self.retStep[3])


        return 
        # set defaults for the statuses
        stat1 = 'PASSED'
        stat2 = 'PASSED' 
        stat3 = 'PASSED'
        stat4 = 'PASSED'
        if not self.wf.cmdStep2: stat2 = 'NOSTEP'
        if not self.wf.cmdStep3: stat3 = 'NOSTEP'
        if not self.wf.cmdStep4: stat4 = 'NOSTEP'
        
        # run the first workflow:
        cmd = preamble


        inFile = None
        if self.wf.cmdStep1.startswith('DATAINPUT'):
            print "going to run with file input ... "
            if self.wf.input.run!=[]:
                run      = str(self.wf.input.run[-1])
            else:
                run=None

            label    = self.wf.input.label
            location = self.wf.input.location.lower().strip()
            if 'caf' in location and not onCAF or onCAF and 'caf' not in location:
                print "ignoring workflow ",self.wf.numId, self.wf.nameId, ' as this is on '+location+' and we are ',
                if onCAF: print 'on the CAF.'
                else:     print 'we are NOT.'
                
                self.npass = [0,0,0,0]
                self.nfail = [0,0,0,0]

                logStat = 'Step1-NOTRUN Step2-NOTRUN Step3-NOTRUN Step4-NOTRUN ' 
                self.report+='%s_%s %s - time %s; exit: %s %s %s %s \n' % (self.wf.numId, self.wf.nameId, logStat, 0, 0,0,0,0)
                return
                
            files  = str(self.wf.input.files)
            events = '10' # ignore the give number ...    str(self.wf.input.events)
            if self.wf.cmdStep2 and ' -n ' not in self.wf.cmdStep2: self.wf.cmdStep2 += ' -n ' + events
            if self.wf.cmdStep3 and ' -n ' not in self.wf.cmdStep3: self.wf.cmdStep3 += ' -n ' + events
            if self.wf.cmdStep4 and ' -n ' not in self.wf.cmdStep4: self.wf.cmdStep4 += ' -n ' + events

            print "run, files, events, label", run, files, events, label 
            cmd += 'dbs search --noheader --url=http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet '
            cmd += "--query='find file where dataset like "+self.wf.input.dataSet
            if run: cmd += " and run=" + run
            cmd += "' "
            #cmd += ' > %s 2>&1; ' % ('step1_'+self.wf.nameId+'-dbsquery.log',)
            cmd += ' > %s 2>&1; ' % ('step1_dbsquery.log',)
            retStep1 = self.doCmd(cmd)
            if retStep1 == 0:
                lf = open(wfDir+'/step1_dbsquery.log', 'r')
                lines = lf.readlines()
                lf.close()
                if not lines or len(lines)==0 :
                    inFile = "NoFileFoundInDBS"
                    retStep1 = -95
                else:
                    try:
                        toJoin=[]
                        for aline in lines:
                            if len(toJoin)>50: break
                            toJoin.append(aline.strip())
                        #inFile = lines[0].strip()
                        inFile=','.join(toJoin)
                    except Exception, e:
                        print "ERROR determining file from DBS query: ", str(e)
                        inFile = "NoFileFoundInDBS"
                        retStep1 = -90
        else:
            cmd += self.wf.cmdStep1
            if not 'fileout' in self.wf.cmdStep1:
                cmd += ' --fileout file:raw.root '
            cmd += ' > %s 2>&1; ' % ('step1_'+self.wf.nameId+'.log ',)
            retStep1 = self.doCmd(cmd)

        print " ... ret: " , retStep1

        # prepare and run the next workflows -- if the previous step was OK :
        # set some defaults
        retStep2 = 0
        retStep3 = 0
        retStep4 = 0
        if self.wf.cmdStep2 and retStep1 == 0:
            fullcmd = preamble
            fullcmd += self.wf.cmdStep2
            if ' -n ' not in fullcmd : fullcmd += ' -n -1 '
            if not 'fileout' in self.wf.cmdStep2:
                fullcmd += ' --fileout file:reco.root '
            print '=====>>> ', self.wf.nameId, self.wf.numId

            if (not '--filein' in self.wf.cmdStep2) or inFile:
                fullcmd += ' --filein '+inFile+ ' '
                
            fullcmd += ' > %s 2>&1; ' % ('step2_'+self.wf.nameId+'.log ',)
            # print fullcmd
            retStep2 = self.doCmd(fullcmd)
#            if random.randint(0,100) < 20 : retStep2 = -42

            if self.wf.cmdStep3 and retStep2 == 0:
                fullcmd = preamble
                fullcmd += self.wf.cmdStep3
                if ' -n ' not in fullcmd : fullcmd += ' -n -1 '
                # FIXME: dirty hack for beam-spot dedicated relval
                if not '134' in str(self.wf.numId):
                    if 'HARVESTING' in fullcmd:
                        fullcmd += ' --filein file:step2_inDQM.root --fileout file:step3.root '
                    else:
                        if not '--filein' in fullcmd:
                            fullcmd += ' --filein file:step2.root'
                        if not 'fileout' in fullcmd:
                            fullcmd += '--fileout file:step3.root '
                fullcmd += ' > %s 2>&1; ' % ('step3_'+self.wf.nameId+'.log ',)
                retStep3 = self.doCmd(fullcmd)
                if self.wf.cmdStep4 and retStep3 == 0:
                    fullcmd = preamble
                    fullcmd += self.wf.cmdStep4
                    if ' -n ' not in fullcmd : fullcmd += ' -n -1 '
                    # FIXME: dirty hack for beam-spot dedicated relval
                    if not '134' in str(self.wf.numId):
                        if 'HARVESTING' in fullcmd:
                            fullcmd += ' --filein file:step3_inDQM.root --fileout file:step4.root '
                        else:
                            if not '--filein' in fullcmd:
                                fullcmd += ' --filein file:step3.root '
                            if not 'fileout' in fullcmd:
                                fullcmd += '--fileout file:step.root '
                    fullcmd += ' > %s 2>&1; ' % ('step4_'+self.wf.nameId+'.log ',)
                    # print fullcmd
                    retStep4 = self.doCmd(fullcmd)
#                    if random.randint(0,100) < 40 : retStep4 = -42

        os.chdir(startDir)

        endtime='date %s' %time.asctime()
        tottime='%s-%s'%(endtime,startime)

        self.nfail = [0,0,0,0]
        self.npass = [1,1,1,1]
        if 'NOSTEP' in stat2: # don't say reco/alca is passed if we don't have to run them
            self.npass = [1,0,0,0]
        else: # we have a reco step, check for alca:
            if 'NOSTEP' in stat3 :
                self.npass = [1,1,0,0]
                if 'NOSTEP' in stat4 :
                    self.npass = [1,1,1,0]
        if retStep1 != 0 :
            stat1 = 'FAILED'
            stat2 = 'NOTRUN'
            stat3 = 'NOTRUN'
            stat4 = 'NOTRUN'
            self.npass = [0,0,0,0]
            self.nfail = [1,0,0,0]

        if retStep2 != 0 :
            stat2 = 'FAILED'
            stat3 = 'NOTRUN'
            stat4 = 'NOTRUN'
            self.npass = [1,0,0,0]
            self.nfail = [0,1,0,0]

        if retStep3 != 0 :
            stat3 = 'FAILED'
            stat4 = 'NOTRUN'
            self.npass = [1,1,0,0]
            self.nfail = [0,0,1,0]

        if retStep4 != 0 :
            stat4 = 'FAILED'
            self.npass = [1,1,1,0]
            self.nfail = [0,0,0,1]

        logStat = 'Step1-'+stat1+' Step2-'+stat2+' Step3-'+stat3+' '+' Step4-'+stat4+' ' 
        self.report+='%s_%s %s - time %s; exit: %s %s %s %s \n' % (self.wf.numId, self.wf.nameId, logStat, tottime, retStep1,retStep2,retStep3, retStep4)

        return


