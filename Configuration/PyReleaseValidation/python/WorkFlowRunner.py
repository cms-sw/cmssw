
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

        # check where we are running:
        onCAF = False
        if 'cms/caf/cms' in os.environ['CMS_PATH']:
            onCAF = True

        inFile = None
        if self.wf.cmdStep1.startswith('DATAINPUT'):
            print "going to run with file input ... "
            if self.wf.input.run:
                run      = str(self.wf.input.run)
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

            # for HI B0 step2 use a file from a previous relval production as step1 doesn't write
            # any output in 1 hour. Add himix flag here as this is only needed when run on the mixed
            # input files (the relvals are OK)
            # useInput in the IB
            #if ( '40.0' in str(self.wf.numId) ) :
            #    #nono fullcmd += ' --himix '
            #    #nono inFile = '/store/relval/CMSSW_3_9_7/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V7HI-v1/0054/102FF831-9B0F-E011-A3E9-003048678BC6.root'
            #    #nono fullcmd += ' --process HIMIX '
            #    inFile = '/store/relval/CMSSW_4_4_0_pre5/RelValHydjetQ_MinBias_2760GeV/GEN-SIM/STARTHI44_V1-v1/0018/34C7FA16-59B2-E011-BC88-002618943843.root'
            # just taken out
            #if ( '41.0' in str(self.wf.numId) ) : 
            #    fullcmd += ' --himix '
            #    inFile = '/store/relval/CMSSW_3_9_7/RelValPyquen_GammaJet_pt20_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V7HI-v1/0054/06B4F699-A50F-E011-AD62-0018F3D0962E.root'
            #    fullcmd += ' --process HIMIX '
                
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
                # this trick is not necessary anymore
                #if ( '40.0' in str(self.wf.numId) or '41.0' in str(self.wf.numId) ) :
                #    fullcmd += '--hltProcess=HIMIX'
                    
                fullcmd += ' > %s 2>&1; ' % ('step3_'+self.wf.nameId+'.log ',)
                # print fullcmd
                retStep3 = self.doCmd(fullcmd)
#                if random.randint(0,100) < 40 : retStep3 = -42
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


