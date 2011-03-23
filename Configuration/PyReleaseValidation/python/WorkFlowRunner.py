
from threading import Thread

from Configuration.PyReleaseValidation import WorkFlow

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
            ret = os.system(cmd)
            if ret != 0:
                print "ERROR executing ",cmd,'ret=', ret

        return ret
    
    def run(self):

        startDir = os.getcwd()

        wfDir = self.wf.numId+'_'+self.wf.nameId
        if not os.path.exists(wfDir):
            os.makedirs(wfDir)

        preamble = ''
        if os.path.exists( os.path.join(os.environ["CMS_PATH"],'cmsset_default.sh') ) :
            preamble = 'source $CMS_PATH/cmsset_default.sh; '
        else:
            preamble = 'source $CMS_PATH/sw/cmsset_default.sh; '
        preamble += 'eval `scram run -sh`; '
        preamble += 'cd '+wfDir+'; '
        preamble += 'ulimit -v 4069000;' # make sure processes keep within limits ...
        
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

        inFile = 'file:raw.root'
        if self.wf.cmdStep1.startswith('DATAINPUT'):
            print "going to run with file input ... "
            if self.wf.input.run:
                run      = str(self.wf.input.run)
            else:
                run=None

            label    = self.wf.input.label
            location = self.wf.input.location.lower().strip()
            if 'caf' in location:
                print "ignoring workflow ",self.wf.numId, self.wf.nameId, ' as this is on CAF ...'
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
            cmd += ' > %s 2>&1; ' % ('step1_'+self.wf.nameId+'-dbsquery.log',)
            retStep1 = self.doCmd(cmd)
            if retStep1 == 0:
                lf = open(wfDir+'/step1_'+self.wf.nameId+'-dbsquery.log', 'r')
                lines = lf.readlines()
                lf.close()
                if not lines or len(lines)==0 :
                    inFile = "NoFileFoundInDBS"
                    retStep1 = -95
                else:
                    try:
                        inFile = lines[0].strip()
                    except Exception, e:
                        print "ERROR determining file from DBS query: ", str(e)
                        inFile = "NoFileFoundInDBS"
                        retStep1 = -90
        else:
            cmd += self.wf.cmdStep1 + ' --fileout file:raw.root '
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
            fullcmd += ' --fileout file:reco.root '
            print '=====>>> ', self.wf.nameId, self.wf.numId

            # for HI B0 step2 use a file from a previous relval production as step1 doesn't write
            # any output in 1 hour. Add himix flag here as this is only needed when run on the mixed
            # input files (the relvals are OK)
            if ( '40.0' in str(self.wf.numId) ) :
                fullcmd += ' --himix '
                inFile = '/store/relval/CMSSW_3_8_0_pre1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-RAW/MC_37Y_V5-v1/0001/E0DE7C01-2C6F-DF11-B61F-0026189438F4.root'
            if ( '41.0' in str(self.wf.numId) ) : 
                fullcmd += ' --himix '
                inFile = '/store/relval/CMSSW_3_8_0_pre1/RelValPyquen_GammaJet_pt20_2760GeV/GEN-SIM-RAW/MC_37Y_V5-v1/0001/F68A53A5-2B6F-DF11-8958-003048678FE6.root'

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
                    fullcmd += ' --filein file:reco.root --fileout file:step3.root '
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
                        fullcmd += ' --filein file:step3.root '
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


