#!/usr/bin/env python

import os, sys, re, time

import random
from threading import Thread
        
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


# ================================================================================

class WorkFlow(object):

    def __init__(self, num, nameID, cmd1, cmd2=None, cmd3=None, cmd4=None, inputInfo=None):

        self.numId  = num.strip()
        self.nameId = nameID
        self.cmdStep1 = self.check(cmd1)
        self.cmdStep2 = self.check(cmd2)
        self.cmdStep3 = self.check(cmd3)
        self.cmdStep4 = self.check(cmd4)

        # run on real data requested:
        self.input = inputInfo
        return

    def check(self, cmd=None):
        if not cmd : return None

        # raw data are treated differently ...
        if 'DATAINPUT' in cmd: return cmd

        # force the number of events to process to be 10
        reN = re.compile('\s*-n\s*\d+\s*')
        newCmd = reN.sub(' -n 10 ', cmd)
        if not reN.match(newCmd) : # -n not specified, add it:
            newCmd += ' -n 10 '

        return newCmd

# ================================================================================

class MatrixReader(object):

    def __init__(self):

        self.reset()

        return

    def reset(self):

        self.step1WorkFlows = {}
        self.step2WorkFlows = {}
        self.step3WorkFlows = {}
        self.step4WorkFlows = {}

        self.workFlows = []
        self.nameList  = {}
        
        self.filesPrefMap = {'relval_standard' : 'std-' ,
                             'relval_highstats': 'hi-'  ,
                             'relval_generator': 'gen-'  ,
                             }

        self.files = ['relval_standard' ,
                      'relval_highstats',
                      'relval_generator',
                      ]

        self.relvalModule = None
        
        return

    def makeCmd(self, step):

        cmd = ''
        cfg = None
        input = None
        #print step
        #print defaults
        for k,v in step.items():
            if 'no_exec' in k : continue  # we want to really run it ...
            if k.lower() == 'cfg':
                cfg = v
                continue # do not append to cmd, return separately
            if k.lower() == 'input':
                input = v
                continue # do not append to cmd, return separately
            #print k,v
            cmd += ' ' + k + ' ' + str(v)
        return cfg, input, cmd
    
    def readMatrix(self, fileNameIn, useInput=None):
        
        prefix = self.filesPrefMap[fileNameIn]

        print "processing ", fileNameIn

        try:
            _tmpMod = __import__( 'Configuration.PyReleaseValidation.'+fileNameIn )
            self.relvalModule = sys.modules['Configuration.PyReleaseValidation.'+fileNameIn]
        except Exception, e:
            print "ERROR importing file ", fileNameIn, str(e)
            return

        print "request for INPUT for ", useInput

        for num, wfInfo in self.relvalModule.workflows.items():
            wfName = wfInfo[0]
            stepList = wfInfo[1]
            # if no explicit name given for the workflow, use the name of step1
            if wfName.strip() == '': wfName = stepList[0] 
            stepCmds = ['','','','']
            stepIndex = 0
            name  = wfName
            inputInfo = None
            for step in stepList:
                if len(name) > 0 : name += '+'
                stepName = step
                if stepIndex==0 and useInput and (str(num) in useInput or "all" in useInput):
                    # print "--> using INPUT as step1 for workflow ", num
                    if step+'INPUT' in self.relvalModule.step1.keys():
                        stepName = step+"INPUT"
                name += stepName
                cfg, input, opts = self.makeCmd(self.relvalModule.stepList[stepIndex][stepName])
                if input and cfg :
                    msg = "FATAL ERROR: found both cfg and input for workflow "+str(num)+' step '+stepName
                    raise msg

                if cfg:
                    cmd  = 'cmsDriver.py '+cfg+' '+opts
                if stepIndex==0 and not inputInfo and input: # only if we didn't already set the input
                    inputInfo = input
                    cmd = 'DATAINPUT from '+inputInfo.dataSet
                    
                if stepIndex > 0:
                    cmd  = 'cmsDriver.py step'+str(stepIndex+1)+'.py '+opts
                    
                stepCmds[stepIndex] = cmd
                stepIndex += 1

            self.step1WorkFlows[(float(num),prefix)] = (str(float(num)), name, stepCmds[0], stepCmds[1], stepCmds[2], stepCmds[3], inputInfo)
        
        return

    def showRaw(self, useInput):

        for matrixFile in self.files:
            self.reset()
            try:
                self.readMatrix(matrixFile, useInput)
            except Exception, e:
                print "ERROR reading file:", matrixFile, str(e)
                raise

            if not self.step1WorkFlows: continue

            dataFileName = matrixFile.replace('relval_', 'cmsDriver_')+'_hlt.txt'
            outFile = open(dataFileName,'w')

            print "found ", len(self.step1WorkFlows.keys()), ' workflows for ', dataFileName
            ids = self.step1WorkFlows.keys()
            ids.sort()
            stepCmds = ['','','','']
            for key in ids:
                num, name, stepCmds[0], stepCmds[1], stepCmds[2], stepCmds[3], inputInfo = self.step1WorkFlows[key]
                wfName,stepNames= name.split('+',1)
                otherSteps = None
                if '+' in stepNames:
                    step1,otherSteps = stepNames.split('+',1)
                line = num + ' ++ '+ wfName 
                if otherSteps:
                    line += ' ++ ' +otherSteps.replace('+',',')
                else:
                    line += ' ++ none' 
                if inputInfo :
                    line += ' ++ REALDATA: '+inputInfo.dataSet
                    line += ', FILES: ' +str(inputInfo.files)
                    line += ', EVENTS: '+str(inputInfo.events)
                    line += ', LABEL: ' +inputInfo.label
                    line += ', LOCATION:'+inputInfo.location
                    line += ' @@@'
                else:
                    line += ' @@@ '+stepCmds[0]
                print line
                outFile.write(line+'\n')

            outFile.write('\n'+'\n')
            for stepName in self.relvalModule.step2.keys():
                cfg,input,cmd = self.makeCmd(self.relvalModule.step2[stepName])
                line = 'STEP2 ++ ' +stepName + ' @@@ cmsDriver.py step2 ' +cmd
                print line
                outFile.write(line+'\n')
                
            outFile.write('\n'+'\n')
            for stepName in self.relvalModule.step3.keys():
                cfg,input,cmd = self.makeCmd(self.relvalModule.step3[stepName])
                line ='STEP3 ++ ' +stepName + ' @@@ cmsDriver.py step3_RELVAL ' +cmd 
                print line
                outFile.write(line+'\n')
                
            outFile.write('\n'+'\n')
            for stepName in self.relvalModule.step4.keys():
                cfg,input,cmd = self.makeCmd(self.relvalModule.step4[stepName])
                line = 'STEP4 ++ ' +stepName + ' @@@ cmsDriver.py step4 ' +cmd
                print line
                outFile.write(line+'\n')
                
            outFile.close()

        
        return

    def showWorkFlows(self, selected=None):

        maxLen = 100 # for summary, limit width of output
        fmt1   = "%-6s %-35s [1]: %s ..."
        fmt2   = "       %35s [%d]: %s ..."
        print "\nfound a total of ", len(self.workFlows), ' workflows:'
        if selected:
            print "      of which the following", len(selected), 'were selected:'
        #-ap for now:
        maxLen = -1  # for individual listing, no limit on width
        fmt1   = "%-6s %-35s [1]: %s " 
        fmt2   = "       %35s [%d]: %s"

        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for wf in self.workFlows:
            if selected and float(wf.numId) not in selected: continue
            print ''
            n1+=1
            wfName, stepNames = wf.nameId.split('+',1)
            print fmt1 % (wf.numId, stepNames, (wf.cmdStep1+' ')[:maxLen])
            if wf.cmdStep2:
                n2+=1
                print fmt2 % ( ' ', 2, (wf.cmdStep2+' ')[:maxLen])
                if wf.cmdStep3:
                    n3+=1
                    print fmt2 % ( ' ', 3, (wf.cmdStep3+' ')[:maxLen])
                    if wf.cmdStep4:
                        n4+=1
                        print fmt2 % ( ' ', 4, (wf.cmdStep4+' ')[:maxLen])

        print n1, 'workflows for step1,'
        print n2, 'workflows for step1 + step2,'
        print n3, 'workflows for step1 + step2 + step3'
        print n4, 'workflows for step1 + step2 + step3 + step4'

        return
    
    def createWorkFlows(self, fileNameIn):

        prefixIn = self.filesPrefMap[fileNameIn]

        # get through the list of items and update the requested workflows only
        keyList = self.step1WorkFlows.keys()
        ids = []
        for item in keyList:
            id, pref = item
            if pref != prefixIn : continue
            ids.append( float(id) )
            
        ids.sort()
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for key in ids:
            val = self.step1WorkFlows[(key,prefixIn)]
            num, name, cmd, step2, step3, step4, inputInfo = val
            nameId = num+'_'+name
            if nameId in self.nameList.keys():
                print "==> duplicate name found for ", nameId
                print '    keeping  : ', self.nameList[nameId]
                print '    ignoring : ', val
            else:
                self.nameList[nameId] = val

            cmd2 = None
            cmd3 = None
            cmd4 = None
            
            n1 += 1

            if step2.lower() != '':
                n2 += 1
                cmd2 = step2
                if step3.lower() != '':
                    n3 += 1
                    cmd3 = step3
                    if step4.lower() != '':
                        n4 += 1
                        cmd4 = step4
                    #print '\tstep3 : ', self.step3WorkFlows[step3]
            self.workFlows.append( WorkFlow(num, name, cmd, cmd2, cmd3, cmd4, inputInfo) )

        return

    def prepare(self, useInput=None):
        
        for matrixFile in self.files:
            try:
                self.readMatrix(matrixFile, useInput)
            except Exception, e:
                print "ERROR reading file:", matrixFile, str(e)
                raise

            try:
                self.createWorkFlows(matrixFile)
            except Exception, e:
                print "ERROR creating workflows :", str(e)
                raise
            
    def show(self, selected=None):    
        # self.showRaw()
        self.showWorkFlows(selected)
        print '\n','-'*80,'\n'


    def updateDB(self):

        import pickle
        pickle.dump(self.workFlows, open('theMatrix.pkl', 'w') )

        return

# ================================================================================

class MatrixRunner(object):

    def __init__(self, wfIn=None, nThrMax=8):

        self.workFlows = wfIn

        self.threadList = []
        self.maxThreads = int(nThrMax) # make sure we get a number ...


    def activeThreads(self):

        nActive = 0
        for t in self.threadList:
            if t.isAlive() : nActive += 1

        return nActive

        
    def runTests(self, testList=None):

        startDir = os.getcwd()

    	# make sure we have a way to set the environment in the threads ...
    	if not os.environ.has_key('CMS_PATH'):
    	    cmsPath = '/afs/cern.ch/cms'
    	    print "setting default for CMS_PATH to", cmsPath
    	    os.environ['CMS_PATH'] = cmsPath

    	report=''    	
    	print 'Running in %s thread(s)' % self.maxThreads
                
        for wf in self.workFlows:

            if testList and float(wf.numId) not in [float(x) for x in testList]: continue

            item = wf.nameId
            if os.path.islink(item) : continue # ignore symlinks
            
    	    # make sure we don't run more than the allowed number of threads:
    	    while self.activeThreads() >= self.maxThreads:
    	        time.sleep(10)
                continue
    	    
    	    print '\nPreparing to run %s %s' % (wf.numId, item)
          
##            if testList: # if we only run a selection, run only 5 events instead of 10
##                wf.cmdStep1 = wf.cmdStep1.replace('-n 10', '-n 5')
                
    	    current = WorkFlowRunner(wf)
    	    self.threadList.append(current)
    	    current.start()
            time.sleep(random.randint(1,5)) # try to avoid race cond by sleeping random amount of time [1,5] sec 

    	# wait until all threads are finished
        while self.activeThreads() > 0:
    	    time.sleep(5)
    	    
    	# all threads are done now, check status ...
    	nfail1 = 0
    	nfail2 = 0
        nfail3 = 0
        nfail4 = 0
    	npass  = 0
        npass1 = 0
        npass2 = 0
        npass3 = 0
        npass4 = 0
    	for pingle in self.threadList:
    	    pingle.join()
            try:
                nfail1 += pingle.nfail[0]
                nfail2 += pingle.nfail[1]
                nfail3 += pingle.nfail[2]
                nfail4 += pingle.nfail[3]
                npass1 += pingle.npass[0]
                npass2 += pingle.npass[1]
                npass3 += pingle.npass[2]
                npass4 += pingle.npass[3]
                npass  += npass1+npass2+npass3+npass4
                report += pingle.report
                # print pingle.report
            except Exception, e:
                msg = "ERROR retrieving info from thread: " + str(e)
                nfail1 += 1
                nfail2 += 1
                nfail3 += 1
                nfail4 += 1
                report += msg
                print msg
                
    	report+='\n %s %s %s %s tests passed, %s %s %s %s failed\n' %(npass1, npass2, npass3, npass4, nfail1, nfail2, nfail3, nfail4)
    	print report
    	
    	runall_report_name='runall-report-step123-.log'
    	runall_report=open(runall_report_name,'w')
    	runall_report.write(report)
    	runall_report.close()

        os.chdir(startDir)
    	
    	return

        
# ================================================================================

def showRaw(useInput=None) :

    mrd = MatrixReader()
    mrd.showRaw(useInput)

    return 0
        
# ================================================================================

def runSelected(testList, nThreads=4, show=False, useInput=None) :

    stdList = ['5.2', # SingleMu10 FastSim
               '7',   # Cosmics+RECOCOS+ALCACOS
               '8',   # BeamHalo+RECOCOS+ALCABH
               '25',  # TTbar+RECO2+ALCATT2  STARTUP
               ]
    hiStatList = [
                  '121',   # TTbar_Tauola
                  '123.3', # TTBar FastSim
                   ]

    mrd = MatrixReader()
    mrd.prepare(useInput)

    if testList == []:
        testList = stdList+hiStatList

    ret = 0
    if show:
        mrd.show([float(x) for x in testList])
        print 'selected items:', testList
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, nThreads)
        ret = mRunnerHi.runTests(testList)

    return ret

# ================================================================================

def runData(testList, nThreads=4, show=False, useInput=None) :

    mrd = MatrixReader()
    mrd.prepare(useInput)

    ret = 0
    if show:
        if not testList or testList == ['all']:
            mrd.show()
        else:
            mrd.show([float(x) for x in testList])
        print 'selected items:', testList
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, nThreads)
        if not testList or testList == ['all']:
            ret = mRunnerHi.runTests()
        else:
            ret = mRunnerHi.runTests(testList)

    return ret

# --------------------------------------------------------------------------------

def runAll(testList=None, nThreads=4, show=False, useInput=None) :

    mrd = MatrixReader()
    mrd.prepare(useInput)

    ret = 0
    
    if show:
        mrd.show()
        print "nThreads = ",nThreads
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, nThreads)
        ret = mRunnerHi.runTests()

    return ret


# --------------------------------------------------------------------------------

def runOnly(only, show, nThreads=4, useInput=None):

    if not only: return
    
    for what in only:
        print "found request to run relvals only for ",what
        print "not implemented, nothing done"

# --------------------------------------------------------------------------------

def usage():
    print "Usage:", sys.argv[0], ' [options] '
    print """
Where options is one of the following:
  -d, --data <list> comma-separated list of workflows to use from the realdata file.
                    <list> can be "all" to select all data workflows
  -l, --list <list> comma-separated list of workflows to use from the cmsDriver*.txt files
  -j, --nproc <n>   run <n> processes in parallel (default: 4 procs)
  -s, --selected    run a subset of 8 workflows (usually in the CustomIB)
  -n, -q, --show    show the (selected) workflows
  -i, --useInput <list>   will use data input (if defined) for the step1 instead of step1. <list> can be "all" for this option
  -r, --raw         in combination with --show will create the old style cmsDriver_*_hlt.txt file (in the working dir)
  
<list>s should be put in single- or double-quotes to avoid confusion with/by the shell
"""

# ================================================================================

if __name__ == '__main__':

    import getopt
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hj:sl:nqo:d:i:r", ['help',"nproc=",'selected','list=','showMatrix','only=','data=','useInput=','raw'])
    except getopt.GetoptError, e:
        print "unknown option", str(e)
        sys.exit(2)
        
# check command line parameter

    np=4 # default: four threads
    sel = None
    input = None
    show = False
    only = None
    data = None
    raw  = False
    for opt, arg in opts :
        if opt in ('-h','--help'):
            usage()
            sys.exit(0)
        if opt in ('-j', "--nproc" ):
            np=int(arg)
        if opt in ('-n','-q','--showMatrix', ):
            show = True
        if opt in ('-s','--selected',) :
            sel = []
        if opt in ('-o','--only',) :
            only = []
        if opt in ('-l','--list',) :
            sel = arg.split(',')
        if opt in ('-i','--useInput',) :
            input = arg.split(',')
        if opt in ('-d','--data',) :
            data = arg.split(',')
        if opt in ('-r','--raw') :
            raw = True
            
    if raw and show:
        ret = showRaw(useInput=input)
        sys.exit(ret)

        
    ret = 0
    if sel != None: # explicit distinguish from empty list (which is also false)
        ret = runSelected(testList=sel, nThreads=np, show=show, useInput=input)
    elif only != None:
        ret = runOnly(only=only, show=show, nThreads=np, useInput=input)
    elif data != None:
        ret = runData(testList=data, show=show, nThreads=np, useInput=input)
    else:
        ret = runAll(show=show, nThreads=np, useInput=input)

    sys.exit(ret)
