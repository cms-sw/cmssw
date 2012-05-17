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
        if 'REALDATA' in self.wf.cmdStep1:
            realDataRe = re.compile('REALDATA:\s*(/[A-Za-z].*?),(\s*RUN:\s*(?P<run>\d+),)?(\s*FILES:\s*(?P<files>\d+),)?(\s*EVENTS:\s*(?P<events>\d+))?,\s*LABEL:\s*(?P<label>.*),\s*LOCATION:\s*(?P<location>.*)\s*')
            realDataMatch = realDataRe.match(self.wf.cmdStep1)
            if realDataMatch:
                run = None
                if realDataMatch.group("run") : run = realDataMatch.group("run")
                label  = realDataMatch.group("label")
                location = realDataMatch.group("location").lower().strip()
                if 'caf' in location:
                    print "ignoring workflow ",self.wf.numId, self.wf.nameId, ' as this is on CAF ...'
                    self.npass = [0,0,0,0]
                    self.nfail = [0,0,0,0]

                    logStat = 'Step1-NOTRUN Step2-NOTRUN Step3-NOTRUN Step4-NOTRUN ' 
                    self.report+='%s_%s %s - time %s; exit: %s %s %s %s \n' % (self.wf.numId, self.wf.nameId, logStat, 0, 0,0,0,0)
                    return
                
                files  = None
                events = None
                if realDataMatch.group("files"): 
                  files  = realDataMatch.group("files")
                if realDataMatch.group("events"): 
                  events = realDataMatch.group("events")
                  if self.wf.cmdStep2 and ' -n ' not in self.wf.cmdStep2: self.wf.cmdStep2 += ' -n ' + events
                  if self.wf.cmdStep3 and ' -n ' not in self.wf.cmdStep3: self.wf.cmdStep3 += ' -n ' + events
                  if self.wf.cmdStep4 and ' -n ' not in self.wf.cmdStep4: self.wf.cmdStep4 += ' -n ' + events

                print "run, files, events, label", run, files, events, label 
                cmd += 'dbs search --noheader --url=http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet '
                cmd += "--query='find file where dataset like "+realDataMatch.group(1)
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
                print "ERROR: found REALDATA in '"+self.wf.cmdStep1+"' but not RE match !!??!"
                retStep1 = -99
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
                inFile = '/store/relval/CMSSW_3_9_7/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V7HI-v1/0054/102FF831-9B0F-E011-A3E9-003048678BC6.root'
            if ( '41.0' in str(self.wf.numId) ) : 
                fullcmd += ' --himix '
                inFile = '/store/relval/CMSSW_3_9_7/RelValPyquen_GammaJet_pt20_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V7HI-v1/0054/06B4F699-A50F-E011-AD62-0018F3D0962E.root'

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
                    if 'HARVESTING' in fullcmd and not 'filein' in fullcmd:
                        fullcmd += ' --filein file:reco_inDQM.root --fileout file:step3.root '
                    else:
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

    def __init__(self, num, nameID, cmd1, cmd2=None, cmd3=None, cmd4=None, real=None):
        self.numId  = num.strip()
        self.nameId = nameID
        self.cmdStep1 = cmd1
        if self.cmdStep1: self.cmdStep1 = self.cmdStep1.replace('--no_exec', '') # make sure the commands execute
        self.cmdStep2 = cmd2
        if self.cmdStep2: self.cmdStep2 = self.cmdStep2.replace('--no_exec', '') # make sure the commands execute
        self.cmdStep3 = cmd3
        if self.cmdStep3: self.cmdStep3 = self.cmdStep3.replace('--no_exec', '') # make sure the commands execute
        self.cmdStep4 = cmd4
        if self.cmdStep4: self.cmdStep4 = self.cmdStep4.replace('--no_exec', '') # make sure the commands execute

        # run on real data requested:
        self.real = real
        return

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
        

        self.filesPrefMap = {'cmsDriver_standard_hlt.txt' : 'std-' ,
                             'cmsDriver_highstats_hlt.txt': 'hi-'  ,
                             'cmsDriver_generator.txt'    : 'gen-'  ,
                             # 'cmsDriver_PileUp_hlt.txt'   : 'pu-'  ,
                             # 'cmsDriver_realData_hlt.txt' : 'data-'
                             }

        self.files = ['cmsDriver_standard_hlt.txt' ,
                      'cmsDriver_highstats_hlt.txt',
                      'cmsDriver_generator.txt'    ,
                      # 'cmsDriver_PileUp_hlt.txt'   ,
                      # 'cmsDriver_realData_hlt.txt' 
                      ]
        
        return

    def readMatrix(self, fileNameIn):
        
        prefix = self.filesPrefMap[fileNameIn]

        print "processing ", fileNameIn,
        lines = []
        try:
            try:
                inFile = open(fileNameIn, 'r')
                print ' from local developer area',
            except IOError:
                baseRelPath = os.environ['CMSSW_BASE']
                # print "trying fall-back to cmsDriver files from developer area at:", baseRelPath
                try:
                    inFile = open( os.path.join(baseRelPath, 'src/Configuration/PyReleaseValidation/data' ,fileNameIn), 'r')
                    print ' from ', baseRelPath,
                except IOError:
                    baseRelPath = os.environ['CMSSW_RELEASE_BASE']
                    # print "trying fall back to cmsDriver files from base release at:", baseRelPath
                    inFile = open( os.path.join(baseRelPath, 'src/Configuration/PyReleaseValidation/data' ,fileNameIn), 'r')
                    print ' from ', baseRelPath,
            lines = inFile.readlines()
            inFile.close()
        except Exception, e:
            print "ERROR reading in file ", fileNameIn, str(e)
            return

        print " found ", len(lines), 'entries.'
        
        realRe = re.compile('\s*([1-9][0-9]*\.*[0-9]*)\s*\+\+\s*(.*?)\s*\+\+\s*(.*?)\s*\+\+\s*(.*?)\s*@@@\s*(.*)\s*')
        step1Re = re.compile('\s*([1-9][0-9]*\.*[0-9]*)\s*\+\+\s*(.*?)\s*\+\+\s*(.*?)\s*@@@\s*(.*)\s*')
        step2Re = re.compile('\s*STEP2\s*\+\+\s*(\S*)\s*@@@\s*(.*)\s*')
        step3Re = re.compile('\s*STEP3\s*\+\+\s*(\S*)\s*@@@\s*(.*)\s*')
        step4Re = re.compile('\s*STEP4\s*\+\+\s*(\S*)\s*@@@\s*(.*)\s*')
        for lineIn in lines:
            line = lineIn.strip()

            realMatch = realRe.match(line)
            if realMatch :
                num  = realMatch.group(1).strip()
                name = realMatch.group(2).strip().replace('<','').replace('>','').replace(':','')
                next = realMatch.group(3).strip().replace('+','').replace(',', ' ')
                cmd  = realMatch.group(4).strip()

                step2 = "None"
                step3 = "None"
                step4 = "None"

                steps = next.split()
                if len(steps) > 0:
                    step2 = steps[0].strip()
                if len(steps) > 1:
                    step3 = steps[1].strip()
                if len(steps) > 2:
                    step4 = steps[2].strip()
                
                self.step1WorkFlows[(float(num),prefix)] = (str(float(num)), name, step2, step3, step4, cmd, None)
                continue
            
                
            step1Match = step1Re.match(line)
            if step1Match :
                num  = step1Match.group(1).strip()
                name = step1Match.group(2).strip().replace('<','').replace('>','').replace(':','')
                next = step1Match.group(3).strip().replace('+','').replace(',', ' ')
                cmd  = step1Match.group(4).strip()
                step2 = "None"
                step3 = "None"
                step4 = "None"

                steps = next.split()
                if len(steps) > 0:
                    step2 = steps[0].strip()
                if len(steps) > 1:
                    step3 = steps[1].strip()
                if len(steps) > 2:
                    step4 = steps[2].strip()
                
                self.step1WorkFlows[(float(num),prefix)] = (str(float(num)), name, step2, step3, step4, cmd, None)
                continue
            
            step2Match = step2Re.match(line)
            if step2Match :
                name = step2Match.group(1).strip()
                cmd  = step2Match.group(2).strip()
                self.step2WorkFlows[name] = (cmd.replace('--no_exec','') ) # make sure the command is really run
                continue

            step3Match = step3Re.match(line)
            if step3Match :
                name = step3Match.group(1).strip()
                cmd  = step3Match.group(2).strip()
                self.step3WorkFlows[name] = ( cmd.replace('--no_exec','') ) # make sure the command is really run
                continue

            step4Match = step4Re.match(line)
            if step4Match :
                name = step4Match.group(1).strip()
                cmd  = step4Match.group(2).strip()
                self.step4WorkFlows[name] = ( cmd.replace('--no_exec','') ) # make sure the command is really run
                continue

        return

    def showRaw(self):

        print "found ", len(self.step1WorkFlows.keys()), ' workflows for step1:'
        ids = self.step1WorkFlows.keys()
        ids.sort()
        for key in ids:
            val = self.step1WorkFlows[key]
            print key[0], ':', val
        
        print "found ", len(self.step2WorkFlows.keys()), ' workflows for step2:'
        for key, val in self.step2WorkFlows.items():
            print key, ':', val
        
        print "found ", len(self.step3WorkFlows.keys()), ' workflows for step3:'
        for key, val in self.step3WorkFlows.items():
            print key, ':', val
        
        print "found ", len(self.step4WorkFlows.keys()), ' workflows for step4:'
        for key, val in self.step4WorkFlows.items():
            print key, ':', val
        
        return

    def showWorkFlows(self, selected=None):

        maxLen = 100 # for summary, limit width of output
        fmt1   = "%-6s %-35s [1]: %s ..."
        fmt2   = "       %35s [%d]: %s ..."
        print "\nfound a total of ", len(self.workFlows), ' workflows:'
        if selected:
            print "      of which the following", len(selected), 'were selected:'
            maxLen = -1  # for individual listing, no limit on width
            fmt1   = "%-6s %-35s [1]: %s " 
            fmt2   = "       %35s [%d]: %s"
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for wf in self.workFlows:
            if selected and float(wf.numId) not in selected: continue
            n1+=1
            print fmt1 % (wf.numId, wf.nameId, (wf.cmdStep1+' ')[:maxLen])
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
            num, name, step2, step3, step4, cmd, real = val
            nameId = num+'_'+name
            if step2.lower() != 'none':
                name += '+'+step2
                if step3.lower() != 'none':
                    name += '+'+step3
                    if step4.lower() != 'none':
                        name += '+'+step4
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

            if step2.lower() != 'none':
                n2 += 1
                cmd2 = self.step2WorkFlows[step2]
                if step3.lower() != 'none':
                    n3 += 1
                    cmd3 = self.step3WorkFlows[step3]
                    if step4.lower() != 'none':
                        n4 += 1
                        cmd4 = self.step4WorkFlows[step4]
                    #print '\tstep3 : ', self.step3WorkFlows[step3]
            self.workFlows.append( WorkFlow(num, name, cmd, cmd2, cmd3, cmd4) )

        return

    def prepare(self):
        
        for matrixFile in self.files:
            try:
                self.readMatrix(matrixFile)
            except Exception, e:
                print "ERROR reading file:", matrixFile, str(e)

            try:
                self.createWorkFlows(matrixFile)
            except Exception, e:
                print "ERROR creating workflows :", str(e)

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

            # don't know yet how to treat real data WF ...
            if wf.real : continue

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

def runSelected(testList, nThreads=4, show=False) :

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
    mrd.prepare()

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

def runData(testList, nThreads=4, show=False) :

    mrd = MatrixReader()

    mrd.prepare()

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

def runAll(testList=None, nThreads=4, show=False) :

    mrd = MatrixReader()
    mrd.prepare()

    ret = 0
    
    if show:
        mrd.show()
        print "nThreads = ",nThreads
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, nThreads)
        ret = mRunnerHi.runTests()

    return ret


# --------------------------------------------------------------------------------

def runOnly(only, show, nThreads=4):

    if not only: return
    
    for what in only:
        print "found request to run relvals only for ",what


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

<list>s should be put in single- or double-quotes to avoid confusion with/by the shell
"""

# ================================================================================

if __name__ == '__main__':

    import getopt
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hj:sl:nqo:d:", ['help',"nproc=",'selected','list=','showMatrix','only=','data='])
    except getopt.GetoptError, e:
        print "unknown option", str(e)
        sys.exit(2)
        
# check command line parameter

    np=4 # default: four threads
    sel = None
    show = False
    only = None
    data = None
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
        if opt in ('-d','--data',) :
            data = arg.split(',')

    # print "sel",sel
    ret = 0
    if sel != None: # explicit distinguish from empty list (which is also false)
        ret = runSelected(testList=sel, nThreads=np, show=show)
    elif only != None:
        ret = runOnly(only=only, show=show, nThreads=np)
    elif data != None:
        ret = runData(testList=data, show=show, nThreads=np)
    else:
        ret = runAll(show=show, nThreads=np)

    sys.exit(ret)
