
import os, sys, time
import random

from Configuration.PyReleaseValidation.WorkFlow import WorkFlow
from Configuration.PyReleaseValidation.WorkFlowRunner import WorkFlowRunner

# ================================================================================

class MatrixRunner(object):

    def __init__(self, wfIn=None, nThrMax=4):

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

    	report=''    	
        if self.maxThreads == 0:
            print 'resetting to default number of threads'
            self.maxThreads=4

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

