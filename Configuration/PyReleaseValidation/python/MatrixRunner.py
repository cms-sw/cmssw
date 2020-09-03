from __future__ import print_function
import os, sys, time

from Configuration.PyReleaseValidation.WorkFlow import WorkFlow
from Configuration.PyReleaseValidation.WorkFlowRunner import WorkFlowRunner

# ================================================================================

class MatrixRunner(object):

    def __init__(self, wfIn=None, nThrMax=4, nThreads=1):

        self.workFlows = wfIn

        self.threadList = []
        self.maxThreads = nThrMax
        self.nThreads = nThreads

        #the directories in which it happened
        self.runDirs={}

    def activeThreads(self):

        nActive = 0
        for t in self.threadList:
            if t.isAlive() : nActive += 1

        return nActive


    def runTests(self, opt):

        testList=opt.testList
        dryRun=opt.dryRun
        cafVeto=opt.cafVeto

        startDir = os.getcwd()

        report=''
        noRun=(self.maxThreads==0)
        if noRun:
            print('Not running the wf, only creating cfgs and logs')
            self.maxThreads=4
            print('resetting to default number of process threads = %s' %  self.maxThreads)

        print('Running %s %s %s, each with %s thread%s per process' % ('up to' if self.maxThreads > 1 else '', self.maxThreads, 'concurrent jobs' if self.maxThreads > 1 else 'job', self.nThreads, 's' if self.nThreads > 1 else ''))


        for wf in self.workFlows:

            if testList and float(wf.numId) not in [float(x) for x in testList]: continue

            item = wf.nameId
            if os.path.islink(item) : continue # ignore symlinks

            # make sure we don't run more than the allowed number of threads:
            while self.activeThreads() >= self.maxThreads:
                time.sleep(1)

            print('\nPreparing to run %s %s' % (wf.numId, item))
            sys.stdout.flush()
            current = WorkFlowRunner(wf,noRun,dryRun,cafVeto, opt.dasOptions, opt.jobReports, opt.nThreads, opt.nStreams, opt.maxSteps)
            self.threadList.append(current)
            current.start()
            if not dryRun:
                time.sleep(0.5) # try to avoid race cond by sleeping 0.5 sec

        # wait until all threads are finished
        while self.activeThreads() > 0:
            time.sleep(0.5)


        #wrap up !
        totpassed=[]
        totfailed=[]
        def count(collect,result):
            #pad with zeros
            for i in range(len(collect),len(result)):
                collect.append(0)
            for i,c in enumerate(result):
                collect[i]+=c

        for pingle in self.threadList:
            pingle.join()
            try:
                count(totpassed,pingle.npass)
                count(totfailed,pingle.nfail)
                report+=pingle.report
                self.runDirs[pingle.wf.numId]=pingle.wfDir
            except Exception as e:
                msg = "ERROR retrieving info from thread: " + str(e)
                report += msg

        report+=' '.join(map(str,totpassed))+' tests passed, '+' '.join(map(str,totfailed))+' failed\n'
        print(report)
        sys.stdout.flush()

        runall_report_name='runall-report-step123-.log'
        runall_report=open(runall_report_name,'w')
        runall_report.write(report)
        runall_report.close()
        os.chdir(startDir)

        anyFail=sum(totfailed)

        return anyFail
