from TestClasses import *
from Reports import *
import os, shutil

class TestSuite:
    def __init__(self,dir,method,length,allTests):
        self._dir   = dir
        # fetch tests
        self._tests = []
        for m,l,test in allTests:
            if m != method and method != "*": continue
            if l == "full" and length != "full": continue
            self._tests.append(test)
    def listJobs(self):
        print "The following jobs will be considered: ";
        for t in self._tests: 
            print " - ",t.name()
    def createJobs(self): 
        self._createDir()
        for t in self._tests: 
            t.createScriptBase(self._dir)
    def runLocallySync(self):
        print "The following jobs will be run: "
        for t in self._tests: 
            print " - ",t.name(),"...",
            os.system(t.scriptName(self._dir))
            print " done."
    def runLocallyASync(self,threads=5):
        raise RuntimeError, "Not implemented"
    def runBatch(self,queue="8nh",command="bsub -q %{queue} %{script}"):
        for t in self._tests: 
            os.system(command % {'queue':queue,'script':t.scriptName(self._dir)})
    def report(self):
        import ROOT
        ROOT.gROOT.SetBatch(True)
        report = {}
        for t in self._tests:
            onerep = t.readOutputBase(self._dir)
            if onerep: report[t.name()] = onerep
        fout = open("%s/report.json" % self._dir, "w")
        fout.write(json.dumps(report))
        fout.close()
    def printIt(self,format):
        if os.access("%s/report.json" % self._dir, os.R_OK) == False:
            raise RuntimeError, "%s/report.json not found. please run 'report' before." % self._dir 
        obj = json.loads(''.join([f for f in open("%s/report.json" % self._dir)]))
        if format == "text": textReport(obj)
        else: RuntimeError, "Unknown format %s" % format
    def _selTests(self,method="*",length="full"):
        jobs = []
        for m,l,t in self._tests: 
            if m != method and method != "*": continue
            if l == "full" and length == "short": continue
            jobs.append(t)
        return jobs
    def _createDir(self, clear=False):
        "Prepare directoy"
        if self._dir[0] == "/": raise RuntimeError, "directory must be a relative path"
        if clear and os.access(self._dir, os.W_OK): shutil.rmtree(self._dir)
        if not os.access(self._dir, os.W_OK): os.mkdir(self._dir)
        if not os.access(self._dir, os.W_OK): RuntimeError, "Could not create dir %d" % self._dir
