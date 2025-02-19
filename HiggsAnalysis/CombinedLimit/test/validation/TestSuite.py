from TestClasses import *
from Reports import *
import os, shutil, sys, subprocess

_jobs_total   = 0
_jobs_started = 0
_jobs_done    = 0
def _async_run(test,dir):
    global _jobs_started, _jobs_total
    _jobs_started += 1
    print " - "+test.name()+(" started (%d/%d)..." % (_jobs_started, _jobs_total))
    subprocess.call([test.scriptName(dir)])
    return test
def _async_cb(test):
    global _jobs_done, _jobs_total
    _jobs_done += 1
    print " - "+test.name()+(" done (%d/%d)." % (_jobs_done, _jobs_total))

class TestSuite:
    def __init__(self,dir,options,allTests):
        self._dir   = dir
        # fetch tests
        self._tests = []
        dups = []
        for m,l,test in allTests:
            if m != options.method and options.method != "*": continue
            if l != "*"            and options.suite  != l:   continue
            if options.select  and not re.search(options.select,  test.name()): continue
            if options.exclude and     re.search(options.exclude, test.name()): continue
            if test.name() in dups: raise RuntimeError, "Duplicate test %s" % test.name()
            if options.nofork: test.forceSingleCPU()
            self._tests.append(test)
            dups.append(test.name())
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
            print " - ",t.name(),"...",; sys.stdout.flush()
            os.system(t.scriptName(self._dir))
            print " done."
    def runLocallyASync(self,threads):
        global _jobs_total
        print "Running jobs in parallel on %d cores" % threads
        from multiprocessing import Pool
        pool = Pool(processes=threads); 
        _jobs_total = 0
        for t in self._tests: 
            _jobs_total += 1;
            ret = pool.apply_async(_async_run, (t,self._dir), callback=_async_cb)
        pool.close()
        pool.join()
    def runBatch(self,queue="8nh",command="bsub -q %(queue)s %(script)s"):
        for t in self._tests: 
            cmd = command % {'queue':queue,'script':os.getcwd()+"/"+t.scriptName(self._dir)}
            if t.numCPUs() > 1 and "bsub " in command:
                cmd = cmd.replace("bsub ", "bsub -n %d -R 'span[hosts=1]' " % t.numCPUs())
            os.system(cmd)
    def report(self):
        import ROOT
        ROOT.gROOT.SetBatch(True)
        report = {}
        for t in self._tests:
            onerep = t.readOutputBase(self._dir)
            if onerep: report[t.name()] = onerep
        fout = open("%s/report.json" % self._dir, "w")
        fout.write(json.dumps(report, sort_keys=True, indent=4))
        fout.close()
    def printIt(self,format,reference=None):
        if os.access("%s/report.json" % self._dir, os.R_OK) == False:
            raise RuntimeError, "%s/report.json not found. please run 'report' before." % self._dir 
        obj = json.loads(''.join([f for f in open("%s/report.json" % self._dir)]))
        if reference:
            if not os.access(reference, os.R_OK): raise RuntimeError, "Reference % not found." % reference 
            ref = json.loads(''.join([f for f in open(reference)]))
            self._collate(obj,ref);
        if format == "text": textReport(obj)
        if format == "twiki": twikiReport(obj)
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
    def _collate(self,report,reference):
        for tn,tv in report.items():
            if not reference.has_key(tn): continue
            if not tv.has_key('results'): continue
            if tv['status'] == 'aborted': continue
            tv_ref = reference[tn]
            tv['ref_comment'] = tv_ref['comment']
            if not tv_ref.has_key('results'): continue
            for name, res in tv['results'].items():
                if res['status'] != 'done':  continue
                if not tv_ref['results'].has_key(name): continue
                ref = tv_ref['results'][name]
                if not ref.has_key('limit'): 
                    res['ref'] = { 'comment': ref['comment'] }
                    continue
                (limit, limitErr, time)  = res['limit'], res['limitErr'], res['t_real']
                (limitR,limitErrR,timeR) = ref['limit'], ref['limitErr'], ref['t_real']
                deltaLimRel = abs(limit-limitR)/max(hypot(limitErr,limitErrR),0.01*(limit+limitR),0.005)
                if   deltaLimRel <= 2: res['status'] = 'ok'
                elif deltaLimRel <= 5: res['status'] = 'warning'
                else:                  res['status'] = 'error'
                if res['status'] == 'ok':
                    if limitErr > 0 and limitErrR > 0:
                        if limitErr/limitErrR >= 1.5 and ref['comment'] == '': 
                            res['status']  = 'w unc.'
                            res['comment'] += 'worse uncertainty'
                    if time > 0.5 and timeR > 0.5 and ref['comment'] == '':
                        if time/timeR >= 2: 
                            res['status'] = 'w time'
                            res['comment'] += 'worse timing'
                res['ref'] = { 'limit':limitR, 'limitErr':limitErrR, 't_real':timeR, 'comment':ref['comment'] }
                tv['has_ref'] = True
            errors   = sum([res['status'] == 'error'   for res in tv['results'].values()])
            warnings = sum([res['status'] in ['warning', 'w unc.', 'w time'] for res in tv['results'].values()])
            aborts   = sum([res['status'] == 'aborted' for res in tv['results'].values()])
            if (len(tv['results']) == 1):
                rv = tv['results'].values()[0]
                tv['status'] = rv['status']
                if tv.has_key('has_ref'): report['has_ref'] = True
            else:
                if errors > 0:
                    tv['status'] = 'error'; tv['comment'] = '%d errors, %d warnings' % (errors, warnings)
                    report['has_ref'] = True
                elif warnings > 0:
                    tv['status'] = 'warning'; tv['comment'] = '%d warnings' % warnings
                    report['has_ref'] = True
                elif aborts > 0:
                    tv['status'] = 'mixed'
                elif tv.has_key('has_ref'):
                    tv['status'] = 'ok'
                    report['has_ref'] = True

