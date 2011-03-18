import os, os.path, re, json, stat
import ROOT

class Test:
    def __init__(name):
        self._name = name
    def name(self):
        return self._name
    def scriptName(self,dir):
        return "%s/run_%s.sh" % (dir,self._name)
    def logName(self,dir):
        return "%s/%s.log" % (dir,self._name)
    def jsonName(self,dir):
        return "%s/%s.json" % (dir,self._name)
    def createScriptBase(self,dir):
        """Creates script file, writes preamble, then calls createScript to fill it"""
        print "Will create script ",self.scriptName(dir)
        fname = self.scriptName(dir)
        output = open(fname, "w")
        self._createPreamble(output,dir)
        self.createScript(dir,output)
        output.close()
        os.chmod(fname,stat.S_IRWXU)
    def createScript(self,dir,file):
        """Print out whatever commands have to be printed in the script after it's created"""
        raise RuntimeError, "Not implemented"
    def readOutputBase(self,dir):
        """Calls readOutput to read the output, saves it as JSON file"""
        output = self.readOutput(dir)
        if output != None:
            fout = open(self.jsonName(dir), "w");
            fout.write(json.dumps({self._name : output}))
            fout.close()
        return output
    def readOutput(self,dir):
        """Read output, and return output as python object"""
        raise RuntimeError, "Not implemented"
    def _createPreamble(self,file,dir):
        E=os.environ;
        file.write("#!/bin/bash\n")
        file.write("cd %s/%s;\n" % (os.getcwd(), dir));
        file.write('if [[ "$CMSSW_BASE" != "%s" ]]; then\n'
                   "     export SCRAM_ARCH=%s;          \n"
                   "     eval $(scram runtime -sh);     \n"
                   "fi\n" % (E['CMSSW_BASE'], E['SCRAM_ARCH']));
        file.write(": > %s.log\n" % self._name);
        

class SimpleDatacardTest(Test):
    def __init__(self,datacard,method,options,postfix="",mass=120):
        self._name = re.sub(r"\.(txt|hlf|root)$", "", os.path.basename(datacard))+"_"+method+postfix
        self._datacard = datacard
        self._method   = method
        self._options  = options
        self._mass     = mass
    def createScript(self,dir,file):
        datacard_full = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/benchmarks/%s" % self._datacard
        if os.access(datacard_full, os.R_OK) == False: 
            raise RuntimeError, "Datacard HiggsAnalysis/CombinedLimit/data/benchmarks/%s is not accessible" % self._datacard
        file.write("combine %s -n %s -M %s %s -m %s > %s.log 2>&1\n" % (datacard_full, self._name, self._method, self._options, self._mass, self._name));
    def readOutput(self,dir):
        fname = "%s/higgsCombine%s.%s.mH%s.root" % (dir, self._name, self._method, self._mass)
        if os.access(fname, os.R_OK) == False: return { 'status':'aborted', 'comment':'rootfile does not exist' }
        if os.stat(fname).st_size < 1000:   return { 'status':'aborted', 'comment':'rootfile is smaller than 1kb' }
        f = ROOT.TFile.Open(fname)
        if not f: return { 'status':'aborted', 'comment':'rootfile is not readable by ROOT'}
        t = f.Get("limit")
        if not t: 
            f.Close()
            return { 'status':'aborted', 'comment':'rootfile does not contain the tree' }
        result = {}
        for c in [ "limit", "limitErr", "t_real" ]:  result[c] = t.GetMaximum(c)
        f.Close()
        return { 'status':'done', 'comment':'', 'results':{'':result} }

class MultiDatacardTest(Test):
    def __init__(self,name,datacards,method,options):
        self._name = name 
        self._datacards = [ (dc,i*10+100) for i,dc in enumerate(datacards) ]
        self._method   = method
        self._options  = options
    def createScript(self,dir,file):
        for dc, mass in self._datacards:
            datacard_full = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/benchmarks/%s" % dc 
            if os.access(datacard_full, os.R_OK) == False: 
                raise RuntimeError, "Datacard HiggsAnalysis/CombinedLimit/data/benchmarks/%s is not accessible" % dc 
            file.write("echo    %s -n %s -M %s %s -m %s >> %s.log     \n" % (datacard_full, self._name, self._method, self._options, mass, self._name));
            file.write("combine %s -n %s -M %s %s -m %s >> %s.log 2>&1\n" % (datacard_full, self._name, self._method, self._options, mass, self._name));
    def readOutput(self,dir):
        done = 0; aborted = 0
        ret = { 'results':{}}
        for dc, mass in self._datacards:
            thisres = self.readOneOutput(dir,mass)
            ret['results'][os.path.basename(dc)] = thisres
            if thisres['status'] == 'done': done += 1
            if thisres['status'] == 'aborted': aborted += 1
        if aborted == 0 and done > 0:
            ret['status'] = 'done'; 
            ret['comment'] = 'all %d jobs done' % done; 
        elif done == 0 and aborted > 0:
            ret['status'] = 'aborted'; 
            ret['comment'] = 'all %d jobs aborted' % aborted; 
        else:
            ret['status'] = 'partial'
            ret['comment'] = '%d jobs done, %d jobs aborted' % (done, aborted)
        return ret
    def readOneOutput(self,dir,mass):
        fname = "%s/higgsCombine%s.%s.mH%s.root" % (dir, self._name, self._method, mass)
        if os.access(fname, os.R_OK) == False: return { 'status':'aborted', 'comment':'rootfile does not exist' }
        if os.stat(fname).st_size < 1000:   return { 'status':'aborted', 'comment':'rootfile is smaller than 1kb' }
        f = ROOT.TFile.Open(fname)
        if not f: return { 'status':'aborted', 'comment':'rootfile is not readable by ROOT'}
        t = f.Get("limit")
        if not t: 
            f.Close()
            return { 'status':'aborted', 'comment':'rootfile does not contain the tree' }
        result = { 'status':'done', 'comment':'' }
        for c in [ "limit", "limitErr", "t_real" ]:  result[c] = t.GetMaximum(c)
        return result

def datacardGlob(pattern):
    from glob import iglob
    base = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/benchmarks/"
    return [ f.replace(base, "") for f in iglob(base+pattern) ]

class MultiOptionTest(Test):
    def __init__(self,name,datacard,method,commonOptions,optionsMap):
        self._name = name 
        self._datacard = datacard 
        self._method   = method
        self._options  = [ (on,commonOptions+" "+ov,i*10+100) for i,(on,ov) in enumerate(optionsMap.items()) ]
    def createScript(self,dir,file):
        datacard_full = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/benchmarks/%s" % self._datacard
        if os.access(datacard_full, os.R_OK) == False: 
            raise RuntimeError, "Datacard HiggsAnalysis/CombinedLimit/data/benchmarks/%s is not accessible" % self._datacard
        for on, ov, mass in self._options:
            file.write("echo    %s -n %s -M %s %s -m %s >> %s.log     \n" % (datacard_full, self._name, self._method, ov, mass, self._name));
            file.write("combine %s -n %s -M %s %s -m %s >> %s.log 2>&1\n" % (datacard_full, self._name, self._method, ov, mass, self._name));
    def readOutput(self,dir):
        done = 0; aborted = 0
        ret = { 'results':{}}
        for on, ov, mass in self._options:
            thisres = self.readOneOutput(dir,mass)
            ret['results'][on] = thisres
            if thisres['status'] == 'done': done += 1
            if thisres['status'] == 'aborted': aborted += 1
        if aborted == 0 and done > 0:
            ret['status'] = 'done'; 
            ret['comment'] = 'all %d jobs done' % done; 
        elif done == 0 and aborted > 0:
            ret['status'] = 'aborted'; 
            ret['comment'] = 'all %d jobs aborted' % aborted; 
        else:
            ret['status'] = 'partial'
            ret['comment'] = '%d jobs done, %d jobs aborted' % (done, aborted)
        return ret
    def readOneOutput(self,dir,mass):
        fname = "%s/higgsCombine%s.%s.mH%s.root" % (dir, self._name, self._method, mass)
        if os.access(fname, os.R_OK) == False: return { 'status':'aborted', 'comment':'rootfile does not exist' }
        if os.stat(fname).st_size < 1000:   return { 'status':'aborted', 'comment':'rootfile is smaller than 1kb' }
        f = ROOT.TFile.Open(fname)
        if not f: return { 'status':'aborted', 'comment':'rootfile is not readable by ROOT'}
        t = f.Get("limit")
        if not t: 
            f.Close()
            return { 'status':'aborted', 'comment':'rootfile does not contain the tree' }
        result = { 'status':'done', 'comment':'' }
        for c in [ "limit", "limitErr", "t_real" ]:  result[c] = t.GetMaximum(c)
        return result


