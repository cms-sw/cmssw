#### A python framework for event, collection and object based analysis
#### whose source fits a single python file :-D

from math import *
from array import array
## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)

#### ========= EDM/FRAMEWORK =======================
class Event:
    def __init__(self,tree,entry):
        self._tree = tree
        self._entry = entry
        self._sync()
        self._isEval = False
    def _sync(self):
        if self._tree.entry != self._entry:
            self._tree.GetEntry(self._entry)
            self._tree.entry = self._entry
    def __getattr__(self,name):
        if name in self.__dict__: return self.__dict__[name]
        if name == "metLD": return self._tree.met*0.00397 + self._tree.mhtJet25*0.00265
        self._sync()
        if "(" in name:
            self._isEval = True
            ret = eval(name, globals(), self)
            self._isEval = False
            return ret
        if self._isEval:
            import math
            if hasattr(self._tree,name): return getattr(self._tree,name)
            if hasattr(math, name): return getattr(math,name)
            if hasattr(__builtins__,name): return getattr(__builtins__,name)
            return getattr(ROOT,name)
        return getattr(self._tree,name)
    def __getitem__(self,attr):
        return self.__getattr__(attr)
    def eval(self,expr):
        if not hasattr(self._tree, '_exprs'):
            self._tree._exprs = {}
            # remove useless warning about EvalInstance()
            import warnings
            warnings.filterwarnings(action='ignore', category=RuntimeWarning, 
                                    message='creating converter for unknown type "const char\*\*"$')
        if expr not in self._tree._exprs:
            formula = ROOT.TTreeFormula(expr,expr,self._tree)
            if formula.IsInteger():
                formula.go = formula.EvalInstance64
            else:
                formula.go = formula.EvalInstance
            self._tree._exprs[expr] = formula
            # force sync, to be safe
            self._tree.GetEntry(self._entry)
            self._tree.entry = self._entry
            #self._tree._exprs[expr].SetQuickLoad(False)
        else:
            self._sync()
            formula = self._tree._exprs[expr]
        if "[" in expr: # unclear why this is needed, but otherwise for some arrays x[i] == 0 for all i > 0
            formula.GetNdata()
        return formula.go()
            

class Object:
    def __init__(self,event,prefix,index=None):
        self._event = event
        self._prefix = prefix+"_"
        self._index = index
    def __getattr__(self,name):
        if name in self.__dict__: return self.__dict__[name]
        if name == "pdgLabel": return self.pdgLabel_()
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError
        try:
            val = getattr(self._event,self._prefix+name)
            if self._index != None:
                val = val[self._index]
        except AttributeError, e:
            import math
            if hasattr(math, name): val = getattr(math,name)
            elif hasattr(__builtins__,name): val = getattr(__builtins__,name)
            else:
                try:
                    val = getattr(ROOT,name)
                except AttributeError, e2:
                    raise e
        self.__dict__[name] = val ## cache
        return val
    def __getitem__(self,attr):
        return self.__getattr__(attr)
    def pdgLabel_(self):
        if self.pdgId == +13: return "#mu-";
        if self.pdgId == -13: return "#mu+";
        if self.pdgId == +11: return "e-";
        if self.pdgId == -11: return "e+";
    def p4(self):
        ret = ROOT.TLorentzVector()
        ret.SetPtEtaPhiM(self.pt,self.eta,self.phi,self.mass)
        return ret
    def subObj(self,prefix):
        return Object(self,self._event,self._prefix+prefix)
    def __repr__(self):
        return ("<%s[%s]>" % (self._prefix[:-1],self._index)) if self._index != None else ("<%s>" % self._prefix[:-1])
    def __str__(self):
        return self.__repr__()

class Collection:
    def __init__(self,event,prefix,len=None,maxlen=None,testVar="pt"):
        self._event = event
        self._prefix = prefix
        self._testVar = testVar
        self._vector = hasattr(event._tree, "vectorTree") and event._tree.vectorTree
        if len != None:
            self._len = getattr(event,len)
            if maxlen and self._len > maxlen: self._len = maxlen
        elif self._vector:
            self._len = getattr(event,"n"+prefix)
            if maxlen and self._len > maxlen: self._len = maxlen
        elif testVar != None:
            self._len = None
        else:
            raise RuntimeError, "must provide either len or testVar"
        self._cache = {}
    def __getitem__(self,index):
        if type(index) == int and index in self._cache: return self._cache[index]
        if self._testVar != None and self._len == None: self._countMe()
        if index >= self._len: raise IndexError, "Invalid index %r (len is %r) at %s" % (index,self._len,self._prefix)
        if self._vector:
            ret = Object(self._event,self._prefix,index=index)
        else: 
            ret = Object(self._event,"%s%d" % (self._prefix,index+1))
        if type(index) == int: self._cache[index] = ret
        return ret
    def __len__(self):
        if self._testVar != None and self._len == None: self._countMe()
        return self._len
    def _countMe(self):
        n = 0; ok = True
        while ok:
            try:
                val = getattr(self._event,"%s%d_%s" % (self._prefix,n+1,self._testVar))
                ok = (val > -98) 
                if ok: n += 1
            except:
                ok = False
        self._len = n

class Module:
    def __init__(self,name,booker=None):
        self._name = name
        self._booker = booker.mkdir(name) if booker != None else None
    def beginJob(self):
        pass
    def endJob(self):
        pass
    def analyze(self,event):
        pass
    def book(self,what,name,*args):
        return self._booker.book(what,name,*args)
    def beginComponent(self,component):
        """Used when running within tree2yields"""
        pass
    def endComponent(self,component):
        """Used when running within tree2yields"""
        pass

class EventLoop:
    def __init__(self,modules):
        self._modules = modules
    def loop(self,trees,maxEvents=-1,cut=None,eventRange=None):
        modules = self._modules
        for m in modules: m.beginJob()
        if type(trees) != list: trees = [ trees ]
        for tree in trees:
            tree.entry = -1
            for i in xrange(tree.GetEntries()) if eventRange == None else eventRange:
                if maxEvents > 0 and i >= maxEvents-1: break
                e = Event(tree,i)
                if cut != None and not e.eval(cut): 
                    continue
                ret = True
                for m in modules: 
                    ret = m.analyze(e)
                    if ret == False: break
                if i > 0 and i % 10000 == 0:
                    print "Processed %8d/%8d entries of this tree" % (i,tree.GetEntries())
        for m in modules: m.endJob()
    def beginComponent(self,component):
        for m in self._modules: m.beginComponent(component)
    def endComponent(self,component):
        for m in self._modules: m.endComponent(component)

#### ========= NTUPLING AND HISTOGRAMMING =======================
class PyTree:
    def __init__(self,tree):
        self.tree = tree
        self._branches = {} ## must be the last line
    def branch(self,name,type,n=1,lenVar=None):
        arr = array(type.lower(), n*[0 if type in 'iI' else 0.]) 
        self._branches[name] = arr
        if n == 1:
            self.tree.Branch(name, arr, name+"/"+type.upper())
        else:
            if lenVar != None:
                self.tree.Branch(name, arr, "%s[%s]/%s" % (name,lenVar,type.upper()))
            else:
                self.tree.Branch(name, arr, "%s[%d]/%s" % (name,n,type.upper()))
    def __setattr__(self,name,val):
        if hasattr(self,'_branches'):
            arr = self._branches[name]
            if len(arr) == 1:
                arr[0] = val
            else:
                for i,v in enumerate(val):
                    if i >= len(arr): break
                    arr[i]  = v
        else:
            self.__dict__[name] = val
    def fill(self):
        self.tree.Fill()

class BookDir:
    def __init__(self,tdir):
        self.tdir    = tdir
        self._objects = {}
        self._subs = []
    def mkdir(self,name):
        ret = BookDir(self.tdir.mkdir(name))
        ret.name = name
        self._subs.append(ret)
        return ret
    def book(self,what,name,*args):
        gdir = ROOT.gDirectory
        self.tdir.cd()
        obj = getattr(ROOT,what)(name,*args)
        self._objects[name] = obj
        gdir.cd()
        return obj
    def done(self):
        for s in self._subs: s.done()
        for k,v in self._objects.iteritems():
            self.tdir.WriteTObject(v)
    def printObj(self,on,o,dir):
        c1 = ROOT.TCanvas("c1","c1",800,600)
        o.Draw()
        for e in "png", "pdf":
            c1.Print("%s/%s.%s" % (dir, on, e))
    def printAll(self,dir):
        ROOT.gSystem.Exec("mkdir -p %s" % dir)
        for on,o in self._objects.iteritems():
            self.printObj(on,o,dir)
        for s in self._subs:
            s.printAll(dir+"/"+s.name)

class Booker(BookDir):
    def __init__(self,fileName):
        BookDir.__init__(self,ROOT.TFile(fileName,"RECREATE"))
    def done(self):
        BookDir.done(self)
        self.tdir.Close()

#### ========= UTILITIES =======================
def deltaPhi(phi1,phi2):
    ## Catch if being called with two objects
    if type(phi1) != float and type(phi1) != int:
        phi1 = phi1.phi
    if type(phi2) != float and type(phi2) != int:
        phi2 = phi2.phi
    ## Otherwise
    dphi = (phi1-phi2)
    while dphi >  pi: dphi -= 2*pi
    while dphi < -pi: dphi += 2*pi
    return dphi
def deltaR(eta1,phi1,eta2=None,phi2=None):
    ## catch if called with objects
    if eta2 == None:
        return deltaR(eta1.eta,eta1.phi,phi1.eta,phi1.phi)
    ## otherwise
    return hypot(eta1-eta2, deltaPhi(phi1,phi2))
def closest(object,list,presel=lambda x,y: True):
    ret = None; drMin = 999
    for x in list:
        if not presel(object,x): continue
        dr = deltaR(object,x)
        if dr < drMin: 
            ret = x; drMin = dr
    return (ret,drMin)


#### ========= TEST =======================
if __name__ == '__main__':
    class DummyModule(Module):
        def beginJob(self):
            self.maxEta = self.book("TH1F","maxEta","maxEta",20,0.,5.0)
            print "Booked histogram 'maxEta'"
        def analyze(self,event):
            genB = Collection(event,"LepGood") #,"nGenBQuarks",2) 
            print "Number of generated b quarks: %d" % len(genB)
            if not event.eval("nLepGood == 3"): return False
            for i in xrange(len(genB)):
                print "eta of gen b #%d: %+5.3f" % (i+1, genB[i].eta)
            print ""
            maxEta = max([abs(gb.eta) for gb in genB])
            self.maxEta.Fill(maxEta)
    from sys import argv
    f = ROOT.TFile(argv[1])
    if "ttHLepTreeProducerNew" in argv[1]: 
        t = f.Get("ttHLepTreeProducerNew")
        t.vectorTree = True
    else:
        t = f.Get("ttHLepTreeProducerBase")
        t.vectorTree = False
    booker = Booker("test.root")
    el = EventLoop([DummyModule("dummy",booker)])
    el.loop(t,1000)
    booker.done()
    print "Wrote to test.root"

