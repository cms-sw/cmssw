import os, ROOT
from array import array

class MVAVar:
    def __init__(self,name,type='f',func=None):
        self.name = name
        if func != None:
            self.func = func
        else:
            if ":=" in self.name:
                self.expr = self.name.split(":=")[1]
                self.func = lambda ev : ev.eval(self.expr)
            else:
                self.func = lambda ev : ev[self.name]
        self.type = type
        self.var  = array('f',[0.]) #TMVA wants ints as floats! #array(type,[0 if type == 'i' else 0.])
    def set(self,ev): 
        self.var[0] = self.func(ev)

class MVATool:
    def __init__(self,name,xml,vars,rarity=False,specs=[]):
        self.name = name
        self.reader = ROOT.TMVA.Reader("Silent")
        self.vars  = vars
        self.specs = specs
        for s in specs: self.reader.AddSpectator(s.name,s.var)
        for v in vars:  self.reader.AddVariable(v.name,v.var)
        #print "Would like to load %s from %s! " % (name,xml)
        self.reader.BookMVA(name,xml)
        self.rarity = rarity
    def __call__(self,ev): 
        for s in self.vars:  s.set(ev)
        for s in self.specs: s.set(ev)
        return self.reader.EvaluateMVA(self.name) if not self.rarity else self.reader.GetRarity(self.name)  

class CategorizedMVA:
    def __init__(self,catMvaPairs):
        self.catMvaPairs = catMvaPairs
    def __call__(self,lep):
        for c,m in self.catMvaPairs:
            if c(lep): return m(lep)
        return -99.



