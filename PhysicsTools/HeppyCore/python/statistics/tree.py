import numpy
from ROOT import TTree
import ROOT

class Tree(object):
    
    def __init__(self, name, title, defaultFloatType="D", defaultIntType="I"):
        self.vars = {}
        self.vecvars = {}
        self.tree = TTree(name, title)
        self.defaults = {}
        self.vecdefaults = {}
        self.defaultFloatType = defaultFloatType
        self.defaultIntType = defaultIntType
        self.fillers = {}

    def setDefaultFloatType(self, defaultFloatType):
        self.defaultFloatType = defaultFloatType

    def setDefaultIntType(self, defaultIntType):
        self.defaultIntType = defaultIntType
        
    def copyStructure(self, tree):
        for branch in tree.GetListOfBranches():
            name = branch.GetName() 
            typeName = branch.GetListOfLeaves()[0].GetTypeName()
            type = float
            if typeName == 'Int_t':
                type = int
            self.var(name, type)            
    
    def branch_(self, selfmap, varName, type, len, postfix="", storageType="default", title=None):
        """Backend function used to create scalar and vector branches. 
           Users should call "var" and "vector", not this function directly."""
        if storageType == "default": 
            storageType = self.defaultIntType if type is int else self.defaultFloatType
        if type is float  :
            if storageType == "F": 
                selfmap[varName]=numpy.zeros(len,numpy.float32)
                self.tree.Branch(varName,selfmap[varName],varName+postfix+'/F')
            elif storageType == "D":
                selfmap[varName]=numpy.zeros(len,numpy.float64)
                self.tree.Branch(varName,selfmap[varName],varName+postfix+'/D')
            else:
                raise RuntimeError, 'Unknown storage type %s for branch %s' % (storageType, varName)
        elif type is int: 
            dtypes = {
                "i" : numpy.uint32,
                "s" : numpy.uint16,
                "b" : numpy.uint8,
                "l" : numpy.uint64,
                "I" : numpy.int32,
                "S" : numpy.int16,
                "B" : numpy.int8,
                "L" : numpy.int64,
            }
            if storageType not in dtypes: 
                raise RuntimeError, 'Unknown storage type %s for branch %s' % (storageType, varName)
            selfmap[varName]=numpy.zeros(len,dtypes[storageType])
            self.tree.Branch(varName,selfmap[varName],varName+postfix+'/'+storageType)
        else:
            raise RuntimeError, 'Unknown type %s for branch %s' % (type, varName)
        if title:
            self.tree.GetBranch(varName).SetTitle(title)

    def var(self, varName,type=float, default=-99, title=None, storageType="default", filler=None ):
        if type in [int, float]:
            self.branch_(self.vars, varName, type, 1, title=title, storageType=storageType)
            self.defaults[varName] = default
        elif __builtins__['type'](type) == str:
            # create a value, looking up the type from ROOT and calling the default constructor
            self.vars[varName] = getattr(ROOT,type)()
            if type in [ "TLorentzVector" ]: # custom streamer classes
                self.tree.Branch(varName+".", type, self.vars[varName], 8000,-1)
            else:
                self.tree.Branch(varName+".", type, self.vars[varName])
            if filler is None:
                raise RuntimeError, "Error: when brancing with an object, filler should be set to a function that takes as argument an object instance and a value, and set the instance to the value (as otherwise python assignment of objects changes the address as well)"
            self.fillers[varName] = filler
        else:
            raise RuntimeError, 'Unknown type %s for branch %s: it is not int, float or a string' % (type, varName)
        self.defaults[varName] = default

    def vector(self, varName, lenvar, maxlen=None, type=float, default=-99, title=None, storageType="default", filler=None ):
        """either lenvar is a string, and maxlen an int (variable size array), or lenvar is an int and maxlen is not specified (fixed array)"""
        if type in [int, float]:
            if __builtins__['type'](lenvar) == int:  # need the __builtins__ since 'type' is a variable here :-/
                self.branch_(self.vecvars, varName, type, lenvar, postfix="[%d]" % lenvar, title=title, storageType=storageType)
            else:
                if maxlen == None: RuntimeError, 'You must specify a maxlen if making a dynamic array';
                self.branch_(self.vecvars, varName, type, maxlen, postfix="[%s]" % lenvar, title=title, storageType=storageType)
        elif __builtins__['type'](type) == str:
            self.vecvars[varName] = ROOT.TClonesArray(type,(lenvar if __builtins__['type'](lenvar) == int else maxlen))
            if type in [ "TLorentzVector" ]: # custom streamer classes
                self.tree.Branch(varName+".", self.vecvars[varName], 32000, -1)
            else:
                self.tree.Branch(varName+".", self.vecvars[varName])
            if filler is None:
                raise RuntimeError, "Error: when brancing with an object, filler should be set to a function that takes as argument an object instance and a value, and set the instance to the value (as otherwise python assignment of objects changes the address as well)"
            self.fillers[varName] = filler
        self.vecdefaults[varName] = default

    def reset(self):
        for name,value in self.vars.iteritems():
            if name in self.fillers:
                self.fillers[name](value, self.defaults[name])
            else:
                value[0]=self.defaults[name]
        for name,value in self.vecvars.iteritems():
            if isinstance(value, numpy.ndarray):
                value.fill(self.vecdefaults[name])
            else:
                if isinstance(value, ROOT.TObject) and value.ClassName() == "TClonesArray":
                    value.ExpandCreateFast(0)
            
    def fill(self, varName, value ):
        if isinstance(self.vars[varName], numpy.ndarray):
            self.vars[varName][0]=value
        else:
            self.fillers[varName](self.vars[varName],value)

    def vfill(self, varName, values ):
        a = self.vecvars[varName]
        if isinstance(a, numpy.ndarray):
            for (i,v) in enumerate(values):
                a[i]=v
        else:
            if isinstance(a, ROOT.TObject) and a.ClassName() == "TClonesArray":
                a.ExpandCreateFast(len(values))
            fillit = self.fillers[varName]
            for (i,v) in enumerate(values):
                fillit(a[i],v)
