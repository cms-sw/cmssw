# Copyright (C) 2014 Giovanni Petrucciani 
# https://github.com/cbernet/heppy/blob/master/LICENSE

import numpy
from ROOT import TTree

class Tree(object):
    """ROOT tree adaptor, based on numpy arrays
    #TODO do I have a better version of this code somewhere?
    """

    def __init__(self, name, title, defaultFloatType="D", defaultIntType="I"):
        self.vars = {}
        self.vecvars = {}
        self.tree = TTree(name, title)
        self.defaults = {}
        self.vecdefaults = {}
        self.defaultFloatType = defaultFloatType
        self.defaultIntType = defaultIntType

    def setDefaultFloatType(self, defaultFloatType):
        self.defaultFloatType = defaultFloatType

    def setDefaultIntType(self, defaultFloatType):
        self.defaultIntType = defaultIntType

    def copyStructure(self, tree):
        for branch in tree.GetListOfBranches():
            name = branch.GetName()
            typeName = branch.GetListOfLeaves()[0].GetTypeName()
            type = float
            if typeName == 'Int_t':
                type = int
            self.var(name, type)

    def branch_(self, selfmap, varName, type, len, postfix="",
                storageType="default", title=None):
        """Backend function used to create scalar and vector branches.
           Users should call "var" and "vector", not this function directly."""
        if storageType == "default":
            storageType = self.defaultIntType if type is int \
                          else self.defaultFloatType
        if type is float  :
            if storageType == "F":
                selfmap[varName]=numpy.zeros(len,numpy.float32)
                self.tree.Branch(varName,selfmap[varName],varName+postfix+'/F')
            elif storageType == "D":
                selfmap[varName]=numpy.zeros(len,numpy.float64)
                self.tree.Branch(varName,selfmap[varName],varName+postfix+'/D')
            else:
                raise RuntimeError, 'Unknown storage type %s for branch %s'\
                      % (storageType, varName)
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
                raise RuntimeError, 'Unknown storage type %s for branch %s'\
                      % (storageType, varName)
            selfmap[varName]=numpy.zeros(len,dtypes[storageType])
            self.tree.Branch(varName,selfmap[varName],varName+postfix+'/I')
        else:
            raise RuntimeError, 'Unknown type %s for branch %s'\
                  % (type, varName)
        if title:
            self.tree.GetBranch(varName).SetTitle(title)

    def var(self, varName,type=float, default=-99,
            title=None, storageType="default" ):
        self.branch_(self.vars, varName, type, 1,
                     title=title, storageType=storageType)
        self.defaults[varName] = default

    def vector(self, varName, lenvar, maxlen=None,
               type=float, default=-99, title=None, storageType="default" ):
        """either lenvar is a string, and maxlen an int (variable size array),
        or lenvar is an int and maxlen is not specified (fixed array)"""
        if __builtins__['type'](lenvar) == int:
            #Mike need the __builtins__ since 'type' is a variable here :-/
            #TODO colin that's not true, get rid of this.
            self.branch_(self.vecvars, varName, type, lenvar, postfix="[%d]"\
                         % lenvar, title=title, storageType=storageType)
        else:
            if maxlen == None: RuntimeError,\
               'You must specify a maxlen if making a dynamic array';
            self.branch_(self.vecvars, varName, type, maxlen, postfix="[%s]"\
                         % lenvar, title=title, storageType=storageType)
        self.vecdefaults[varName] = default

    def reset(self):
        for name,value in self.vars.iteritems():
            value[0]=self.defaults[name]
        for name,value in self.vecvars.iteritems():
            value.fill(self.vecdefaults[name])

    def fill(self, varName, value ):
        self.vars[varName][0]=value

    def vfill(self, varName, values ):
        a = self.vecvars[varName]
        for (i,v) in enumerate(values):
            a[i]=v

if __name__=='__main__':

    from ROOT import TFile

    f = TFile('Tree.root','RECREATE')
    t = Tree('Colin', 'Another test tree')
    t.var('a')
    t.var('b')

    t.fill('a', 3)
    t.fill('a', 4)
    t.fill('b', 5)
    t.tree.Fill()

    f.Write()
    f.Close()
