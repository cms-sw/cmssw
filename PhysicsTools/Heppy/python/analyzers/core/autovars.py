#!/bin/env python
#
# Objects to be used with AutoFillTreeProducer
# 
# the variable declaration contains both the booking information and a function to fill the variable
#
# TODO: more documentation needed here!

class NTupleVariable:
    def __init__(self, name, function, type=float, help="", default=-99, mcOnly=False, filler=None):
        self.name = name
        self.function = function
        self.type = type
        self.help = help
        self.default = default
        self.mcOnly  = mcOnly
        self.filler  = filler
    def __call__(self,object):
        ret = self.function(object)
        return ret
    def makeBranch(self,treeNumpy,isMC):
        if self.mcOnly and not isMC: return
        treeNumpy.var(self.name, type=self.type, default=self.default, title=self.help, filler=self.filler)
    def fillBranch(self,treeNumpy,object,isMC):
        if self.mcOnly and not isMC: return
        treeNumpy.fill(self.name, self(object))

class NTupleObjectType:
    def __init__(self,name,baseObjectTypes=[],mcOnly=[],variables=[]):
        self.name = name
        self.baseObjectTypes = baseObjectTypes
        self.mcOnly = mcOnly
        self.variables = variables
    def allVars(self,isMC):
        ret = []; names = {}
        if not isMC and self.mcOnly: return []
        for base in self.baseObjectTypes:
            if not isMC and base.mcOnly: continue
            for var in base.allVars(isMC):
                if var.name in names: raise RuntimeError, "Duplicate definition of variable %s from %s and %s" % (var.name, base.name, names[var.name])
                names[var.name] = base.name
                ret.append(var)
        for var in self.variables:
            if not isMC and var.mcOnly: continue
            if var.name in names: raise RuntimeError, "Duplicate definition of variable %s from %s and %s" % (var.name, self.name, names[var.name])
            names[var.name] = self.name
            ret.append(var)
        return ret
    def removeVariable(self,name):
        self.variables = [ v for v in self.variables if v.name != name]

class NTupleObject:
    def __init__(self, name, objectType, help="", mcOnly=False):
        self.name = name
        self.objectType = objectType
        self.mcOnly = mcOnly
        self.help = ""
    def makeBranches(self,treeNumpy,isMC):
        if not isMC and self.mcOnly: return
        allvars = self.objectType.allVars(isMC)
        for v in allvars:
            h = v.help
            if self.help: h = "%s for %s" % ( h if h else v.name, self.help )
            treeNumpy.var("%s_%s" % (self.name, v.name), type=v.type, default=v.default, title=h, filler=v.filler)
    def fillBranches(self,treeNumpy,object,isMC):
        if self.mcOnly and not isMC: return
        allvars = self.objectType.allVars(isMC)
        for v in allvars:
            treeNumpy.fill("%s_%s" % (self.name, v.name), v(object))


class NTupleCollection:
    def __init__(self, name, objectType, maxlen, help="", mcOnly=False, sortAscendingBy=None, sortDescendingBy=None, filter=None):
        self.name = name
        self.objectType = objectType
        self.maxlen = maxlen
        self.help = help
        if objectType.mcOnly and mcOnly == False: 
            print "collection %s is set to mcOnly since the type %s is mcOnly" % (name, objectType.name)
            mcOnly = True
        self.mcOnly = mcOnly
        if sortAscendingBy != None and sortDescendingBy != None:
            raise RuntimeError, "Cannot specify two sort conditions"
        self.filter = filter
        self.sortAscendingBy  = sortAscendingBy
        self.sortDescendingBy = sortDescendingBy
    def makeBranchesScalar(self,treeNumpy,isMC):
        if not isMC and self.objectType.mcOnly: return
        treeNumpy.var("n"+self.name, int)
        allvars = self.objectType.allVars(isMC)
        for v in allvars:
            for i in xrange(1,self.maxlen+1):
                h = v.help
                if self.help: h = "%s for %s [%d]" % ( h if h else v.name, self.help, i-1 )
                treeNumpy.var("%s%d_%s" % (self.name, i, v.name), type=v.type, default=v.default, title=h, filler=v.filler)
    def makeBranchesVector(self,treeNumpy,isMC):
        if not isMC and self.objectType.mcOnly: return
        treeNumpy.var("n"+self.name, int)
        allvars = self.objectType.allVars(isMC)
        for v in allvars:
            h = v.help
            if self.help: h = "%s for %s" % ( h if h else v.name, self.help )
            treeNumpy.vector("%s_%s" % (self.name, v.name), "n"+self.name, self.maxlen, type=v.type, default=v.default, title=h, filler=v.filler)
    def fillBranchesScalar(self,treeNumpy,collection,isMC):
        if not isMC and self.objectType.mcOnly: return
        if self.filter != None: collection = [ o for o in collection if self.filter(o) ]
        if self.sortAscendingBy != None: collection  = sorted(collection, key=self.sortAscendingBy)
        if self.sortDescendingBy != None: collection = sorted(collection, key=self.sortDescendingBy, reverse=True)
        num = min(self.maxlen,len(collection))
        treeNumpy.fill("n"+self.name, num)
        allvars = self.objectType.allVars(isMC)
        for i in xrange(num): 
            o = collection[i]
            for v in allvars:
                treeNumpy.fill("%s%d_%s" % (self.name, i+1, v.name), v(o))
    def fillBranchesVector(self,treeNumpy,collection,isMC):
        if not isMC and self.objectType.mcOnly: return
        if self.filter != None: collection = [ o for o in collection if self.filter(o) ]
        if self.sortAscendingBy != None: collection  = sorted(collection, key=self.sortAscendingBy)
        if self.sortDescendingBy != None: collection = sorted(collection, key=self.sortDescendingBy, reverse=True)
        num = min(self.maxlen,len(collection))
        treeNumpy.fill("n"+self.name, num)
        allvars = self.objectType.allVars(isMC)
        for v in allvars:
            treeNumpy.vfill("%s_%s" % (self.name, v.name), [ v(collection[i]) for i in xrange(num) ])


