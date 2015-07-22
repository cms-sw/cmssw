#!/bin/env python
#
# Objects to be used with AutoFillTreeProducer
# 
# the variable declaration contains both the booking information and a function to fill the variable
#
# TODO: more documentation needed here!

class NTupleVariable:
    """Branch containing an individual variable (either of the event or of an object), created with a name and a function to compute it
       - name, type, help, default: obvious 
       - function: a function that taken an object computes the value to fill (e.g. lambda event : len(event.goodVertices))
    """
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
    def __repr__(self):
        return "<NTupleVariable[%s]>" % self.name


class NTupleObjectType:
    """Type defining a collection of variables associated to a single object. Contans NTupleVariable and NTupleSubObject"""
    def __init__(self,name,baseObjectTypes=[],mcOnly=[],variables=[]):
        self.name = name
        self.baseObjectTypes = baseObjectTypes
        self.mcOnly = mcOnly
        self.variables = []
        self.subObjects = []
        for v in variables:
           if issubclass(v.__class__,NTupleSubObject):
                self.subObjects.append(v)
           else:
                self.variables.append(v)
        self._subObjectVars = {}
    def ownVars(self,isMC):
        """Return only my vars, not including the ones from the bases"""
        vars = [ v for v in self.variables if (isMC or not v.mcOnly) ]
        if self.subObjects:
            if isMC not in self._subObjectVars:
                subvars = []
                for so in self.subObjects:
                    if so.mcOnly and not isMC: continue
                    for subvar in so.objectType.allVars(isMC):
                        subvars.append(NTupleVariable(so.name+"_"+subvar.name,
                                  #DebugComposer(so,subvar),#lambda object : subvar(so(object)),
                                  lambda object, subvar=subvar, so=so : subvar(so(object)), 
                                  # ^-- lambda object : subvar(so(object)) doesn't work due to scoping, see
                                  #     http://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture-in-python/2295372#2295372
                                  type = subvar.type, help = subvar.help, default = subvar.default, mcOnly = subvar.mcOnly,
                                  filler = subvar.filler))
                self._subObjectVars[isMC] = subvars
            vars += self._subObjectVars[isMC]
        return vars
    def allVars(self,isMC):
        """Return all vars, including the base ones. Duplicate bases are not added twice"""
        ret = []; names = {}
        if not isMC and self.mcOnly: return []
        for base in self.allBases():
            if not isMC and base.mcOnly: continue
            for var in base.ownVars(isMC):
                if var.name in names: raise RuntimeError, "Duplicate definition of variable %s from %s and %s" % (var.name, base.name, names[var.name])
                names[var.name] = base.name
                ret.append(var)
        for var in self.ownVars(isMC):
            if var.name in names: raise RuntimeError, "Duplicate definition of variable %s from %s and %s" % (var.name, self.name, names[var.name])
            names[var.name] = self.name
            ret.append(var)
        return ret
    def allBases(self):
        ret = []
        for b in self.baseObjectTypes:
            if b not in ret: 
                ret.append(b)
            for b2 in b.allBases():
                if b2 not in ret:
                    ret.append(b2)
        return ret
    def removeVariable(self,name):
        self.variables = [ v for v in self.variables if v.name != name]
    def __repr__(self):
        return "<NTupleObjectType[%s]>" % self.name




class NTupleSubObject:
    """Type to add a sub-object within an NTupleObjectType, given a name (used as prefix), a function to extract the sub-object and NTupleObjectType to define tye type"""
    def __init__(self,name,function,objectType,mcOnly=False):
        self.name = name
        self.function = function
        self.objectType = objectType
        self.mcOnly = mcOnly
    def __call__(self,object):
        return self.function(object)

class NTupleObject:
    """Type defining a set of branches associated to a single object (i.e. an instance of NTupleObjectType)"""
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
    def __repr__(self):
        return "<NTupleObject[%s]>" % self.name


class NTupleCollection:
    """Type defining a set of branches associated to a list of objects (i.e. an instance of NTupleObjectType)"""
    def __init__(self, name, objectType, maxlen, help="", mcOnly=False, sortAscendingBy=None, sortDescendingBy=None, filter=None):
        self.name = name
        self.objectType = objectType
        self.maxlen = maxlen
        self.help = help
        if objectType.mcOnly and mcOnly == False: 
            #print "collection %s is set to mcOnly since the type %s is mcOnly" % (name, objectType.name)
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
            name="%s_%s" % (self.name, v.name) if v.name != "" else self.name
            treeNumpy.vector(name, "n"+self.name, self.maxlen, type=v.type, default=v.default, title=h, filler=v.filler)
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
            name="%s_%s" % (self.name, v.name) if v.name != "" else self.name
            treeNumpy.vfill(name, [ v(collection[i]) for i in xrange(num) ])
    def __repr__(self):
        return "<NTupleCollection[%s]>" % self.name

    def get_cpp_declaration(self, isMC):
        s = []
        for v in self.objectType.allVars(isMC):
            s += ["{0} {1}__{2}[{3}];".format(v.type.__name__, self.name, v.name, self.maxlen)]
        return "\n".join(s)

    def get_cpp_wrapper_class(self, isMC):
        s = "class %s {\n" % self.name
        s += "public:\n"
        for v in self.objectType.allVars(isMC):
            s += "    {0} {1};\n".format(v.type.__name__, v.name)
        s += "};\n"
        return s

    def get_py_wrapper_class(self, isMC):
        s = "class %s:\n" % self.name
        s += "    def __init__(self, tree, n):\n"
        for v in self.objectType.allVars(isMC):
            if len(v.name)>0:
                s += "        self.{0} = tree.{1}_{2}[n];\n".format(v.name, self.name, v.name)
            else:
                s += "        self.{0} = tree.{0}[n];\n".format(self.name)

        s += "    @staticmethod\n"
        s += "    def make_array(event):\n"
        s += "        return [{0}(event.input, i) for i in range(event.input.n{0})]\n".format(self.name)
        return s


