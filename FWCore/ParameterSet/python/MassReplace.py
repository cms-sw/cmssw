from __future__ import print_function
import FWCore.ParameterSet.Config as cms

class MassSearchReplaceAnyInputTagVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and replace its value
       It will climb down within PSets, VPSets and VInputTags to find its target"""
    def __init__(self,paramSearch,paramReplace,verbose=False,moduleLabelOnly=False,skipLabelTest=False):
        self._paramSearch  = self.standardizeInputTagFmt(paramSearch)
        self._paramReplace = self.standardizeInputTagFmt(paramReplace)
        self._moduleName   = ''
        self._verbose=verbose
        self._moduleLabelOnly=moduleLabelOnly
        self._skipLabelTest=skipLabelTest
    def doIt(self,pset,base):
        if isinstance(pset, cms._Parameterizable):
            for name in pset.parameterNames_():
                # if I use pset.parameters_().items() I get copies of the parameter values
                # so I can't modify the nested pset
                value = getattr(pset,name)
                type = value.pythonTypeName()
                if type == 'cms.PSet':
                    self.doIt(value,base+"."+name)
                elif type == 'cms.VPSet':
                    for (i,ps) in enumerate(value): self.doIt(ps, "%s.%s[%d]"%(base,name,i) )
                elif type == 'cms.VInputTag':
                    for (i,n) in enumerate(value):
                        # VInputTag can be declared as a list of strings, so ensure that n is formatted correctly
                        n = self.standardizeInputTagFmt(n)
                        if (n == self._paramSearch):
                            if self._verbose:print("Replace %s.%s[%d] %s ==> %s " % (base, name, i, self._paramSearch, self._paramReplace))
                            value[i] = self._paramReplace
                        elif self._moduleLabelOnly and n.moduleLabel == self._paramSearch.moduleLabel:
                            nrep = n; nrep.moduleLabel = self._paramReplace.moduleLabel
                            if self._verbose:print("Replace %s.%s[%d] %s ==> %s " % (base, name, i, n, nrep))
                            value[i] = nrep
                elif type.endswith('.InputTag'):
                    if value == self._paramSearch:
                        if self._verbose:print("Replace %s.%s %s ==> %s " % (base, name, self._paramSearch, self._paramReplace))
                        from copy import deepcopy
                        if 'untracked' in type:
                            setattr(pset, name, cms.untracked.InputTag(self._paramReplace.getModuleLabel(),
                                                                       self._paramReplace.getProductInstanceLabel(),
                                                                       self._paramReplace.getProcessName()))
                        else:
                            setattr(pset, name, deepcopy(self._paramReplace) )
                    elif self._moduleLabelOnly and value.moduleLabel == self._paramSearch.moduleLabel:
                        from copy import deepcopy
                        repl = deepcopy(getattr(pset, name))
                        repl.moduleLabel = self._paramReplace.moduleLabel
                        setattr(pset, name, repl)
                        if self._verbose:print("Replace %s.%s %s ==> %s " % (base, name, value, repl))


    @staticmethod
    def standardizeInputTagFmt(inputTag):
        ''' helper function to ensure that the InputTag is defined as cms.InputTag(str) and not as a plain str '''
        if not isinstance(inputTag, cms.InputTag):
            return cms.InputTag(inputTag)
        return inputTag

    def enter(self,visitee):
        label = ''
        if (not self._skipLabelTest):
            if hasattr(visitee,"hasLabel_") and visitee.hasLabel_():
                label = visitee.label_()
            else: label = '<Module not in a Process>'
        else:
            label = '<Module label not tested>'
        self.doIt(visitee, label)
    def leave(self,visitee):
        pass

def massSearchReplaceAnyInputTag(sequence, oldInputTag, newInputTag,verbose=False,moduleLabelOnly=False,skipLabelTest=False) :
    """Replace InputTag oldInputTag with newInputTag, at any level of nesting within PSets, VPSets, VInputTags..."""
    sequence.visit(MassSearchReplaceAnyInputTagVisitor(oldInputTag,newInputTag,verbose=verbose,moduleLabelOnly=moduleLabelOnly,skipLabelTest=skipLabelTest))

def massReplaceInputTag(process,old="rawDataCollector",new="rawDataRepacker",verbose=False,moduleLabelOnly=False,skipLabelTest=False):
    for s in process.paths_().keys():
        massSearchReplaceAnyInputTag(getattr(process,s), old, new, verbose, moduleLabelOnly, skipLabelTest)
    for s in process.endpaths_().keys():
        massSearchReplaceAnyInputTag(getattr(process,s), old, new, verbose, moduleLabelOnly, skipLabelTest)
    if process.schedule_() is not None:
        for task in process.schedule_()._tasks:
            massSearchReplaceAnyInputTag(task, old, new, verbose, moduleLabelOnly, skipLabelTest)
    return(process)

class MassSearchParamVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and returns a list of modules that have it"""
    def __init__(self,paramName,paramSearch):
        self._paramName   = paramName
        self._paramSearch = paramSearch
        self._modules = []
    def enter(self,visitee):
        if (hasattr(visitee,self._paramName)):
            if getattr(visitee,self._paramName) == self._paramSearch:
                self._modules.append(visitee)
    def leave(self,visitee):
        pass
    def modules(self):
        return self._modules

class MassSearchReplaceParamVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and replaces its value"""
    def __init__(self,paramName,paramSearch,paramValue,verbose=False):
        self._paramName   = paramName
        self._paramValue  = paramValue
        self._paramSearch = paramSearch
        self._verbose = verbose
    def enter(self,visitee):
        if (hasattr(visitee,self._paramName)):
            if getattr(visitee,self._paramName) == self._paramSearch:
                if self._verbose:print("Replaced %s.%s: %s => %s" % (visitee,self._paramName,getattr(visitee,self._paramName),self._paramValue))
                setattr(visitee,self._paramName,self._paramValue)
    def leave(self,visitee):
        pass

def massSearchReplaceParam(sequence,paramName,paramOldValue,paramValue,verbose=False):
    sequence.visit(MassSearchReplaceParamVisitor(paramName,paramOldValue,paramValue,verbose))

def massReplaceParameter(process,name="label",old="rawDataCollector",new="rawDataRepacker",verbose=False):
    for s in process.paths_().keys():
        massSearchReplaceParam(getattr(process,s),name,old,new,verbose)
    for s in process.endpaths_().keys():
        massSearchReplaceParam(getattr(process,s),name,old,new,verbose)
    if process.schedule_() is not None:
        for task in process.schedule_()._tasks:
            massSearchReplaceParam(task, name, old, new, verbose)
    return(process)

if __name__=="__main__":
    import unittest
    class TestModuleCommand(unittest.TestCase):

        def testMassSearchReplaceAnyInputTag(self):
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("b"), src2 = cms.InputTag("c")),
                                 nestedv = cms.VPSet(cms.PSet(src = cms.InputTag("b")), cms.PSet(src = cms.InputTag("d"))),
                                 vec = cms.VInputTag(cms.InputTag("a"), cms.InputTag("b"), cms.InputTag("c"), cms.InputTag("d"))
                                )
            p.s = cms.Sequence(p.a*p.b*p.c)
            massSearchReplaceAnyInputTag(p.s, cms.InputTag("b"), cms.InputTag("new"))
            self.assertNotEqual(cms.InputTag("new"), p.b.src)
            self.assertEqual(cms.InputTag("new"), p.c.src)
            self.assertEqual(cms.InputTag("new"), p.c.nested.src)
            self.assertEqual(cms.InputTag("new"), p.c.nested.src)
            self.assertNotEqual(cms.InputTag("new"), p.c.nested.src2)
            self.assertEqual(cms.InputTag("new"), p.c.nestedv[0].src)
            self.assertNotEqual(cms.InputTag("new"), p.c.nestedv[1].src)
            self.assertNotEqual(cms.InputTag("new"), p.c.vec[0])
            self.assertEqual(cms.InputTag("new"), p.c.vec[1])
            self.assertNotEqual(cms.InputTag("new"), p.c.vec[2])
            self.assertNotEqual(cms.InputTag("new"), p.c.vec[3])

        def testMassReplaceInputTag(self):
            process1 = cms.Process("test")
            massReplaceInputTag(process1, "a", "b", False, False, False)
            self.assertEqual(process1.dumpPython(),
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

""")
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("a"), src2 = cms.InputTag("c")),
                                 nestedv = cms.VPSet(cms.PSet(src = cms.InputTag("a")), cms.PSet(src = cms.InputTag("d"))),
                                 vec = cms.VInputTag(cms.InputTag("a"), cms.InputTag("b"), cms.InputTag("c"), cms.InputTag("d"))
                                )
            p.d = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.e = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.f = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.g = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.h = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.i = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.s1 = cms.Sequence(p.a*p.b*p.c)
            p.path1 = cms.Path(p.s1)
            p.s2 = cms.Sequence(p.d)
            p.path2 = cms.Path(p.e)
            p.s3 = cms.Sequence(p.f)
            p.endpath1 = cms.EndPath(p.s3)
            p.endpath2 = cms.EndPath(p.g)
            p.t1 = cms.Task(p.h)
            p.t2 = cms.Task(p.i)
            p.schedule = cms.Schedule()
            p.schedule.associate(p.t1, p.t2)
            massReplaceInputTag(p, "a", "b", False, False, False)
            self.assertEqual(cms.InputTag("b"), p.b.src)
            self.assertEqual(cms.InputTag("b"), p.c.vec[0])
            self.assertEqual(cms.InputTag("c"), p.c.vec[2])
            self.assertEqual(cms.InputTag("a"), p.d.src)
            self.assertEqual(cms.InputTag("b"), p.e.src)
            self.assertEqual(cms.InputTag("b"), p.f.src)
            self.assertEqual(cms.InputTag("b"), p.g.src)
            self.assertEqual(cms.InputTag("b"), p.h.src)
            self.assertEqual(cms.InputTag("b"), p.i.src)

        def testMassSearchReplaceParam(self):
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("c"))
                                )
            p.s = cms.Sequence(p.a*p.b*p.c)
            massSearchReplaceParam(p.s,"src",cms.InputTag("b"),"a")
            self.assertEqual(cms.InputTag("a"),p.c.src)
            self.assertNotEqual(cms.InputTag("a"),p.c.nested.src)

        def testMassReplaceParam(self):
            process1 = cms.Process("test")
            massReplaceParameter(process1, "src", cms.InputTag("a"), "b", False)
            self.assertEqual(process1.dumpPython(),
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

""")
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("a"), src2 = cms.InputTag("c")),
                                 nestedv = cms.VPSet(cms.PSet(src = cms.InputTag("a")), cms.PSet(src = cms.InputTag("d"))),
                                 vec = cms.VInputTag(cms.InputTag("a"), cms.InputTag("b"), cms.InputTag("c"), cms.InputTag("d"))
                                )
            p.d = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.e = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.f = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.g = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.h = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.i = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.s1 = cms.Sequence(p.a*p.b*p.c)
            p.path1 = cms.Path(p.s1)
            p.s2 = cms.Sequence(p.d)
            p.path2 = cms.Path(p.e)
            p.s3 = cms.Sequence(p.f)
            p.endpath1 = cms.EndPath(p.s3)
            p.endpath2 = cms.EndPath(p.g)
            p.t1 = cms.Task(p.h)
            p.t2 = cms.Task(p.i)
            p.schedule = cms.Schedule()
            p.schedule.associate(p.t1, p.t2)
            massReplaceParameter(p, "src",cms.InputTag("a"), "b", False)
            self.assertEqual(cms.InputTag("gen"), p.a.src)
            self.assertEqual(cms.InputTag("b"), p.b.src)
            self.assertEqual(cms.InputTag("a"), p.c.vec[0])
            self.assertEqual(cms.InputTag("c"), p.c.vec[2])
            self.assertEqual(cms.InputTag("a"), p.d.src)
            self.assertEqual(cms.InputTag("b"), p.e.src)
            self.assertEqual(cms.InputTag("b"), p.f.src)
            self.assertEqual(cms.InputTag("b"), p.g.src)
            self.assertEqual(cms.InputTag("b"), p.h.src)
            self.assertEqual(cms.InputTag("b"), p.i.src)
    unittest.main()
