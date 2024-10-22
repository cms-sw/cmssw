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
                if isinstance(value, cms.PSet) or isinstance(value, cms.EDProducer) or isinstance(value, cms.EDAlias):
                    # EDProducer and EDAlias to support SwitchProducer
                    self.doIt(value,base+"."+name)
                elif value.isCompatibleCMSType(cms.VPSet):
                    for (i,ps) in enumerate(value): self.doIt(ps, "%s.%s[%d]"%(base,name,i) )
                elif value.isCompatibleCMSType(cms.VInputTag) and value:
                    for (i,n) in enumerate(value):
                        # VInputTag can be declared as a list of strings, so ensure that n is formatted correctly
                        n = self.standardizeInputTagFmt(n)
                        if (n == self._paramSearch):
                            if self._verbose:print("Replace %s.%s[%d] %s ==> %s " % (base, name, i, self._paramSearch, self._paramReplace))
                            if not value.isTracked():
                                value[i] = cms.untracked.InputTag(self._paramReplace.getModuleLabel(),
                                                                  self._paramReplace.getProductInstanceLabel(),
                                                                  self._paramReplace.getProcessName())
                            else:
                                value[i] = self._paramReplace
                        elif self._moduleLabelOnly and n.moduleLabel == self._paramSearch.moduleLabel:
                            nrep = n; nrep.moduleLabel = self._paramReplace.moduleLabel
                            if self._verbose:print("Replace %s.%s[%d] %s ==> %s " % (base, name, i, n, nrep))
                            value[i] = nrep
                elif value.isCompatibleCMSType(cms.InputTag) and value:
                    if value == self._paramSearch:
                        if self._verbose:print("Replace %s.%s %s ==> %s " % (base, name, self._paramSearch, self._paramReplace))
                        from copy import deepcopy
                        if not value.isTracked():
                            # the existing value should stay untracked even if the given parameter is tracked
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
        if isinstance(visitee, cms.SwitchProducer):
            for modName in visitee.parameterNames_():
                self.doIt(getattr(visitee, modName), "%s.%s"%(str(visitee), modName))
        else:
            self.doIt(visitee, str(visitee))
    def doIt(self, mod, name):
        if (hasattr(mod,self._paramName)):
            if getattr(mod,self._paramName) == self._paramSearch:
                if self._verbose:print("Replaced %s.%s: %s => %s" % (name,self._paramName,getattr(mod,self._paramName),self._paramValue))
                setattr(mod,self._paramName,self._paramValue)
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
    class SwitchProducerTest(cms.SwitchProducer):
        def __init__(self, **kargs):
            super(SwitchProducerTest,self).__init__(
                dict(
                    test1 = lambda: (True, -10),
                    test2 = lambda: (True, -9),
                    test3 = lambda: (True, -8),
                    test4 = lambda: (True, -7)
                ), **kargs)

    class TestModuleCommand(unittest.TestCase):

        def testMassSearchReplaceAnyInputTag(self):
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"), usrc=cms.untracked.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("b"), src2 = cms.InputTag("c"), usrc = cms.untracked.InputTag("b")),
                                 nestedv = cms.VPSet(cms.PSet(src = cms.InputTag("b")), cms.PSet(src = cms.InputTag("d"))),
                                 unestedv = cms.untracked.VPSet(cms.untracked.PSet(src = cms.InputTag("b")), cms.untracked.PSet(src = cms.InputTag("d"))),
                                 vec = cms.VInputTag(cms.InputTag("a"), cms.InputTag("b"), cms.InputTag("c"), cms.InputTag("d")),
                                 uvec = cms.untracked.VInputTag(cms.untracked.InputTag("a"), cms.untracked.InputTag("b"), cms.untracked.InputTag("c"), cms.untracked.InputTag("d")),
                                )
            p.sp = SwitchProducerTest(
                test1 = cms.EDProducer("a", src = cms.InputTag("b"),
                                       nested = cms.PSet(src = cms.InputTag("b"), src2 = cms.InputTag("c"), usrc = cms.untracked.InputTag("b"))
                                       ),
                test2 = cms.EDProducer("b", src = cms.InputTag("c"),
                                       nested = cms.PSet(src = cms.InputTag("b"), src2 = cms.InputTag("c"), usrc = cms.untracked.InputTag("b"))
                                       ),
            )
            p.op = cms.EDProducer("op", src = cms.optional.InputTag, unset = cms.optional.InputTag, vsrc = cms.optional.VInputTag, vunset = cms.optional.VInputTag)
            p.op2 = cms.EDProducer("op2", src = cms.optional.InputTag, unset = cms.optional.InputTag, vsrc = cms.optional.VInputTag, vunset = cms.optional.VInputTag)
            p.op.src="b"
            p.op.vsrc = ["b"]
            p.op2.src=cms.InputTag("b")
            p.op2.vsrc = cms.VInputTag("b")
            p.s = cms.Sequence(p.a*p.b*p.c*p.sp*p.op*p.op2)
            massSearchReplaceAnyInputTag(p.s, cms.InputTag("b"), cms.InputTag("new"))
            self.assertNotEqual(cms.InputTag("new"), p.b.src)
            self.assertEqual(cms.InputTag("new"), p.c.src)
            self.assertEqual(cms.InputTag("new"), p.c.usrc)
            self.assertEqual(cms.InputTag("new"), p.c.nested.src)
            self.assertEqual(cms.InputTag("new"), p.c.nested.usrc)
            self.assertFalse(p.c.nested.usrc.isTracked())
            self.assertNotEqual(cms.InputTag("new"), p.c.nested.src2)
            self.assertEqual(cms.InputTag("new"), p.c.nestedv[0].src)
            self.assertNotEqual(cms.InputTag("new"), p.c.nestedv[1].src)
            self.assertEqual(cms.InputTag("new"), p.c.unestedv[0].src)
            self.assertNotEqual(cms.InputTag("new"), p.c.unestedv[1].src)
            self.assertNotEqual(cms.InputTag("new"), p.c.vec[0])
            self.assertEqual(cms.InputTag("new"), p.c.vec[1])
            self.assertNotEqual(cms.InputTag("new"), p.c.vec[2])
            self.assertNotEqual(cms.InputTag("new"), p.c.vec[3])
            self.assertNotEqual(cms.InputTag("new"), p.c.uvec[0])
            self.assertEqual(cms.InputTag("new"), p.c.uvec[1])
            self.assertNotEqual(cms.InputTag("new"), p.c.uvec[2])
            self.assertNotEqual(cms.InputTag("new"), p.c.uvec[3])
            self.assertFalse(p.c.uvec[0].isTracked())
            self.assertFalse(p.c.uvec[1].isTracked())
            self.assertFalse(p.c.uvec[2].isTracked())
            self.assertFalse(p.c.uvec[3].isTracked())
            self.assertEqual(cms.InputTag("new"), p.sp.test1.src)
            self.assertEqual(cms.InputTag("new"), p.sp.test1.nested.src)
            self.assertEqual(cms.InputTag("c"), p.sp.test1.nested.src2)
            self.assertEqual(cms.untracked.InputTag("new"), p.sp.test1.nested.usrc)
            self.assertEqual(cms.InputTag("c"), p.sp.test2.src)
            self.assertEqual(cms.InputTag("new"), p.sp.test2.nested.src)
            self.assertEqual(cms.InputTag("c"), p.sp.test2.nested.src2)
            self.assertEqual(cms.untracked.InputTag("new"), p.sp.test2.nested.usrc)
            self.assertEqual(cms.InputTag("new"), p.op.src)
            self.assertEqual(cms.InputTag("new"), p.op.vsrc[0])
            self.assertEqual(cms.InputTag("new"), p.op2.src)
            self.assertEqual(cms.InputTag("new"), p.op2.vsrc[0])

        def testMassReplaceInputTag(self):
            process1 = cms.Process("test")
            massReplaceInputTag(process1, "a", "b", False, False, False)
            self.assertEqual(process1.dumpPython(), cms.Process('test').dumpPython())
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("a"), src2 = cms.InputTag("c"), usrc = cms.untracked.InputTag("a")),
                                 nestedv = cms.VPSet(cms.PSet(src = cms.InputTag("a")), cms.PSet(src = cms.InputTag("d"))),
                                 unestedv = cms.untracked.VPSet(cms.untracked.PSet(src = cms.InputTag("a")), cms.untracked.PSet(src = cms.InputTag("d"))),
                                 vec = cms.VInputTag(cms.InputTag("a"), cms.InputTag("b"), cms.InputTag("c"), cms.InputTag("d")),
                                 uvec = cms.untracked.VInputTag(cms.untracked.InputTag("a"), cms.untracked.InputTag("b"), cms.untracked.InputTag("c"), cms.InputTag("d")),
                                )
            p.d = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.e = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.f = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.g = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.h = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.i = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.sp = SwitchProducerTest(
                test1 = cms.EDProducer("a", src = cms.InputTag("a"),
                                       nested = cms.PSet(src = cms.InputTag("a"), src2 = cms.InputTag("c"), usrc = cms.untracked.InputTag("a"))
                                       ),
                test2 = cms.EDProducer("b", src = cms.InputTag("c"),
                                       nested = cms.PSet(src = cms.InputTag("a"), src2 = cms.InputTag("c"), usrc = cms.untracked.InputTag("a"))
                                       ),
            )
            p.s1 = cms.Sequence(p.a*p.b*p.c*p.sp)
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
            self.assertEqual(cms.InputTag("b"), p.c.nested.src)
            self.assertEqual(cms.InputTag("b"), p.c.nested.usrc)
            self.assertFalse(p.c.nested.usrc.isTracked())
            self.assertEqual(cms.InputTag("b"), p.c.nestedv[0].src)
            self.assertEqual(cms.InputTag("b"), p.c.unestedv[0].src)
            self.assertEqual(cms.InputTag("b"), p.c.vec[0])
            self.assertEqual(cms.InputTag("c"), p.c.vec[2])
            self.assertEqual(cms.InputTag("b"), p.c.uvec[0])
            self.assertEqual(cms.InputTag("c"), p.c.uvec[2])
            self.assertFalse(p.c.uvec[0].isTracked())
            self.assertFalse(p.c.uvec[1].isTracked())
            self.assertFalse(p.c.uvec[2].isTracked())
            self.assertEqual(cms.InputTag("a"), p.d.src)
            self.assertEqual(cms.InputTag("b"), p.e.src)
            self.assertEqual(cms.InputTag("b"), p.f.src)
            self.assertEqual(cms.InputTag("b"), p.g.src)
            self.assertEqual(cms.InputTag("b"), p.h.src)
            self.assertEqual(cms.InputTag("b"), p.i.src)
            self.assertEqual(cms.InputTag("b"), p.sp.test1.src)
            self.assertEqual(cms.InputTag("b"), p.sp.test1.nested.src)
            self.assertEqual(cms.InputTag("c"), p.sp.test1.nested.src2)
            self.assertEqual(cms.untracked.InputTag("b"), p.sp.test1.nested.usrc)
            self.assertEqual(cms.InputTag("c"), p.sp.test2.src)
            self.assertEqual(cms.InputTag("b"), p.sp.test2.nested.src)
            self.assertEqual(cms.InputTag("c"), p.sp.test2.nested.src2)
            self.assertEqual(cms.untracked.InputTag("b"), p.sp.test2.nested.usrc)

        def testMassSearchReplaceParam(self):
            p = cms.Process("test")
            p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
            p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
            p.c = cms.EDProducer("ac", src=cms.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("c"), src2 = cms.InputTag("b"))
                                )
            p.d = cms.EDProducer("ac", src=cms.untracked.InputTag("b"),
                                 nested = cms.PSet(src = cms.InputTag("c"), src2 = cms.InputTag("b"))
                                )
            p.sp = SwitchProducerTest(
                test1 = cms.EDProducer("a", src = cms.InputTag("b"),
                                       nested = cms.PSet(src = cms.InputTag("b"))
                                       ),
                test2 = cms.EDProducer("b", src = cms.InputTag("b")),
            )
            p.s = cms.Sequence(p.a*p.b*p.c*p.d*p.sp)
            massSearchReplaceParam(p.s,"src",cms.InputTag("b"),"a")
            self.assertEqual(cms.InputTag("a"),p.c.src)
            self.assertEqual(cms.InputTag("c"),p.c.nested.src)
            self.assertEqual(cms.InputTag("b"),p.c.nested.src2)
            self.assertEqual(cms.untracked.InputTag("a"),p.d.src)
            self.assertEqual(cms.InputTag("c"),p.d.nested.src)
            self.assertEqual(cms.InputTag("b"),p.d.nested.src2)
            self.assertEqual(cms.InputTag("a"),p.sp.test1.src)
            self.assertEqual(cms.InputTag("b"),p.sp.test1.nested.src)
            self.assertEqual(cms.InputTag("a"),p.sp.test2.src)

        def testMassReplaceParam(self):
            process1 = cms.Process("test")
            massReplaceParameter(process1, "src", cms.InputTag("a"), "b", False)
            self.assertEqual(process1.dumpPython(), cms.Process("test").dumpPython())
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
            p.j = cms.EDProducer("ab", src=cms.untracked.InputTag("a"))
            p.sp = SwitchProducerTest(
                test1 = cms.EDProducer("a", src = cms.InputTag("a"),
                                       nested = cms.PSet(src = cms.InputTag("a"))
                                       ),
                test2 = cms.EDProducer("b", src = cms.InputTag("a")),
            )
            p.s1 = cms.Sequence(p.a*p.b*p.c*p.sp)
            p.path1 = cms.Path(p.s1)
            p.s2 = cms.Sequence(p.d)
            p.path2 = cms.Path(p.e)
            p.s3 = cms.Sequence(p.f)
            p.endpath1 = cms.EndPath(p.s3)
            p.endpath2 = cms.EndPath(p.g)
            p.t1 = cms.Task(p.h)
            p.t2 = cms.Task(p.i, p.j)
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
            self.assertEqual(cms.untracked.InputTag("b"), p.j.src)
            self.assertEqual(cms.InputTag("b"),p.sp.test1.src)
            self.assertEqual(cms.InputTag("a"),p.sp.test1.nested.src)
            self.assertEqual(cms.InputTag("b"),p.sp.test2.src)
    unittest.main()
