import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod


def getModulesFromSequence(sequence,list):
    item = sequence._seq
    if isinstance(item,mod._Module):
        list.append(item)
    elif isinstance(item,cms.Sequence):
         getModulesFromSequence(item,list)
    else:
         _getModulesFromOp(item,list)
                                                    

def _getModulesFromOp(op,list):
    for item in dir(op):
        o = getattr(op,item)
        if isinstance(o,mod._Module):
            list.append(o)
        elif isinstance(o, cms.Sequence):
            _getModulesFromOp(o,list)
        elif isinstance(o,sqt._Sequenceable):
            _getModulesFromOp(o,list)
                    

def extractUsedOutputs(process):
    allEndPathModules = []
    for name in process._Process__endpaths:
        endpath = getattr(process,name)
        list = []
        getModulesFromSequence(endpath,list)
        allEndPathModules.extend(list)
    allUsedOutputModules = []
    for module in allEndPathModules:
        if isinstance(module, cms.OutputModule):
            allUsedOutputModules.append(module)
    return allUsedOutputModules

if __name__ == "__main__":
    import unittest
    class TestPrintPath(unittest.TestCase):
        def testGetModules(self):
            p=cms.Process("Test")
            p.foo = cms.EDProducer("Foo")
            p.p = cms.Path(p.foo)
            list = []
            getModulesFromSequence(p.p,list)
            print len(list)

            p=cms.Process("Test")
            p.foo = cms.OutputModule("Foo")
            p.bar = cms.OutputModule("Bar")
            p.unused = cms.OutputModule("Unused")
            p.p = cms.EndPath(p.foo*p.bar)
            usedOutputs = extractUsedOutputs(p)
            print len(usedOutputs)

            p=cms.Process("Test")
            p.foo = cms.EDProducer("Foo")
            p.bar = cms.EDProducer("Bar")
            p.s = cms.Sequence(p.foo*p.bar)
            p.fii = cms.EDProducer("Fii")
            p.p = cms.Path(p.s*p.fii)
            list = []
            getModulesFromSequence(p.p,list)
            print len(list)
                       

    unittest.main()
