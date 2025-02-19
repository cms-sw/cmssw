import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod

def printPaths(process):
    "print all the paths in the process"
    for p  in process.paths.itervalues():
        printPath(p)

def printPath(pth, indent="", indentDelta=" ", type="path"):
    item = pth._seq
    print indent+type+": "+pth.label_()
    indent += indentDelta
    if isinstance(item,mod._Module):
        print indent+"module: "+item.label_()+" <"+item.type_()+">"
    elif isinstance(item,cms.Sequence):
        printPath(item,indent,indentDelta,"seq")
    else:
        _printOp(item,indent,indentDelta)

def _printOp(op,indent, indentDelta):
    indent += indentDelta
    for i in dir(op):
        o = getattr(op,i)
        if isinstance(o,mod._Module):
            print indent+"module: "+o.label_()+" <"+o.type_()+">"            
        elif isinstance(o, cms.Sequence):
            printPath(o,indent,indentDelta, "seq")
        elif isinstance(o,sqt._Sequenceable):
            _printOp(o,indent,indentDelta)

if __name__ == "__main__":
    import unittest
    class TestPrintPath(unittest.TestCase):
        def testPrint(self):
            p=cms.Process("Test")
            p.foo = cms.EDProducer("Foo")
            p.p = cms.Path(p.foo)
            printPath(p.p)

            p=cms.Process("Test")
            p.foo = cms.EDProducer("Foo")
            p.bar = cms.EDProducer("Bar")
            p.p = cms.Path(p.foo*p.bar)
            printPath(p.p)

            p=cms.Process("Test")
            p.foo = cms.EDProducer("Foo")
            p.bar = cms.EDProducer("Bar")
            p.s = cms.Sequence(p.foo*p.bar)
            p.fii = cms.EDProducer("Fii")
            p.p = cms.Path(p.s*p.fii)
            printPath(p.p)
            
            printPaths(p)

    unittest.main()
