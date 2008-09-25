#from Modules import OutputModule, EDProducer, EDFilter
from SequenceTypes import *
from Modules import OutputModule, EDProducer, EDFilter, EDAnalyzer

class PathValidator(object):
    def enter(self,visitee):
        if isinstance(visitee,OutputModule):
            raise ValueError("Path cannot have an OutputModule, "+visitee.label_())
    def leave(self,visitee):
        pass

class EndPathValidator(object):
    def enter(self,visitee):
        if isinstance(visitee,EDProducer) or isinstance(visitee,EDFilter):
            raise ValueError("EndPath cannot have an EDProducer or an EDFilter, "+visitee.label_())
    def leave(self,visitee):
        pass


if __name__=="__main__":
    import unittest
    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            pass
        def testValidators(self):
            producer = EDProducer("Producer")
            analyzer = EDAnalyzer("Analyzer")
            output = OutputModule("Out")
            filter = EDFilter("Filter")
            producer.setLabel("producer")
            analyzer.setLabel("analyzer")
            output.setLabel("output")
            filter.setLabel("filter")
            s1 = Sequence(analyzer*producer)
            s2 = Sequence(output+filter)
            p1 = Path(s1)
            p2 = Path(s1*s2)
            ep1 = EndPath(output+analyzer)
            ep2 = EndPath(s1)
            ep3 = EndPath(s2)
            pathValidator = PathValidator()
            endpathValidator = EndPathValidator()
            p1.visit(pathValidator)
            self.assertRaises(ValueError, p2.visit, pathValidator) 
            ep1.visit(endpathValidator) 
            self.assertRaises(ValueError, ep2.visit, endpathValidator)
            self.assertRaises(ValueError, ep3.visit, endpathValidator)

    unittest.main()


