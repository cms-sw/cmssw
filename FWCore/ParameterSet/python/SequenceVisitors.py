#from Modules import OutputModule, EDProducer, EDFilter
from SequenceTypes import *
from Modules import OutputModule, EDProducer, EDFilter, EDAnalyzer


class PathValidator(object):
    def __init__(self):
        self.__label = ''
    def setLabel(self,label):
        self.__label = "'"+label+"' "
    def enter(self,visitee):
        if isinstance(visitee,OutputModule):
            raise ValueError("Path "+self.__label+"cannot contain an OutputModule, '"+visitee.type_()+"', with label '"+visitee.label_()+"'")
        if hasattr(visitee, "label_") and not isinstance(visitee,Sequence):
            if not visitee.hasLabel_():
                raise ValueError("Path "+self.__label+"contains a module of type '"+visitee.type_()+"' which has no assigned label.\n Most likely the module was never added to the process or it got replaced before being inserted into the process.")
    def leave(self,visitee):
        pass

class EndPathValidator(object):
    _presetFilters = ["TriggerResultsFilter", "HLTPrescaler"]
    def __init__(self):   
        self.filtersOnEndpaths = []
        self.__label = ''
    def setLabel(self,label):
        self.__label = "'"+label+"' "
    def enter(self,visitee):
        if isinstance(visitee,EDFilter):
	    if (visitee.type_() in self._presetFilters):
                if (visitee.type_() not in self.filtersOnEndpaths):
                    self.filtersOnEndpaths.append(visitee.type_())
        if hasattr(visitee, "label_") and not isinstance(visitee,Sequence):
            if not visitee.hasLabel_():
                raise ValueError("EndPath "+self.__label+"contains a module of type '"+visitee.type_()+"' which has no assigned label.\n Most likely the module was never added to the process or it got replaced before being inserted into the process.")
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
            unlabeled = EDAnalyzer("UnLabeled")
            producer.setLabel("producer")
            analyzer.setLabel("analyzer")
            output.setLabel("output")
            filter.setLabel("filter")
            s1 = Sequence(analyzer*producer)
            s2 = Sequence(output+filter)
            p1 = Path(s1)
            p2 = Path(s1*s2)
            p3 = Path(s1+unlabeled)
            ep1 = EndPath(producer+output+analyzer)
            ep2 = EndPath(filter+output)
            ep3 = EndPath(s2)
            ep4 = EndPath(unlabeled)
            pathValidator = PathValidator()
            endpathValidator = EndPathValidator()
            p1.visit(pathValidator)
            self.assertRaises(ValueError, p2.visit, pathValidator) 
            self.assertRaises(ValueError, p3.visit, pathValidator) 
            ep1.visit(endpathValidator) 
            ep2.visit(endpathValidator) 
            ep3.visit(endpathValidator) 
            self.assertRaises(ValueError, ep4.visit, endpathValidator) 

    unittest.main()

