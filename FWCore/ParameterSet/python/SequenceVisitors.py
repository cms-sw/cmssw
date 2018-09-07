from SequenceTypes import *
from Modules import OutputModule, EDProducer, EDFilter, EDAnalyzer, Service, ESProducer, ESSource, _Module
from Mixins import _Labelable
import six

# Use this on Tasks in the Schedule
class ScheduleTaskValidator(object):
    def __init__(self):
        pass
    def enter(self,visitee):
        if visitee.isLeaf():
            if isinstance(visitee, _Labelable):
                if not visitee.hasLabel_():
                    raise ValueError("A task associated with the Schedule contains a module of type '"+visitee.type_()+"'\nwhich has no assigned label.")
            elif isinstance(visitee, Service):
                if not visitee._inProcess:
                    raise ValueError("A task associated with the Schedule contains a service of type '"+visitee.type_()+"'\nwhich is not attached to the process.")
    def leave(self,visitee):
        pass

# Use this on Paths
class PathValidator(object):
    def __init__(self):
        self.__label = ''
    def setLabel(self,label):
        self.__label = "'"+label+"' "
    def enter(self,visitee):
        if isinstance(visitee,OutputModule):
            raise ValueError("Path "+self.__label+"cannot contain an OutputModule, '"+visitee.type_()+"', with label '"+visitee.label_()+"'")
        if visitee.isLeaf():
            if isinstance(visitee, _Labelable):
                if not visitee.hasLabel_():
                    raise ValueError("Path "+self.__label+"contains a module of type '"+visitee.type_()+"' which has no assigned label.")
            elif isinstance(visitee, Service):
                if not visitee._inProcess:
                    raise ValueError("Path "+self.__label+"contains a service of type '"+visitee.type_()+"' which is not attached to the process.\n")
    def leave(self,visitee):
        pass

# Use this on EndPaths
class EndPathValidator(object):
    _presetFilters = ["TriggerResultsFilter", "HLTPrescaler"]
    def __init__(self):   
        self.filtersOnEndpaths = []
        self.__label = ''
        self._levelInTasks = 0
    def setLabel(self,label):
        self.__label = "'"+label+"' "
    def enter(self,visitee):
        if visitee.isLeaf():
            if isinstance(visitee, _Labelable):
                if not visitee.hasLabel_():
                    raise ValueError("EndPath "+self.__label+"contains a module of type '"+visitee.type_()+"' which has\nno assigned label.")
            elif isinstance(visitee, Service):
                if not visitee._inProcess:
                    raise ValueError("EndPath "+self.__label+"contains a service of type '"+visitee.type_()+"' which is not attached to the process.\n")
        if isinstance(visitee, Task):
            self._levelInTasks += 1
        if self._levelInTasks > 0:
            return
        if isinstance(visitee,EDFilter):
            if (visitee.type_() in self._presetFilters):
                if (visitee.type_() not in self.filtersOnEndpaths):
                    self.filtersOnEndpaths.append(visitee.type_())
    def leave(self,visitee):
        if self._levelInTasks > 0:
            if isinstance(visitee, Task):
                self._levelInTasks -= 1

class NodeVisitor(object):
    """Form sets of all modules, ESProducers, ESSources and Services in visited objects. Can be used
    to visit Paths, EndPaths, Sequences or Tasks. Includes in sets objects on sub-Sequences and sub-Tasks"""
    def __init__(self):
        self.modules = set()
        self.esProducers = set()
        self.esSources = set()
        self.services = set()
    def enter(self,visitee):
        if visitee.isLeaf():
            if isinstance(visitee, _Module):
                self.modules.add(visitee)
            elif isinstance(visitee, ESProducer):
                self.esProducers.add(visitee)
            elif isinstance(visitee, ESSource):
                self.esSources.add(visitee)
            elif isinstance(visitee, Service):
                self.services.add(visitee)
    def leave(self,visitee):
        pass

class CompositeVisitor(object):
    """ Combines 3 different visitor classes in 1 so we only have to visit all the paths and endpaths once"""
    def __init__(self, validator, node, decorated):
        self._validator = validator
        self._node = node
        self._decorated = decorated
    def enter(self, visitee):
        self._validator.enter(visitee)
        self._node.enter(visitee)
        self._decorated.enter(visitee)
    def leave(self, visitee):
        self._validator.leave(visitee)
        # The node visitor leave function does nothing
        #self._node.leave(visitee)
        self._decorated.leave(visitee)

class ModuleNamesFromGlobalsVisitor(object):
    """Fill a list with the names of Event module types in a sequence. The names are determined
    by using globals() to lookup the variable names assigned to the modules. This
    allows the determination of the labels before the modules have been attached to a Process."""
    def __init__(self,globals_,l):
        self._moduleToName = { v[1]:v[0] for v in six.iteritems(globals_) if isinstance(v[1],_Module) }
        self._names =l
    def enter(self,node):
        if isinstance(node,_Module):
            self._names.append(self._moduleToName[node])
    def leave(self,node):
        return

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

