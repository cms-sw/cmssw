import FWCore.ParameterSet.Config as cms
import sys

## Helpers to perform some technically boring tasks like looking for all modules with a given parameter
## and replacing that to a given value

# Next two lines are for backward compatibility, the imported functions and
# classes used to be defined in this file.
from FWCore.ParameterSet.MassReplace import massSearchReplaceAnyInputTag, MassSearchReplaceAnyInputTagVisitor
from FWCore.ParameterSet.MassReplace import massSearchReplaceParam, MassSearchParamVisitor, MassSearchReplaceParamVisitor

def getPatAlgosToolsTask(process):
    taskName = "patAlgosToolsTask"
    if hasattr(process, taskName):
        task = getattr(process, taskName)
        if not isinstance(task, cms.Task):
            raise Exception("patAlgosToolsTask does not have type Task")
    else:
        setattr(process, taskName, cms.Task())
        task = getattr(process, taskName)
    return task

def associatePatAlgosToolsTask(process):
    task = getPatAlgosToolsTask(process)
    process.schedule.associate(task)

def addToProcessAndTask(label, module, process, task):
    setattr(process, label, module)
    task.add(getattr(process, label))

def addESProducers(process,config):
	config = config.replace("/",".")
	#import RecoBTag.Configuration.RecoBTag_cff as btag
	#print btag
	module = __import__(config)
	for name in dir(sys.modules[config]):
		item = getattr(sys.modules[config],name)
		if isinstance(item,cms._Labelable) and not isinstance(item,cms._ModuleSequenceType) and not name.startswith('_') and not (name == "source" or name == "looper" or name == "subProcess") and not type(item) is cms.PSet:
			if 'ESProducer' in item.type_():
				setattr(process,name,item)

def loadWithPrefix(process,moduleName,prefix='',loadedProducersAndFilters=None):
        loadWithPrePostfix(process,moduleName,prefix,'',loadedProducersAndFilters)

def loadWithPostfix(process,moduleName,postfix='',loadedProducersAndFilters=None):
        loadWithPrePostfix(process,moduleName,'',postfix,loadedProducersAndFilters)

def loadWithPrePostfix(process,moduleName,prefix='',postfix='',loadedProducersAndFilters=None):
	moduleName = moduleName.replace("/",".")
        module = __import__(moduleName)
	#print module.PatAlgos.patSequences_cff.patDefaultSequence
        extendWithPrePostfix(process,sys.modules[moduleName],prefix,postfix,loadedProducersAndFilters)

def addToTask(loadedProducersAndFilters, module):
    if loadedProducersAndFilters:
        if isinstance(module, cms.EDProducer) or isinstance(module, cms.EDFilter):
            loadedProducersAndFilters.add(module)

def extendWithPrePostfix(process,other,prefix,postfix,loadedProducersAndFilters=None):
    """Look in other and find types which we can use"""
    # enable explicit check to avoid overwriting of existing objects
    #__dict__['_Process__InExtendCall'] = True

    if loadedProducersAndFilters:
        task = getattr(process, loadedProducersAndFilters)
        if not isinstance(task, cms.Task):
            raise Exception("extendWithPrePostfix argument must be name of Task type object attached to the process or None")
    else:
        task = None

    sequence = cms.Sequence()
    sequence._moduleLabels = []
    for name in dir(other):
        #'from XX import *' ignores these, and so should we.
        if name.startswith('_'):
            continue
        item = getattr(other,name)
        if name == "source" or name == "looper" or name == "subProcess":
            continue
        elif isinstance(item,cms._ModuleSequenceType):
            continue
        elif isinstance(item,cms.Task):
            continue
        elif isinstance(item,cms.Schedule):
            continue
        elif isinstance(item,cms.VPSet) or isinstance(item,cms.PSet):
            continue
        elif isinstance(item,cms._Labelable):
            if not item.hasLabel_():
                item.setLabel(name)
            if prefix != '' or postfix != '':
                newModule = item.clone()
                if isinstance(item,cms.ESProducer):
                    newName =name
                else:
                    if 'TauDiscrimination' in name:
                        process.__setattr__(name,item)
                        addToTask(task, item)
                    newName = prefix+name+postfix
                process.__setattr__(newName,newModule)
                addToTask(task, newModule)
                if isinstance(newModule, cms._Sequenceable) and not newName == name:
                    sequence +=getattr(process,newName)
                    sequence._moduleLabels.append(item.label())
            else:
                process.__setattr__(name,item)
                addToTask(task, item)

    if prefix != '' or postfix != '':
        for label in sequence._moduleLabels:
            massSearchReplaceAnyInputTag(sequence, label, prefix+label+postfix,verbose=False,moduleLabelOnly=True)

def applyPostfix(process, label, postfix):
    result = None
    if hasattr(process, label+postfix):
        result = getattr(process, label + postfix)
    else:
        raise ValueError("Error in <applyPostfix>: No module of name = %s attached to process !!" % (label + postfix))
    return result

def removeIfInSequence(process, target,  sequenceLabel, postfix=""):
    labels = __labelsInSequence(process, sequenceLabel, postfix)
    if target+postfix in labels:
        getattr(process, sequenceLabel+postfix).remove(
            getattr(process, target+postfix)
            )

def __labelsInSequence(process, sequenceLabel, postfix=""):
    result = [ m.label()[:-len(postfix)] for m in listModules( getattr(process,sequenceLabel+postfix))]
    result.extend([ m.label()[:-len(postfix)] for m in listSequences( getattr(process,sequenceLabel+postfix))]  )
    if postfix == "":
        result = [ m.label() for m in listModules( getattr(process,sequenceLabel+postfix))]
        result.extend([ m.label() for m in listSequences( getattr(process,sequenceLabel+postfix))]  )
    return result

#FIXME name is not generic enough now
class GatherAllModulesVisitor(object):
    """Visitor that travels within a cms.Sequence, and returns a list of objects of type gatheredInance(e.g. modules) that have it"""
    def __init__(self, gatheredInstance=cms._Module):
        self._modules = []
        self._gatheredInstance= gatheredInstance
    def enter(self,visitee):
        if isinstance(visitee,self._gatheredInstance):
            self._modules.append(visitee)
    def leave(self,visitee):
        pass
    def modules(self):
        return self._modules

class CloneSequenceVisitor(object):
    """Visitor that travels within a cms.Sequence, and returns a cloned version of the Sequence.
    All modules and sequences are cloned and a postfix is added"""
    def __init__(self, process, label, postfix, removePostfix="", noClones = [], addToTask = False):
        self._process = process
        self._postfix = postfix
        self._removePostfix = removePostfix
        self._noClones = noClones
        self._addToTask = addToTask
        self._moduleLabels = []
        self._clonedSequence = cms.Sequence()
        setattr(process, self._newLabel(label), self._clonedSequence)
        if addToTask:
            self._patAlgosToolsTask = getPatAlgosToolsTask(process)

    def enter(self, visitee):
        if isinstance(visitee, cms._Module):
            label = visitee.label()
            newModule = None
            if label in self._noClones: #keep unchanged
                newModule = getattr(self._process, label)
            elif label in self._moduleLabels: # has the module already been cloned ?
                newModule = getattr(self._process, self._newLabel(label))
            else:
                self._moduleLabels.append(label)
                newModule = visitee.clone()
                setattr(self._process, self._newLabel(label), newModule)
                if self._addToTask:
                    self._patAlgosToolsTask.add(getattr(self._process, self._newLabel(label)))
            self.__appendToTopSequence(newModule)

    def leave(self, visitee):
        pass

    def clonedSequence(self):
        for label in self._moduleLabels:
            massSearchReplaceAnyInputTag(self._clonedSequence, label, self._newLabel(label), moduleLabelOnly=True, verbose=False)
        self._moduleLabels = [] # prevent the InputTag replacement next time the 'clonedSequence' function is called.
        return self._clonedSequence

    def _newLabel(self, label):
        if self._removePostfix != "":
            if label[-len(self._removePostfix):] == self._removePostfix:
                label = label[0:-len(self._removePostfix)]
            else:
                raise Exception("Tried to remove postfix %s from label %s, but it wasn't there" % (self._removePostfix, label))
        return label + self._postfix

    def __appendToTopSequence(self, visitee):
        self._clonedSequence += visitee

def listModules(sequence):
    visitor = GatherAllModulesVisitor(gatheredInstance=cms._Module)
    sequence.visit(visitor)
    return visitor.modules()

def listSequences(sequence):
    visitor = GatherAllModulesVisitor(gatheredInstance=cms.Sequence)
    sequence.visit(visitor)
    return visitor.modules()

def jetCollectionString(prefix='', algo='', type=''):
    """
    ------------------------------------------------------------------
    return the string of the jet collection module depending on the
    input vaules. The default return value will be 'patAK5CaloJets'.

    algo   : indicating the algorithm type of the jet [expected are
             'AK5', 'IC5', 'SC7', ...]
    type   : indicating the type of constituents of the jet [expec-
             ted are 'Calo', 'PFlow', 'JPT', ...]
    prefix : prefix indicating the type of pat collection module (ex-
             pected are '', 'selected', 'clean').
    ------------------------------------------------------------------
    """
    if(prefix==''):
        jetCollectionString ='pat'
    else:
        jetCollectionString =prefix
        jetCollectionString+='Pat'
    jetCollectionString+='Jets'
    jetCollectionString+=algo
    jetCollectionString+=type
    return jetCollectionString

def contains(sequence, moduleName):
    """
    ------------------------------------------------------------------
    return True if a module with name 'module' is contained in the
    sequence with name 'sequence' and False otherwise. This version
    is not so nice as it also returns True for any substr of the name
    of a contained module.

    sequence : sequence [e.g. process.patDefaultSequence]
    module   : module name as a string
    ------------------------------------------------------------------
    """
    return not sequence.__str__().find(moduleName)==-1



def cloneProcessingSnippet(process, sequence, postfix, removePostfix="", noClones = [], addToTask = False):
   """
   ------------------------------------------------------------------
   copy a sequence plus the modules and sequences therein
   both are renamed by getting a postfix
   input tags are automatically adjusted
   ------------------------------------------------------------------
   """
   result = sequence
   if not postfix == "":
       visitor = CloneSequenceVisitor(process, sequence.label(), postfix, removePostfix, noClones, addToTask)
       sequence.visit(visitor)
       result = visitor.clonedSequence()
   return result

if __name__=="__main__":
   import unittest
   class TestModuleCommand(unittest.TestCase):
       def setUp(self):
           """Nothing to do """
           pass
       def testCloning(self):
           p = cms.Process("test")
           p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
           p.b = cms.EDProducer("b", src=cms.InputTag("a"))
           p.c = cms.EDProducer("c", src=cms.InputTag("b","instance"))
           p.s = cms.Sequence(p.a*p.b*p.c *p.a)
           cloneProcessingSnippet(p, p.s, "New", addToTask = True)
           self.assertEqual(p.dumpPython(),
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.a = cms.EDProducer("a",
    src = cms.InputTag("gen")
)


process.aNew = cms.EDProducer("a",
    src = cms.InputTag("gen")
)


process.b = cms.EDProducer("b",
    src = cms.InputTag("a")
)


process.bNew = cms.EDProducer("b",
    src = cms.InputTag("aNew")
)


process.c = cms.EDProducer("c",
    src = cms.InputTag("b","instance")
)


process.cNew = cms.EDProducer("c",
    src = cms.InputTag("bNew","instance")
)


process.patAlgosToolsTask = cms.Task(process.aNew, process.bNew, process.cNew)


process.s = cms.Sequence(process.a+process.b+process.c+process.a)


process.sNew = cms.Sequence(process.aNew+process.bNew+process.cNew+process.aNew)


""")
       def testContains(self):
           p = cms.Process("test")
           p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
           p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
           p.c = cms.EDProducer("ac", src=cms.InputTag("b"))
           p.s1 = cms.Sequence(p.a*p.b*p.c)
           p.s2 = cms.Sequence(p.b*p.c)
           self.assert_( contains(p.s1, "a") )
           self.assert_( not contains(p.s2, "a") )
       def testJetCollectionString(self):
           self.assertEqual(jetCollectionString(algo = 'Foo', type = 'Bar'), 'patJetsFooBar')
           self.assertEqual(jetCollectionString(prefix = 'prefix', algo = 'Foo', type = 'Bar'), 'prefixPatJetsFooBar')
       def testListModules(self):
           p = cms.Process("test")
           p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
           p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
           p.c = cms.EDProducer("ac", src=cms.InputTag("b"))
           p.s = cms.Sequence(p.a*p.b*p.c)
           self.assertEqual([p.a,p.b,p.c], listModules(p.s))

   unittest.main()
