import FWCore.ParameterSet.Config as cms
import sys

## Helpers to perform some technically boring tasks like looking for all modules with a given parameter
## and replacing that to a given value

def addESProducers(process,config):
	config = config.replace("/",".")
	#import RecoBTag.Configuration.RecoBTag_cff as btag
	#print btag
	module = __import__(config)
	for name in dir(sys.modules[config]):
		item = getattr(sys.modules[config],name)
		if isinstance(item,_Labelable) and not isinstance(item,_ModuleSequenceType) and not name.startswith('_') and not (name == "source" or name == "looper" or name == "subProcess") and not type(item) is cms.PSet:
			if 'ESProducer' in item.type_():
				setattr(process,name,item)

def loadWithPostfix(process,moduleName,postfix=''):
	moduleName = moduleName.replace("/",".")
        module = __import__(moduleName)
	#print module.PatAlgos.patSequences_cff.patDefaultSequence
        extendWithPostfix(process,sys.modules[moduleName],postfix)

def extendWithPostfix(process,other,postfix,items=()):
        """Look in other and find types which we can use"""
        # enable explicit check to avoid overwriting of existing objects
        #__dict__['_Process__InExtendCall'] = True

        seqs = dict()
	sequence = cms.Sequence()
	sequence._moduleLabels = []
	sequence.setLabel('tempSequence')
        for name in dir(other):
            #'from XX import *' ignores these, and so should we.
            	if name.startswith('_'):
                	continue
            	item = getattr(other,name)
            	if name == "source" or name == "looper" or name == "subProcess":
			continue
            	elif isinstance(item,cms._ModuleSequenceType):
			continue
            	elif isinstance(item,cms.Schedule):
			continue
            	elif isinstance(item,cms.VPSet) or isinstance(item,cms.PSet):
			continue
            	elif isinstance(item,cms._Labelable):
                	if not item.hasLabel_():
                   		item.setLabel(name)
			if postfix != '':
				newModule = item.clone()
				if isinstance(item,cms.ESProducer):
					newLabel = item.label()
					newName =name
				else:
				        if 'TauDiscrimination' in name:
				                       process.__setattr__(name,item)
					newLabel = item.label()+postfix
					newName = name+postfix
				process.__setattr__(newName,newModule)
				if isinstance(newModule, cms._Sequenceable) and not newName == name:
					sequence +=getattr(process,newName)
					sequence._moduleLabels.append(item.label())
			else:
				process.__setattr__(name,item)

	if postfix != '':
		for label in sequence._moduleLabels:
			massSearchReplaceAnyInputTag(sequence, label, label+postfix,verbose=False,moduleLabelOnly=True)

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
                if self._verbose:print "Replaced %s.%s: %s => %s" % (visitee,self._paramName,getattr(visitee,self._paramName),self._paramValue)
                setattr(visitee,self._paramName,self._paramValue)
    def leave(self,visitee):
        pass

class MassSearchReplaceAnyInputTagVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and replace its value
       It will climb down within PSets, VPSets and VInputTags to find its target"""
    def __init__(self,paramSearch,paramReplace,verbose=False,moduleLabelOnly=False):
        self._paramSearch  = self.standardizeInputTagFmt(paramSearch)
        self._paramReplace = self.standardizeInputTagFmt(paramReplace)
        self._moduleName   = ''
        self._verbose=verbose
        self._moduleLabelOnly=moduleLabelOnly
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
                            if self._verbose:print "Replace %s.%s[%d] %s ==> %s " % (base, name, i, self._paramSearch, self._paramReplace)
                            value[i] = self._paramReplace
                         elif self._moduleLabelOnly and n.moduleLabel == self._paramSearch.moduleLabel:
                            nrep = n; nrep.moduleLabel = self._paramReplace.moduleLabel
                            if self._verbose:print "Replace %s.%s[%d] %s ==> %s " % (base, name, i, n, nrep)
                            value[i] = nrep
                elif type.endswith('.InputTag'):
                    if value == self._paramSearch:
                        if self._verbose:print "Replace %s.%s %s ==> %s " % (base, name, self._paramSearch, self._paramReplace)
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
                        if self._verbose:print "Replace %s.%s %s ==> %s " % (base, name, value, repl)


    @staticmethod
    def standardizeInputTagFmt(inputTag):
       ''' helper function to ensure that the InputTag is defined as cms.InputTag(str) and not as a plain str '''
       if not isinstance(inputTag, cms.InputTag):
          return cms.InputTag(inputTag)
       return inputTag

    def enter(self,visitee):
        label = ''
        try:    label = visitee.label_()
        except AttributeError: label = '<Module not in a Process>'
        self.doIt(visitee, label)
    def leave(self,visitee):
        pass

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
    def __init__(self, process, label, postfix, removePostfix=""):
        self._process = process
        self._postfix = postfix
        self._removePostfix = removePostfix
        self._moduleLabels = []
        self._clonedSequence = cms.Sequence()
        setattr(process, self._newLabel(label), self._clonedSequence)

    def enter(self, visitee):
        if isinstance(visitee, cms._Module):
            label = visitee.label()
            newModule = None
            if label in self._moduleLabels: # has the module already been cloned ?
                newModule = getattr(self._process, self._newLabel(label))
            else:
                self._moduleLabels.append(label)
                newModule = visitee.clone()
                setattr(self._process, self._newLabel(label), newModule)
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
                raise StandardError("Tried to remove postfix %s from label %s, but it wasn't there" % (self._removePostfix, label))
        return label + self._postfix

    def __appendToTopSequence(self, visitee):
        self._clonedSequence += visitee

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


def massSearchReplaceParam(sequence,paramName,paramOldValue,paramValue,verbose=False):
    sequence.visit(MassSearchReplaceParamVisitor(paramName,paramOldValue,paramValue,verbose))

def listModules(sequence):
    visitor = GatherAllModulesVisitor(gatheredInstance=cms._Module)
    sequence.visit(visitor)
    return visitor.modules()

def listSequences(sequence):
    visitor = GatherAllModulesVisitor(gatheredInstance=cms.Sequence)
    sequence.visit(visitor)
    return visitor.modules()

def massSearchReplaceAnyInputTag(sequence, oldInputTag, newInputTag,verbose=False,moduleLabelOnly=False) :
    """Replace InputTag oldInputTag with newInputTag, at any level of nesting within PSets, VPSets, VInputTags..."""
    sequence.visit(MassSearchReplaceAnyInputTagVisitor(oldInputTag,newInputTag,verbose=verbose,moduleLabelOnly=moduleLabelOnly))

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



def cloneProcessingSnippet(process, sequence, postfix, removePostfix=""):
   """
   ------------------------------------------------------------------
   copy a sequence plus the modules and sequences therein
   both are renamed by getting a postfix
   input tags are automatically adjusted
   ------------------------------------------------------------------
   """
   result = sequence
   if not postfix == "":
       visitor = CloneSequenceVisitor(process, sequence.label(), postfix, removePostfix)
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
           cloneProcessingSnippet(p, p.s, "New")
           self.assertEqual(p.dumpPython(),'import FWCore.ParameterSet.Config as cms\n\nprocess = cms.Process("test")\n\nprocess.a = cms.EDProducer("a",\n    src = cms.InputTag("gen")\n)\n\n\nprocess.c = cms.EDProducer("c",\n    src = cms.InputTag("b","instance")\n)\n\n\nprocess.cNew = cms.EDProducer("c",\n    src = cms.InputTag("bNew","instance")\n)\n\n\nprocess.bNew = cms.EDProducer("b",\n    src = cms.InputTag("aNew")\n)\n\n\nprocess.aNew = cms.EDProducer("a",\n    src = cms.InputTag("gen")\n)\n\n\nprocess.b = cms.EDProducer("b",\n    src = cms.InputTag("a")\n)\n\n\nprocess.s = cms.Sequence(process.a*process.b*process.c*process.a)\n\n\nprocess.sNew = cms.Sequence(process.aNew+process.bNew+process.cNew)\n\n\n')
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
           self.assertEqual(jetCollectionString(algo = 'Foo', type = 'Bar'), 'patFooBarJets')
           self.assertEqual(jetCollectionString(prefix = 'prefix', algo = 'Foo', type = 'Bar'), 'prefixPatFooBarJets')
       def testListModules(self):
           p = cms.Process("test")
           p.a = cms.EDProducer("a", src=cms.InputTag("gen"))
           p.b = cms.EDProducer("ab", src=cms.InputTag("a"))
           p.c = cms.EDProducer("ac", src=cms.InputTag("b"))
           p.s = cms.Sequence(p.a*p.b*p.c)
           self.assertEqual([p.a,p.b,p.c], listModules(p.s))
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

   unittest.main()
