import FWCore.ParameterSet.Config as cms

## Helpers to perform some technically boring tasks like looking for all modules with a given parameter
## and replacing that to a given value

# Next two lines are for backward compatibility, the imported functions and
# classes used to be defined in this file.
from FWCore.ParameterSet.MassReplace import massSearchReplaceAnyInputTag, MassSearchReplaceAnyInputTagVisitor
from FWCore.ParameterSet.MassReplace import massSearchReplaceParam, MassSearchParamVisitor, MassSearchReplaceParamVisitor

def applyPostfix(process, label, postfix):
    ''' If a module is in patHeavyIonDefaultSequence use the cloned module.
    Will crash if patHeavyIonDefaultSequence has not been cloned with 'postfix' beforehand'''
    result = None 
    defaultLabels = __labelsInSequence(process, "patHeavyIonDefaultSequence", postfix)
    if hasattr(process, "patPF2PATSequence"):
        defaultLabels = __labelsInSequence(process, "patPF2PATSequence", postfix)
    if label in defaultLabels and hasattr(process, label+postfix):
        result = getattr(process, label+postfix)
    elif hasattr(process, label):
        print "WARNING: called applyPostfix for module/sequence %s which is not in patHeavyIonDefaultSequence%s!"%(label,postfix)
        result = getattr(process, label)    
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
    def __init__(self, process, label, postfix):
        self._process = process
        self._postfix = postfix
        self._sequenceStack = [label]
        self._moduleLabels = []
        self._sequenceLabels = []
        self._waitForSequenceToClose = None # modules will only be cloned or added if this is None

    def enter(self,visitee):
        if not self._waitForSequenceToClose is None:
            return #we are in a already cloned sequence
        if isinstance(visitee,cms._Module):
            label = visitee.label()
            newModule = None
            if label in self._moduleLabels:
                newModule = getattr(self._process, label+self._postfix)
            else:
                self._moduleLabels.append(label)
                
                newModule = visitee.clone()
                setattr(self._process, label+self._postfix, newModule)
            self.__appendToTopSequence(newModule)

        if isinstance(visitee,cms.Sequence):
            if visitee.label() in self._sequenceLabels: # is the sequence allready cloned?
                self._waitForSequenceToClose = visitee.label()
                self._sequenceStack.append(  getattr(self._process, visitee.label()+self._postfix) )
            else:
                self._sequenceStack.append(visitee.label())#save desired label as placeholder until we have a module to create the sequence

    def leave(self,visitee):
        if isinstance(visitee,cms.Sequence):
            if self._waitForSequenceToClose == visitee.label():
                self._waitForSequenceToClose = None
            if not isinstance(self._sequenceStack[-1], cms.Sequence):
                raise Exception("empty Sequence encountered during cloneing. sequnece stack: %s"%self._sequenceStack)
            self.__appendToTopSequence( self._sequenceStack.pop() )

    def clonedSequence(self):
        if not len(self._sequenceStack) == 1:
            raise Exception("someting went wrong, the sequence stack looks like: %s"%self._sequenceStack)
        for label in self._moduleLabels:
            massSearchReplaceAnyInputTag(self._sequenceStack[-1], label, label+self._postfix, moduleLabelOnly=True, verbose=False)
        self._moduleLabels = [] #prevent the InputTag replacement next time this is called.
        return self._sequenceStack[-1]

    def __appendToTopSequence(self, visitee):#this is darn ugly because empty cms.Sequences are not supported
        if isinstance(self._sequenceStack[-1], str):#we have the name of an empty sequence on the stack. create it!
            oldSequenceLabel = self._sequenceStack.pop()
            newSequenceLabel = oldSequenceLabel + self._postfix
            self._sequenceStack.append(cms.Sequence(visitee))
            if hasattr(self._process, newSequenceLabel):
                raise Exception("Cloning the sequence "+self._sequenceStack[-1].label()+" would overwrite existing object." )
            setattr(self._process, newSequenceLabel, self._sequenceStack[-1])
            self._sequenceLabels.append(oldSequenceLabel)
        else:
            self._sequenceStack[-1] += visitee
        
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

    sequence : sequence [e.g. process.patHeavyIonDefaultSequence]
    module   : module name as a string
    ------------------------------------------------------------------    
    """
    return not sequence.__str__().find(moduleName)==-1    



def cloneProcessingSnippet(process, sequence, postfix):
   """
   ------------------------------------------------------------------
   copy a sequence plus the modules and sequences therein 
   both are renamed by getting a postfix
   input tags are automatically adjusted
   ------------------------------------------------------------------
   """
   result = sequence
   if not postfix == "":
       visitor = CloneSequenceVisitor(process,sequence.label(),postfix)
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
           
   unittest.main()
