import FWCore.ParameterSet.Config as cms

## Helpers to perform some technically boring tasks like looking for all modules with a given parameter
## and replacing that to a given value

class MassSearchReplaceParamVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and replaces its value"""
    def __init__(self,paramName,paramSearch,paramValue):
        self._paramName   = paramName
        self._paramValue  = paramValue
        self._paramSearch = paramSearch
    def enter(self,visitee):
        if (hasattr(visitee,self._paramName)):
            if getattr(visitee,self._paramName) == self._paramSearch:
                print "Replaced %s.%s: %s => %s" % (visitee,self._paramName,getattr(visitee,self._paramName),self._paramValue)
                setattr(visitee,self._paramName,self._paramValue)
    def leave(self,visitee):
        pass

class MassSearchReplaceAnyInputTagVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and replace its value
       It will climb down within PSets, VPSets and VInputTags to find its target"""
    def __init__(self,paramSearch,paramReplace):
        self._paramSearch  = self.standardizeInputTagFmt(paramSearch)
        self._paramReplace = self.standardizeInputTagFmt(paramReplace)
        self._moduleName   = ''
    def doIt(self,pset,base):
        if isinstance(pset, cms._Parameterizable):
            for name in pset.parameters_().keys():
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
                            print "Replace %s.%s[%d] %s ==> %s " % (base, name, i, self._paramSearch, self._paramReplace)
                            value[i] = self._paramReplace
                elif type == 'cms.InputTag':
                    if value == self._paramSearch:
                        print "Replace %s.%s %s ==> %s " % (base, name, self._paramSearch, self._paramReplace)
                        setattr(pset, name, self._paramReplace)

    @staticmethod 
    def standardizeInputTagFmt(inputTag):
       ''' helper function to ensure that the InputTag is defined as cms.InputTag(str) and not as a plain str '''
       if not isinstance(inputTag, cms.InputTag):
          return cms.InputTag(inputTag)
       return inputTag

    def enter(self,visitee):
        label = ''
        try:    label = visitee.label()
        except AttributeError: label = '<Module not in a Process>'
        self.doIt(visitee, label)
    def leave(self,visitee):
        pass

class GatherAllModulesVisitor(object):
    """Visitor that travels within a cms.Sequence, and returns a list of modules that have it"""
    def __init__(self):
        self._modules = []
    def enter(self,visitee):
        if isinstance(visitee,cms._Module):
            self._modules.append(visitee)
    def leave(self,visitee):
        pass
    def modules(self):
        return self._modules
 

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
    
def massSearchReplaceParam(sequence,paramName,paramOldValue,paramValue):
    sequence.visit(MassSearchReplaceParamVisitor(paramName,paramOldValue,paramValue))

def listModules(sequence):
    visitor = GatherAllModulesVisitor()
    sequence.visit(visitor)
    return visitor.modules()

def massSearchReplaceAnyInputTag(sequence, oldInputTag, newInputTag) : 
    """Replace InputTag oldInputTag with newInputTag, at any level of nesting within PSets, VPSets, VInputTags..."""
    sequence.visit(MassSearchReplaceAnyInputTagVisitor(oldInputTag,newInputTag))
    
def jetCollectionString(prefix='', algo='AK5', type='Calo'):
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
    jetCollectionString+=algo
    jetCollectionString+=type
    jetCollectionString+='Jets'
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



def cloneProcessingSnippet(process, sequence, postfix):
   """
   ------------------------------------------------------------------
   copy a sequence plus the modules therein 
   the modules are renamed by getting a postfix
   input tags are automatically adjusted
   ------------------------------------------------------------------
   """
   # retrieve modules from old sequence 
   modules = listModules(sequence)
   # clone modules and put them into a new sequence
   labels = []
   tmpSequence = None
   for module in modules:
       label = module.label() 
       if label not in labels:
          labels.append(label) 
          newModule = module.clone()  
          setattr(process, label+postfix,newModule)
          if tmpSequence:    
             tmpSequence += newModule      
          else:
             tmpSequence = newModule  
   # create a new sequence
   newSequence = cms.Sequence(tmpSequence)
   newSequenceLabel = sequence.label()+postfix
   if hasattr(process, newSequenceLabel):
       raise AtributeError("Cloning the sequence "+sequence.label()+" would overwrite existing object." )
   else:
       setattr(process, newSequenceLabel, newSequence)
   # adjust all input tags
   for label in labels:
       massSearchReplaceAnyInputTag(newSequence, label, label+postfix)
   # return the new Sequence
   return newSequence


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
           p.c = cms.EDProducer("c", src=cms.InputTag("b"))
           p.s = cms.Sequence(p.a*p.b*p.c *p.a)
           cloneProcessingSnippet(p, p.s, "New")
           self.assertEqual(p.dumpPython(),'import FWCore.ParameterSet.Config as cms\n\nprocess = cms.Process("test")\n\nprocess.a = cms.EDProducer("a",\n    src = cms.InputTag("gen")\n)\n\n\nprocess.c = cms.EDProducer("c",\n    src = cms.InputTag("b")\n)\n\n\nprocess.cNew = cms.EDProducer("c",\n    src = cms.InputTag("bNew")\n)\n\n\nprocess.bNew = cms.EDProducer("b",\n    src = cms.InputTag("aNew")\n)\n\n\nprocess.aNew = cms.EDProducer("a",\n    src = cms.InputTag("gen")\n)\n\n\nprocess.b = cms.EDProducer("b",\n    src = cms.InputTag("a")\n)\n\n\nprocess.s = cms.Sequence(process.a*process.b*process.c*process.a)\n\n\nprocess.sNew = cms.Sequence(process.aNew+process.bNew+process.cNew)\n\n\n')

   unittest.main()
