import FWCore.ParameterSet.Config as cms

## Helpers to perform some technically boring tasks like looking for all modules with a given parameter
## and replacing that to a given value

class MassSearchReplaceParamVisitor(object):
    """Visitor that travels within a cms.Sequence, looks for a parameter and replace its value"""
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
 
