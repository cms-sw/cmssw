import logging

from Vispa.Main.BasicDataAccessor import *
from Vispa.Main.Exceptions import exception_traceback

class FindAlgorithm(object):
    """ Searches for label and properties in a list of objects using a BasicDataAccessor.
    
    One can search using findUsingFindDialog where a FindDialog is used as input for the search label and properties.
    Navigation through the results in supported by next(), previous(),numberOfResult(),currentNumber()
    """
    def __init__(self):
        self._dataAccessor = None
        self._dataObjects = []
        self._results=[]
        self._index=0
        self._message=None

    def setDataAccessor(self, accessor):
        logging.debug(__name__ + ": setDataAccessor")
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        self._dataAccessor = accessor
    
    def dataAccessor(self):
        return self._dataAccessor

    def setDataObjects(self, objects):
        logging.debug(__name__ + ": setDataObjects")
        self._dataObjects = objects
        
    def dataObjects(self):
        return self._dataObjects
    
    def findUsingFindDialog(self, dialog):
        logging.debug(__name__ +': findUsingFindDialog')
        self._message=None
        self._results=[]
        if self._dataAccessor:
            for object in self._dataObjects:
                self._results+=self._findIn(object,dialog)
        self._index=0
        if len(self._results)>0:
            return self._results[0]
        else:
            return None
        
    def _findIn(self, object,dialog):
        # find Label
        label=self._dataAccessor.label(object)
        logging.debug(__name__ +': _findIn: ' + label)
        findLabel=dialog.label()
        if not dialog.caseSensitive():
            label=label.lower()
            findLabel=findLabel.lower()
        if dialog.exactMatch():
            foundLabel=findLabel=="" or findLabel==label
        else:
            foundLabel=findLabel in label

        # find property
        properties=[(p[1],p[2]) for p in self._dataAccessor.properties(object)]
        findProperties=dialog.properties()
        if not dialog.caseSensitive():
            properties=[(property[0].lower(),property[1].lower()) for property in properties]
            findProperties=[(property[0].lower(),property[1].lower()) for property in findProperties]
        if dialog.exactMatch():
            foundProperties=True
            for findProperty in findProperties:
                foundProperties=(foundProperties and\
                    True in [(findProperty[0]=="" or findProperty[0]==p[0]) and\
                             (findProperty[1]=="" or findProperty[1]==p[1]) for p in properties])
        else:
            foundProperties=True
            for findProperty in findProperties:
                foundProperties=(foundProperties and\
                    True in [findProperty[0] in p[0] and\
                             findProperty[1] in p[1] for p in properties])

        # find property
        findScripts=dialog.scripts()
        foundScripts=True
        dataAccessorObject=BasicDataAccessorInterface(object,self._dataAccessor)
        for findScript in findScripts:
            try:
                foundScripts=(foundScripts and\
                    (findScript=="" or dataAccessorObject.applyScript(findScript)))
            except Exception,e:
                logging.info("Error in script: "+ exception_traceback())
                self._message="Error in script: "+ str(e)

        # combine the searches
        found=foundLabel and foundProperties and foundScripts
        if found:
            results=[object]
        else:
            results=[]
        for daughter in self._dataAccessor.children(object):
            for object in self._findIn(daughter,dialog):
                if not object in results:
                    results+=[object]
        return results

    def results(self):
        return self._results
    
    def numberOfResults(self):
        return len(self._results)
    
    def currentNumber(self):
        return self._index+1
    
    def next(self):
        if len(self._results)==0:
            return None
        self._index+=1
        if self._index>len(self._results)-1:
            self._index=0
        return self._results[self._index]
    
    def previous(self):
        if len(self._results)==0:
            return None
        self._index-=1
        if self._index<0:
            self._index=len(self._results)-1
        return self._results[self._index]

    def message(self):
        return self._message
