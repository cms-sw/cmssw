import logging

from Vispa.Share.ObjectHolder import ObjectHolder
from Vispa.Share.BasicDataAccessor import BasicDataAccessor,BasicDataAccessorInterface
from Vispa.Main.Exceptions import exception_traceback

class FindAlgorithm(ObjectHolder):
    """ Searches for label and properties in a list of objects using a BasicDataAccessor.
    
    One can search using findUsingFindDialog where a FindDialog is used as input for the search label and properties.
    Navigation through the results in supported by next(), previous(),numberOfResult(),currentNumber()
    """
    def __init__(self):
        ObjectHolder.__init__(self)
        self._results=[]
        self._index=0
        self._message=None

    def setDataAccessor(self, accessor):
        logging.debug(__name__ + ": setDataAccessor")
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        ObjectHolder.setDataAccessor(self, accessor)
    
    def clear(self):
        self._message=None
        self._results=[]
            
    def findUsingFindDialog(self, dialog):
        logging.debug(__name__ +': findUsingFindDialog')
        self.clear()
        if self.dataAccessor():
            for object in self.dataObjects():
                self._results+=self._findIn(object,dialog)
        self._index=0
        if len(self._results)>0:
            return self._results[0]
        else:
            return []
        
    def _findIn(self, object,dialog):
        # find Label
        foundLabel=True
        findLabel=dialog.label()
        if findLabel!="":
            label=self.dataAccessor().label(object)
            #logging.debug(__name__ +': _findIn: ' + label)
            if not dialog.caseSensitive():
                label=label.lower()
                findLabel=findLabel.lower()
            if dialog.exactMatch():
                foundLabel=findLabel=="" or findLabel==label
            else:
                foundLabel=findLabel in label

        # find property
        foundProperties=True
        findProperties=dialog.properties()
        if len(findProperties)>0 and (findProperties[0][0]!="" or findProperties[0][1]!=""): 
            properties=[(p[1],p[2]) for p in self.dataAccessor().properties(object)]
            if not dialog.caseSensitive():
                properties=[(str(property[0]).lower(),str(property[1]).lower()) for property in properties]
                findProperties=[(str(property[0]).lower(),str(property[1]).lower()) for property in findProperties]
            if dialog.exactMatch():
                for findProperty in findProperties:
                    foundProperties=(foundProperties and\
                        True in [(findProperty[0]=="" or findProperty[0]==p[0]) and\
                                 (findProperty[1]=="" or findProperty[1]==p[1]) for p in properties])
            else:
                for findProperty in findProperties:
                    foundProperties=(foundProperties and\
                        True in [findProperty[0] in p[0] and\
                                 findProperty[1] in p[1] for p in properties])

        # find property
        findScripts=dialog.scripts()
        foundScripts=True
        if len(findScripts)>0 and findScripts[0]!="":
            dataAccessorObject=BasicDataAccessorInterface(object,self.dataAccessor())
            for findScript in findScripts:
                try:
                    foundScripts=(foundScripts and\
                       (findScript=="" or dataAccessorObject.runScript(findScript)))
                except Exception,e:
                    foundScripts=False
                    logging.info("Error in script: "+ exception_traceback())
                    self._message="Error in script: "+ str(e)

        # combine the searches
        found=foundLabel and foundProperties and foundScripts
        if found:
            results=[object]
        else:
            results=[]
        for daughter in self.applyFilter(self.dataAccessor().children(object)):
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
