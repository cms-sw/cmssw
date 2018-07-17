import copy
import inspect
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Mixins import *
from os import path

#### patches needed for deepcopy of sorted dicts ####

import FWCore.ParameterSet.DictTypes as dicttypes
    
def new_SortedKeysDict__copy__(self):
    return self.__class__(self)
dicttypes.SortedKeysDict.__copy__ = new_SortedKeysDict__copy__

def new_SortedKeysDict__deepcopy__(self, memo=None):
    from copy import deepcopy
    if memo is None:
        memo = {}
    d = memo.get(id(self), None)
    if d is not None:
        return d
    memo[id(self)] = d = self.__class__()
    d.__init__(deepcopy(self.items(), memo))
    return d
dicttypes.SortedKeysDict.__deepcopy__ = new_SortedKeysDict__deepcopy__


class parameter:
    pass

### Base class for object oriented designed tools
        
class ConfigToolBase(object) :

    """ Base class for PAT tools
    """
    _label="ConfigToolBase"
    _defaultValue="No default value. Set parameter value."
    _path = ""
    def __init__(self):
        self._parameters=dicttypes.SortedKeysDict()
        self._description=self.__doc__
        self._comment = ''
        self.parAccepted=True
        saveOrigin(self,1)
        self._path = path.realpath(self._filename)        
        self._path = self._path.split("/src/")
        self._path = self._path[1].replace("/python","")
        #self._path = "".join(self._path)
        self._path = self._path.replace("/",".")
        self._path = self._path.replace(".py","")


    def __call__(self,process):
        """ Call the instance 
        """
        raise NotImplementedError
    
    def apply(self,process):
        
        if hasattr(process, "addAction"):
            process.disableRecording()
            
        try:
            comment=inspect.stack(2)[2][4][0].rstrip("\n")
            if comment.startswith("#"):
                self.setComment(comment.lstrip("#"))
        except:
            pass
            
        self.toolCode(process)
        
        if hasattr(process, "addAction"):
            process.enableRecording()
            action=self.__copy__()
            process.addAction(action)
            
    def toolCode(self, process):
        raise NotImplementedError

            
    ### __copy__(self) returns a copy of the tool
    def __copy__(self):
        c=type(self)()
        c.setParameters(copy.deepcopy(self._parameters))
        c.setComment(self._comment)
        return c
    def reset(self):
        self._parameters=copy.deepcopy(self._defaultParameters)
    def getvalue(self,name):
        """ Return the value of parameter 'name'
        """
        return self._parameters[name].value
    def description(self):
        """ Return a string with a detailed description of the action.
        """
        return self._description
    
    ### use addParameter method in the redefinition of tool constructor in order to add parameters to the tools
    ### each tool is defined by its label, default value, description, type and allowedValues (the last two attribute can be ignored
    ### if the user gives a valid default values and if there is not a list of allowed values)
    def addParameter(self,dict,parname, parvalue, description,Type=None, allowedValues=None, acceptNoneValue=False):
        """ Add a parameter with its label, value, description and type to self._parameters
        """
        par=parameter()
        par.name=parname
        par.value=parvalue
        par.description=description
        if Type==None:
            par.type=type(parvalue)
        else: par.type=Type
        par.allowedValues=allowedValues
        par.acceptNoneValue=acceptNoneValue
        dict[par.name]=par        
    def getParameters(self):
        """ Return a copy of the dict of the parameters.
        """
        return copy.deepcopy(self._parameters)
    def setParameter(self, name, value, typeNone=False):
        """ Change parameter 'name' to a new value
        """
        self._parameters[name].value=value
        ### check about input value type 
        self.typeError(name)
        ### check about input value (it works if allowedValues for the specific parameter is set)
        if self._defaultParameters[name].allowedValues is not None: self.isAllowed(name,value )
    def setParameters(self, parameters):
        self._parameters=copy.deepcopy(parameters)
    #def dumpPython(self):
     #   """ Return the python code to perform the action
     #   """
     #   raise NotImplementedError

    def dumpPython(self):
        """ Return the python code to perform the action
        """ 
        dumpPythonImport = "\nfrom "+self._path+" import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython += "\n"+self._label+"(process "
        for key in self._parameters.keys():
	  if repr(self._parameters[key].value)!=repr(self._defaultParameters[key].value):
            dumpPython+= ", "+str(key)+" = "
            if self._parameters[key].type is str:
                string = "'"+str(self.getvalue(key))+"'"
            else:
                string = str(self.getvalue(key))
            dumpPython+= string
        dumpPython+=")"+'\n'
        return (dumpPythonImport,dumpPython)
    
    def setComment(self, comment):
        """ Write a comment in the configuration file
        """
        self._comment = str(comment)
    def comment(self):
        """ Return the comment set for this tool
        """
        return self._comment
    def errorMessage(self,value,type):
        return "The type for parameter "+'"'+str(value)+'"'+" is not "+'"'+str(type)+'"'
    ### method isAllowed is called by setParameter to check input values for a specific parameter
    def isAllowed(self,name,value):
        self.parAccepted=True
        if value==[]:
            self.parAccepted=False
        elif (isinstance(value,dict)) and (isinstance(self._parameters[name].allowedValues,list)):
            for key in value.keys():
                if (key not in self._parameters[name].allowedValues):
                    raise ValueError("The input key value "+'"'+str(key)+'"'+" for parameter "+'"'+name+'"'+" is not supported. Supported ones are: "+str(self._parameters[name].allowedValues)[1:-1])
        elif (isinstance(value,list)) and (isinstance(self._parameters[name].allowedValues,list )):
            for val in value:
                if (val not in self._parameters[name].allowedValues) :
                    raise ValueError("The input value "+'"'+str(val)+'"'+" for parameter "+'"'+name+'"'+" is not supported. Supported ones are: "+str(self._parameters[name].allowedValues)[1:-1])
        elif (not isinstance(value,list))and (isinstance(self._parameters[name].allowedValues,list)) :
            if (value not in self._parameters[name].allowedValues and value == None) and (not self._parameters[name].acceptNoneValue) :
                self.parAccepted=False
        elif not isinstance(self._parameters[name].allowedValues,list):
            if (value!=self._parameters[name].allowedValues and value == None) and (not self._parameters[name].acceptNoneValue) :
                self.parAccepted=False  
        if self.parAccepted==False:
            raise ValueError("The input value "+'"'+str(value)+'"'+" for parameter "+'"'+name+'"'+" is not supported. Supported ones are: "+str(self._parameters[name].allowedValues)[1:-1])
    ### check about input value type        
    def typeError(self,name):
        if self._parameters[name].acceptNoneValue is False:
            if not isinstance(self._parameters[name].value,self._parameters[name].type):
                raise TypeError(self.errorMessage(self._parameters[name].value,self._parameters[name].type))
        else:
            if not (isinstance(self._parameters[name].value,self._parameters[name].type) or self._parameters[name].value is  None):
                raise TypeError(self.errorMessage(self._parameters[name].value,self._parameters[name].type))
    def getAllowedValues(self,name):
        return self._defaultParameters[name].allowedValues
