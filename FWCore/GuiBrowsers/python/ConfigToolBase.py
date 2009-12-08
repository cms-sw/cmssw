import copy
import FWCore.ParameterSet.Config as cms

class parameter:
    pass


### Base class for tools object oriented designed 
        
class ConfigToolBase(object) :

    """ Base class for PAT tools
    """
    _label="ConfigToolBase"
    _defaultValue="No default value. Set parameter value."
    def __init__(self):
        self._parameters={}
        self._description=self.__doc__
        self._comment = ''
        self.parAccepted=True
    def __call__(self,process):
        """ Call the instance 
        """
        raise NotImplementedError
    
    def apply(self,process):
        
        if hasattr(process, "addAction"):
            process.disableRecording()
            
        self.toolCode(process)
        
        if hasattr(process, "addAction"):
            process.enableRecording()
            action=self.__copy__()
            process.addAction(action)
            
    def toolCode(self, process):
        pass

            
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
    def addParameter(self,dict,parname, parvalue, description,Type=None, allowedValues=None):
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
        dict[par.name]=par        
    def getParameters(self):
        """ Return the list of the parameters of an action.

        Each parameters is represented by a tuple containing its
        type, name, value and description.
        The type determines how the parameter is represented in the GUI.
        Possible types are: 'Category','String','Text','File','FileVector','Boolean','Integer','Float'.
        """
        return copy.deepcopy(self._parameters)
    def setParameter(self, name, value, typeNone=False):
        """ Change parameter 'name' to a new value
        """
        self._parameters[name].value=value
        ### check about input value type 
        self.typeError(name,typeNone )
        ### check about input value (it works if allowedValues for the specific parameter is set)
        if self._defaultParameters[name].allowedValues is not None: self.isAllowed(name,value )
    def setParameters(self, parameters):
        self._parameters=copy.deepcopy(parameters)
    def dumpPython(self):
        """ Return the python code to perform the action
        """
        raise NotImplementedError
    def setComment(self, comment):
        """ Write a comment in the configuration file
        """
        self._comment = comment
    def errorMessage(self,value,type):
        return "The type for parameter "+'"'+str(value)+'"'+" is not "+'"'+str(type)+'"'
    ### method isAllowed is called by setParameter to check input values for a specific parameter
    def isAllowed(self,name,value):
        self.parAccepted=True
        if value==[]:
            self.parAccepted=False
        elif (isinstance(value,dict)) and (isinstance(self._parameters[name].allowedValues,list)):
            
            for key in value.keys():
                if key not in self._parameters[name].allowedValues:
                    raise ValueError("The input key value "+'"'+str(key)+'"'+" for parameter "+'"'+name+'"'+" is not supported. Supported ones are: "+str(self._parameters[name].allowedValues))
        elif (isinstance(value,list)) and (isinstance(self._parameters[name].allowedValues,list )):
            for i in value:
                if i not in self._parameters[name].allowedValues:
                   self.parAccepted=False
        elif (not isinstance(value,list))and (isinstance(self._parameters[name].allowedValues,list)):
            if value not in self._parameters[name].allowedValues:
                self.parAccepted=False
        elif not isinstance(self._parameters[name].allowedValues,list):
            if value!=self._parameters[name].allowedValues:
              self.parAccepted=False  
        if self.parAccepted==False:
            raise ValueError("The input value "+'"'+str(value)+'"'+" for parameter "+'"'+name+'"'+" is not supported. Supported ones are: "+str(self._parameters[name].allowedValues)[1:-1])
    ### check about input value type        
    def typeError(self,name, bool=False):
        if bool is False:
            if not isinstance(self._parameters[name].value,self._parameters[name].type):
                raise TypeError(self.errorMessage(self._parameters[name].value,self._parameters[name].type))
        else:
            if not (isinstance(self._parameters[name].value,self._parameters[name].type) or self._parameters[name].value is None):
                raise TypeError(self.errorMessage(self._parameters[name].value,self._parameters[name].type))
            
