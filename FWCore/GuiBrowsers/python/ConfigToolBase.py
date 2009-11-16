import copy
import FWCore.ParameterSet.Config as cms

#import PhysicsTools.PatAlgos.Config as cms
class parameter:
    pass

        
class ConfigToolBase(object) :

    """ Base class for PAT tools
    """
    _label="ConfigToolBase"
    _defaultValue="No default value. Set parameter value."
    def __init__(self):
        self._parameters={}
        self._description=self.__doc__
        self._comment = ''
    def __call__(self):
        """ Call the istance 
        """
        raise NotImplementedError
    def __copy__(self):
        c=type(self)()
        c.setParameters(copy.deepcopy(self._parameters))
        c.setComment(self._comment)
        return c
    def setDefaultParameters(self):
        pass
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

    def addParameter(self,dict,parname, parvalue, description,Type=None):
        """ Add a parameter with its label, value, description and type to self._parameters
        """
        par=parameter()
        par.name=parname
        par.value=parvalue
        par.description=description
        if Type==None:
            par.type=type(parvalue)
        else: par.type=Type
        dict[par.name]=par
        
    def getParameters(self):
        """ Return the list of the parameters of an action.

        Each parameters is represented by a tuple containing its
        type, name, value and description.
        The type determines how the parameter is represented in the GUI.
        Possible types are: 'Category','String','Text','File','FileVector','Boolean','Integer','Float'.
        """
        return copy.deepcopy(self._parameters)
    def setParameter(self, name, value, bool=False):
        """ Change parameter 'name' to a new value
        """
        self._parameters[name].value=value
        self.typeError(name, bool)
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

    def typeError(self,name, bool=False):
        if bool is False:
            if not isinstance(self._parameters[name].value,self._parameters[name].type):
                raise TypeError(self.errorMessage(self._parameters[name].value,self._parameters[name].type))
        else:
            if not (isinstance(self._parameters[name].value,self._parameters[name].type) or self._parameters[name].value is None):
                raise TypeError(self.errorMessage(self._parameters[name].value,self._parameters[name].type))
            
