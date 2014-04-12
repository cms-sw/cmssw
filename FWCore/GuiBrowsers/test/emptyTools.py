from FWCore.GuiBrowsers.ConfigToolBase import *

class ToolName(ConfigToolBase):

    """
    Tool description
    """
    _label='toolName'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        ### import base class constructor
        ConfigToolBase.__init__(self)
        ### Add each tool parameter with addParameter method
        self.addParameter(self._defaultParameters,'parName',parDefaultValue, parDescription, Type=parType, allowedValues=supportedValues, acceptNoneValue=isNoneallowed)
        self.addParameter(self._defaultParameters,'parName2',parDefaultValue, parDescription, Type=parType, allowedValues=supportedValues, acceptNoneValue=isNoneallowed)
        ### create parameter set starting from default one
        self._parameters=copy.deepcopy(self._defaultParameters)
        ### tool comment, set it by using setComment
        ### it will be included in dump python code
        self._comment = ""
    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 parName1     = None,
                 parName2     = None) :
        ### set deafult parameter values to None
        if  parName is None:
            parName=self._defaultParameters['parName'].value
        if  parName2 is None:
            parName2=self._defaultParameters['parName2'].value
        ### set parameter values to ones input by the user
        ### the setParameter method provides checks about type and about values ( supported values and parameter type are specified in addParameter (Type, alloweValues) )
        self.setParameter('parName',parName)
        self.setParameter('parName2',parName)
        ### apply tool by calling method "apply" defined in the base class. It will call the toolCode method
        self.apply(process) 
        
    def toolCode(self, process):
        
        ### rename parameter names just to avoid change the old code        
        parName=self._parameters['parName'].value
        parName2=self._parameters['parName2'].value
     
      
        ### PUT HERE THE TOOL CODE

       
       

### instance of the tool class
### it allows the user to run old configuration files without change anything, 
### although the new tool structure is very different from previous one
toolName=ToolName()
