from FWCore.GuiBrowsers.ConfigToolBase import *

class ToolName(ConfigToolBase):

    """
    Tool description
    """
    _label='ToolName'
    _defaultParameters={}
    def __init__(self):
        ### import base class constructor
        ConfigToolBase.__init__(self)
        ### Add each tool parameter with addParameter method
        self.addParameter(self._defaultParameters,'parName',parDefaultValue, parDescription, Type=parType, Range=supportedValues)
        self.addParameter(self._defaultParameters,'parName2',parDefaultValue, parDescription, Type=parType, Range=supportedValues)
        ### create parameter set starting from default one
        self._parameters=copy.deepcopy(self._defaultParameters)
        ### tool comment, set it by using setComment
        ### it will be included in dump python code
        self._comment = ""
    def getDefaultParameters(self):
        return self._defaultParameters
    def dumpPython(self):
        ### returns the code to include in a config file to use the tool
        ### the dumpPythonImport variable is the import line to add only if it has not already included
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.trackTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\ntoolName(process, "
        dumpPython += '"'+str(self.getvalue('parName'))+'"'+", "
        dumpPython += '"'+str(self.getvalue('parName2'))+'"'+")"+'\n'
        return (dumpPythonImport,dumpPython) 

    def __call__(self,process,
                 parName1     = None,
                 parName2     = None) :
        ### set deafult parameter values to None
        if  parName is None:
            parName=self._defaultParameters['parName'].value
        if  parName2 is None:
            parName2=self._defaultParameters['parName2'].value
        ### set parameter values to ones input by the user
        ### the setParameter method provides checks about type and about values (if supported ones are specified in addParameter - Range )
        self.setParameter('parName',parName)
        self.setParameter('parName2',parName)
        ### apply tool by calling method apply
        self.apply(process) 
        
    def apply(self, process):
        ### rename parameter names just to avoid change the old code        
        parName=self._parameters['parName'].value
        parName2=self._parameters['parName2'].value
        ### put the following two line to make tool working even outside the ConfigEditor
        ### disableRecording() does not allow to store in the history tools called by the main one
        if hasattr(process, "addAction"):
            process.disableRecording()


        ### PUT HERE THE TOOL CODE

        ### enableRecording() restores the history recording
        if hasattr(process, "addAction"):
            process.enableRecording()
            ### copy the tool and add it to the history
            action=self.__copy__()
            process.addAction(action)

### instance of the tool class
### it allows the user to run old configuration files without change anything, 
### although the new tool structure is very different from previous one
toolName=ToolName()
