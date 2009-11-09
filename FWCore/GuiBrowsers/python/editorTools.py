from FWCore.GuiBrowsers.ConfigToolBase import ConfigToolBase

class UserCodeTool(ConfigToolBase):
    """ User code tool """
    _label="User code"
    def dumpPython(self):
        return ("",self._parameters['code'].value)
    def __call__(self,code):
        self.addParameter(self._parameters,'code',code, 'User code modifying the process: e.g. process.maxevents=1')
        return self
    def apply(self,process):
        code=self._parameters['code'].value
        exec code
    def typeError(self,name,bool):
        pass
