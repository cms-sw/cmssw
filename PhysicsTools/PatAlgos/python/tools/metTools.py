from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

class AddMETCollection(ConfigToolBase):
    """
    Tool to add alternative MET collection(s) to your PAT Tuple
    """
    _label='addMETCollection'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        """
        Initialize elements of the class. Note that the tool needs to be derived from ConfigToolBase
        to be usable in the configEditor.
        """
        ## initialization of the base class
        ConfigToolBase.__init__(self)
        ## add all parameters that should be known to the class
        self.addParameter(self._defaultParameters,'labelName',self._defaultValue, "Label name of the new patMET collection.", str)
        self.addParameter(self._defaultParameters,'metSource',self._defaultValue, "Label of the input collection from which the new patMet collection should be created.", str)
        ## set defaults
        self._parameters=copy.deepcopy(self._defaultParameters)
        ## add comments
        self._comment = "Add alternative MET collections as PAT object to your PAT Tuple"

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,labelName=None,metSource=None):
        """
        Function call wrapper. This will check the parameters and call the actual implementation that
        can be found in toolCode via the base class function apply.
        """
        if labelName is None:
            labelName=self._defaultParameters['labelName'].value
        self.setParameter('labelName', labelName)
        if metSource is None:
            metSource=self._defaultParameters['metSource'].value
        self.setParameter('metSource', metSource)
        self.apply(process)

    def toolCode(self, process):
        """
        Tool code implementation
        """
        ## initialize parameters
        labelName=self._parameters['labelName'].value
        metSource=self._parameters['metSource'].value
        ## do necessary imports
        from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
        ## add module to the process
        task = getPatAlgosToolsTask(process)
        addToProcessAndTask(labelName, patMETs.clone(metSource = metSource, addMuonCorrections=False), process, task)

        ## add module to output
        if hasattr(process, "out"):
            process.out.outputCommands+=["keep *_{LABEL_NAME}_*_*".format(LABEL_NAME=labelName)]

addMETCollection=AddMETCollection()
