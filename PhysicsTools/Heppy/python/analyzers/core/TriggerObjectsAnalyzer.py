import ROOT

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import NTupleVariable
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection, matchObjectCollection3
import PhysicsTools.HeppyCore.framework.config as cfg

""" FindTrigger: check if HLT path triggerName is names (using wildcard)"""
def FindTrigger(triggerName,names):
    if triggerName=="": return True
    triggerName = triggerName.replace("*","")
    triggerName = triggerName.replace("$","")
    for name in names:
        if triggerName in name: return True
    return False

""" 
triggerCollection is a class that interprets the triggerObjectsCfgs elements.
Example:
trColl = triggerCollection(("hltMet","","HLT"),"hltMET90","HLT_PFMET90_PFMHT90_IDTight*")
It gives:
trColl.collectionText   ="hltMet::HLT"
trColl.collectionName   ="hltMet"
trColl.filterName       ="hltMET90"
trColl.path             ="HLT_PFMET90_PFMHT90_IDTight*"
"""
class triggerCollection(object):
    def __init__(self, triggerObjectsCfg):
        collectionInfo = triggerObjectsCfg[0]
        collectionLabel = collectionInfo[0] if len(collectionInfo)>0 else ""
        collectionInstance = collectionInfo[1] if len(collectionInfo)>1 else ""
        collectionProcess = collectionInfo[2] if len(collectionInfo)>2 else "HLT"
        self.collectionText = collectionLabel + ":" + collectionInstance + ":" + collectionProcess

        self.collectionName = collectionLabel 
        if collectionInstance != "": self.collectionName = self.collectionName + "_" + collectionInstance
        if collectionProcess != "HLT": self.collectionName = self.collectionName + "_" + collectionProcess

        self.filterName = triggerObjectsCfg[1] if len(triggerObjectsCfg)>1 else ""
        self.path = triggerObjectsCfg[2] if len(triggerObjectsCfg)>2 else ""

"""
TriggerObjectsAnalyzer is a class that saves the trigger objects matching the parameters defined in triggerObjectsCfgs.
Example
triggerObjectsCfgs = {"caloMet":(("hltMet","","HLT"),"hltMET90","HLT_PFMET90_PFMHT90_IDTight*") )}
means:
    - caloMet is just a name
    - ("hltMet","","HLT") is the label, instance and process name of the EDProducer that produced the trigger objects
    - hltMET90 is the label of the EDFilter that saves the trigger objects
    - HLT_PFMET90_PFMHT90_IDTight* is the name of the HLT path that we require was running the EDFilter module
"""
class TriggerObjectsAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TriggerObjectsAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        triggerObjectsCfgs = getattr(self.cfg_ana,"triggerObjectsCfgs",[])
        self.triggerObjectInputTag  = getattr(self.cfg_ana,"triggerObjectInputTag",("","",""))
        self.triggerBitsInputTag    = getattr(self.cfg_ana,"triggerBitsInputTag",("","",""))
        self.triggerObjectsInfos    = {}
        self.runNumber              = -1
        self.names                  = None
        for collectionName in triggerObjectsCfgs.keys():
            self.triggerObjectsInfos[collectionName] = triggerCollection(triggerObjectsCfgs[collectionName])
        
    def declareHandles(self):
        super(TriggerObjectsAnalyzer, self).declareHandles()

        triggerBitsTagFallback = (self.triggerBitsInputTag[0],
                                  self.triggerBitsInputTag[1],
                                  "HLT2")

        self.handles['TriggerBits'] = AutoHandle( self.triggerBitsInputTag, 
                                                  'edm::TriggerResults', 
                                                  fallbackLabel = triggerBitsTagFallback )
        
        self.handles['TriggerObjects']  = AutoHandle( self.triggerObjectInputTag, 
                                                      'std::vector<pat::TriggerObjectStandAlone>',
                                                      fallbackLabel = 'HLT2')

    def beginLoop(self, setup):
        super(TriggerObjectsAnalyzer,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        run = event.input.eventAuxiliary().id().run()
        # get the trigger names (only for the first event of each run)
        if self.runNumber!= run:
            triggerBits = self.handles['TriggerBits'].product()
            self.names = event.input.object().triggerNames(triggerBits)
            self.runNumber = run
        # get the trigger object
        allTriggerObjects = self.handles['TriggerObjects'].product()
        
        # init objects
        objects = {}
        for collectionName in self.triggerObjectsInfos.keys():
            objects[collectionName]=[]
        # for each collection name save the trigger object matching the triggerObjectsInfo
        for ob in allTriggerObjects:
            for collectionName in self.triggerObjectsInfos.keys():
                triggerObjectsInfo = self.triggerObjectsInfos[collectionName]
                if (triggerObjectsInfo.collectionText!="::HLT") and triggerObjectsInfo.collectionText!=ob.collection(): continue
                if (triggerObjectsInfo.path!=""):
                    if not hasattr(ob,"unpacked"):
                        ob.unpacked = True
                        ob.unpackPathNames(self.names)
                    if not FindTrigger(triggerObjectsInfo.path, ob.pathNames()): continue
                if (triggerObjectsInfo.filterName!="") and not (ob.hasFilterLabel(triggerObjectsInfo.filterName)): continue
                objects[collectionName].append(ob)
        # add object variables in event
        for collectionName in self.triggerObjectsInfos.keys():
            setattr(event,'trgObjects_'+collectionName,objects[collectionName])

setattr(TriggerObjectsAnalyzer,"defaultConfig",cfg.Analyzer(
    TriggerObjectsAnalyzer, name="TriggerObjectsAnalyzerDefault",
    triggerObjectsCfgs = {"caloJets":(("hltAK4CaloJetsCorrectedIDPassed")),"caloMet":(("hltMet","","HLT"),"hltMET90","HLT_PFMET90_PFMHT90_IDTight*")},
    triggerObjectInputTag = ('selectedPatTrigger','','PAT'),
    triggerBitsInputTag = ('TriggerResults','','HLT')
)
)


