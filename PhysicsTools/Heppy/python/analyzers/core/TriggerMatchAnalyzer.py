import ROOT

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import NTupleVariable
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection, matchObjectCollection3
import PhysicsTools.HeppyCore.framework.config as cfg
        
class TriggerMatchAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TriggerMatchAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.processName = getattr(self.cfg_ana,"processName","PAT")
        self.fallbackName = getattr(self.cfg_ana,"fallbackProcessName","RECO")
        self.unpackPathNames = getattr(self.cfg_ana,"unpackPathNames",True)
        self.label = self.cfg_ana.label
        self.trgObjSelectors = []
        self.trgObjSelectors.extend(getattr(self.cfg_ana,"trgObjSelectors",[]))
        self.collToMatch = getattr(self.cfg_ana,"collToMatch",None)
        self.collMatchSelectors = []
        self.collMatchSelectors.extend(getattr(self.cfg_ana,"collMatchSelectors",[]))
        self.collMatchDRCut = getattr(self.cfg_ana,"collMatchDRCut",0.3)
        if self.collToMatch and not hasattr(self.cfg_ana,"univoqueMatching"): raise RuntimeError("Please specify if the matching to trigger objects should be 1-to-1 or 1-to-many")
        self.match1To1 = getattr(self.cfg_ana,"univoqueMatching",True)

    def declareHandles(self):
        super(TriggerMatchAnalyzer, self).declareHandles()
        self.handles['TriggerBits'] = AutoHandle( ('TriggerResults','','HLT'), 'edm::TriggerResults' )
        fallback = ( 'selectedPatTrigger','', self.fallbackName) if self.fallbackName else None
        self.handles['TriggerObjects'] = AutoHandle( ('selectedPatTrigger','',self.processName), 'std::vector<pat::TriggerObjectStandAlone>', fallbackLabel=fallback )

    def beginLoop(self, setup):
        super(TriggerMatchAnalyzer,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        triggerBits = self.handles['TriggerBits'].product()
        allTriggerObjects = self.handles['TriggerObjects'].product()
        names = event.input.object().triggerNames(triggerBits)
        for ob in allTriggerObjects: ob.unpackPathNames(names)
        triggerObjects = [ob for ob in allTriggerObjects if False not in [sel(ob) for sel in self.trgObjSelectors]]

        setattr(event,'trgObjects_'+self.label,triggerObjects)

        if self.collToMatch:
            tcoll = getattr(event,self.collToMatch)
            doubleandselector = lambda lep,ob: False if False in [sel(lep,ob) for sel in self.collMatchSelectors] else True
            pairs = matchObjectCollection3(tcoll,triggerObjects,deltaRMax=self.collMatchDRCut,filter=doubleandselector) if self.match1To1 else matchObjectCollection(tcoll,triggerObjects,self.collMatchDRCut,filter=doubleandselector)
            for lep in tcoll: setattr(lep,'matchedTrgObj'+self.label,pairs[lep])

        if self.verbose:
            print 'Verbose debug for triggerMatchAnalyzer %s'%self.label
            for ob in getattr(event,'trgObjects_'+self.label):
                types = ", ".join([str(f) for f in ob.filterIds()])
                filters = ", ".join([str(f) for f in ob.filterLabels()])
                paths = ", ".join([("%s***" if f in set(ob.pathNames(True)) else "%s")%f for f in ob.pathNames()]) # asterisks indicate final paths fired by this object, see pat::TriggerObjectStandAlone class
                print 'Trigger object: pt=%.2f, eta=%.2f, phi=%.2f, collection=%s, type_ids=%s, filters=%s, paths=%s'%(ob.pt(),ob.eta(),ob.phi(),ob.collection(),types,filters,paths)
            if self.collToMatch:
                for lep in tcoll:
                    mstring = 'None'
                    ob = getattr(lep,'matchedTrgObj'+self.label)
                    if ob: mstring = 'trigger obj with pt=%.2f, eta=%.2f, phi=%.2f, collection=%s'%(ob.pt(),ob.eta(),ob.phi(),ob.collection())
                    print 'Lepton pt=%.2f, eta=%.2f, phi=%.2f matched to %s'%(lep.pt(),lep.eta(),lep.phi(),mstring)

        return True


setattr(TriggerMatchAnalyzer,"defaultConfig",cfg.Analyzer(
    TriggerMatchAnalyzer, name="TriggerMatchAnalyzerDefault",
    label='DefaultTrigObjSelection',
    processName = 'PAT',
    fallbackProcessName = 'RECO',
    unpackPathNames = True,
    trgObjSelectors = [],
    collToMatch = None,
    collMatchSelectors = [],
    collMatchDRCut = 0.3,
    univoqueMatching = True,
    verbose = False
)
)


