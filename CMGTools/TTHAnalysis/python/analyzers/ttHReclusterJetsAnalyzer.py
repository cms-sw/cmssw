from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from ROOT.heppy import ReclusterJets

import ROOT


class ttHReclusterJetsAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHReclusterJetsAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 
        self.pTSubJet = self.cfg_ana.pTSubJet  if hasattr(self.cfg_ana, 'pTSubJet') else 30.0
        self.etaSubJet = self.cfg_ana.etaSubJet  if hasattr(self.cfg_ana, 'etaSubJet') else 5.0

    def declareHandles(self):
        super(ttHReclusterJetsAnalyzer, self).declareHandles()
       #genJets                                                                                                                                                                     
        self.handles['genJets'] = AutoHandle( 'slimmedGenJets','std::vector<reco::GenJet>')

    def beginLoop(self, setup):
        super(ttHReclusterJetsAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')

    def makeFatJets(self, event):
       	objectsJ = [ j for j in event.cleanJetsAll if j.pt() > self.pTSubJet and abs(j.eta())<self.etaSubJet ] 
        if len(objectsJ)>=1:
            
           objects  = ROOT.std.vector(ROOT.reco.Particle.LorentzVector)()
           for jet in objectsJ:
                objects.push_back(jet.p4())
                
           reclusterJets = ReclusterJets(objects, 1.,1.2)
           inclusiveJets = reclusterJets.getGrouping()

           # maybe should dress them as Jet objects in the future
           # for the moment, we just make LorentzVector
           event.reclusteredFatJets = [ ROOT.reco.Particle.LorentzVector(p4) for p4 in inclusiveJets ]
           # note 1: just taking inclusiveJets is not ok, since it's not a python list but a std::vector
           # note 2: [p4 for p4 in inclusiveJets] is also bad, since these are references to values inside a temporary std::vector

    def process(self, event):
        self.readCollections( event.input )

        event.reclusteredFatJets = []
        self.makeFatJets(event)

        return True
