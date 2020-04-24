from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR

class GenHeavyFlavourAnalyzer( Analyzer ):
    """
       Makes:
          event.genallbquarks, event.genallcquarks:
                list of all b and c quarks (without doublecounting of b->b and c->c chains).
                if status2Only == True, only status 2 ones are included.
          event.allBPartons
                all status 2 b-quarks, sorted by pt decreasingly
          event.bPartons:
                status 2 b-quarks passing a pt cut bquarkPtCut (default: 15)

       Requires:
          event.genParticles      
       """

    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(GenHeavyFlavourAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.status2Only = cfg_ana.status2Only 
        self.bquarkPtCut = cfg_ana.bquarkPtCut
 
    def declareHandles(self):
        super(GenHeavyFlavourAnalyzer, self).declareHandles()

    def beginLoop(self,setup):
        super(GenHeavyFlavourAnalyzer,self).beginLoop(setup)

    def makeBPartons(self,event):
        event.allBPartons = [ q for q in event.genParticles if abs(q.pdgId()) == 5 and abs(q.status()) == 2 and abs(q.pt()) > self.bquarkPtCut ]
        event.allBPartons.sort(key = lambda q : q.pt(), reverse = True)
        event.bPartons = []
        for q in event.allBPartons:
            duplicate = False
            for q2 in event.bPartons:
                if deltaR(q.eta(),q.phi(),q2.eta(),q2.phi()) < 0.5:
                    duplicate = True
            continue
            if not duplicate: event.bPartons.append(q)

    def process(self, event):
        self.readCollections( event.input )

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        status2f = (lambda p : p.status() == 2) if self.status2Only else (lambda p : True)
        event.genallcquarks = [ p for p in event.genParticles if abs(p.pdgId()) == 5 and ( p.numberOfDaughters() == 0 or abs(p.daughter(0).pdgId()) != 5) and status2f(p) ]
        event.genallbquarks = [ p for p in event.genParticles if abs(p.pdgId()) == 4 and ( p.numberOfDaughters() == 0 or abs(p.daughter(0).pdgId()) != 4) and status2f(p) ]

        self.makeBPartons(event)

        return True

import PhysicsTools.HeppyCore.framework.config as cfg
setattr(GenHeavyFlavourAnalyzer,"defaultConfig",
    cfg.Analyzer(GenHeavyFlavourAnalyzer,
        status2Only = False,
        bquarkPtCut = 15.0,
   )
)
