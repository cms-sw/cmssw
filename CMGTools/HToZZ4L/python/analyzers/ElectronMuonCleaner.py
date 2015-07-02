from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR


       
class ElectronMuonCleaner( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ElectronMuonCleaner,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ElectronMuonCleaner, self).declareHandles()

    def beginLoop(self, setup):
        super(ElectronMuonCleaner,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )

        muons  = [ mu for mu in event.selectedLeptons if abs(mu.pdgId())==13 and self.cfg_ana.selectedMuCut(mu) ]
        muons += [ mu for mu in event.otherLeptons    if abs(mu.pdgId())==13 and self.cfg_ana.otherMuCut(mu)    ]
        
        selectedElectrons = [ ]
        selectedLeptons = [ mu for mu in event.selectedLeptons if abs(mu.pdgId())==13  ]
        for ele in event.selectedLeptons:
            if abs(ele.pdgId()) != 11: continue
            good = True
            for mu in muons:
                dr = deltaR(mu.eta(), mu.phi(), ele.eta(), ele.phi())
                if self.cfg_ana.mustClean(ele,mu,dr):
                    good = False
                    break
            if good:
                selectedLeptons.append(ele) 
            else: # move to the discarded ones
                event.otherLeptons.append(ele) 

        # re-sort
        selectedLeptons.sort(key = lambda l : l.pt(), reverse = True)
        selectedElectrons.sort(key = lambda l : l.pt(), reverse = True)
        event.otherLeptons.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedLeptons = selectedLeptons
        event.selectedElectrons = selectedElectrons

        return True
        
