from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from ROOT import TH1F

class Histogrammer(Analyzer):

    def beginLoop(self, setup):
        super(Histogrammer, self).beginLoop(setup)
        servname = '_'.join(['PhysicsTools.HeppyCore.framework.services.tfile.TFileService',
                             self.cfg_ana.file_label
                         ]) 
        tfileservice = setup.services[servname]
        tfileservice.file.cd()
        self.hist = TH1F("hist", "an histogram", 200, 0, 200)
        
    def process(self, event):
        self.hist.Fill(event.iEv)
    
