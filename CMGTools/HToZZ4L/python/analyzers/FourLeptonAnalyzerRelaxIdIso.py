from math import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzer import *

        
class FourLeptonAnalyzerRelaxIdIso( FourLeptonAnalyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(FourLeptonAnalyzerRelaxIdIso,self).__init__(cfg_ana,cfg_comp,looperName)
        self.tag = cfg_ana.tag
    def declareHandles(self):
        super(FourLeptonAnalyzerRelaxIdIso, self).declareHandles()

    def beginLoop(self, setup):
        super(FourLeptonAnalyzerRelaxIdIso,self).beginLoop(setup)
        self.counters.addCounter('FourLepton')
        count = self.counters.counter('FourLepton')
        count.register('all events')


    #For the good lepton preselection redefine the thingy so that leptons are loose    
    def leptonID(self,lepton):
        return self.leptonID_loose(lepton)


    def fourLeptonIsolation(self,fourLepton):
        return True        
