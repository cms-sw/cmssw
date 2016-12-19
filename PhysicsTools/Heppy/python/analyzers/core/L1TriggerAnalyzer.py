"""
L1TriggerAnalyzer is an Analyzer that load in-time (BX=0) L1 objects (jets, tau, muons, EGamma).
SumEt objects (MET,MHT,ET,HT) are defined with the PtPhiPair defined above.
"""

import ROOT

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection, matchObjectCollection3
import PhysicsTools.HeppyCore.framework.config as cfg


class PtPhiPair():
    def __init__(self,pt=-1,phi=-10):
        self.pt_ = pt
        self.phi_ = phi
    def pt(self):
        return self.pt_
    def phi(self):
        return self.phi_

class L1TriggerAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(L1TriggerAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.processName  = getattr(self.cfg_ana,"processName",("HLT"))
        self.l1JetInputTag  = getattr(self.cfg_ana,"l1JetInputTag",("caloStage2Digis","Jet",self.processName))
        self.l1TauInputTag  = getattr(self.cfg_ana,"l1JetInputTag",("caloStage2Digis","Tau",self.processName))
        self.l1EGammaInputTag  = getattr(self.cfg_ana,"l1JetInputTag",("caloStage2Digis","EGamma",self.processName))
        self.l1EtSumInputTag  = getattr(self.cfg_ana,"l1JetInputTag",("caloStage2Digis","EtSum",self.processName))
        self.l1MuonInputTag  = getattr(self.cfg_ana,"l1JetInputTag",("gmtStage2Digis","Muon",self.processName))
        
    def declareHandles(self):
        super(L1TriggerAnalyzer, self).declareHandles()
        self.handles['l1tJets']  = AutoHandle( self.l1JetInputTag, 'BXVector<l1t::Jet>' )
        self.handles['l1tTaus']  = AutoHandle( self.l1TauInputTag, 'BXVector<l1t::Tau>' )
        self.handles['l1tEGammas']  = AutoHandle( self.l1EGammaInputTag, 'BXVector<l1t::EGamma>' )
        self.handles['l1tEtSums']  = AutoHandle( self.l1EtSumInputTag, 'BXVector<l1t::EtSum>' )
        self.handles['l1tMuons']  = AutoHandle( self.l1MuonInputTag, 'BXVector<l1t::Muon>' )
        self.validL1handles = True

    def beginLoop(self, setup):
        super(L1TriggerAnalyzer,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        event.l1Jets = []
        event.l1Taus = []
        event.l1Muons = []
        event.l1EGammas = []
        event.l1MET2,event.l1MET,event.l1ET,event.l1MHT,event.l1HT = tuple([PtPhiPair()]*5)
        
        if self.validL1handles:
            try:
                for i in range(self.handles['l1tJets'].product().size(0)):
                    l1Jet = self.handles['l1tJets'].product().at(0,i)
                    event.l1Jets.append(l1Jet)
                
                for i in range(self.handles['l1tTaus'].product().size(0)):
                    l1Tau = self.handles['l1tTaus'].product().at(0,i)
                    event.l1Taus.append(l1Tau)
                
                for i in range(self.handles['l1tMuons'].product().size(0)):
                    l1Muon = self.handles['l1tMuons'].product().at(0,i)
                    event.l1Muons.append(l1Muon)
                
                for i in range(self.handles['l1tEGammas'].product().size(0)):
                    l1EGamma = self.handles['l1tEGammas'].product().at(0,i)
                    event.l1EGammas.append(l1EGamma)
                
                for i in range(self.handles['l1tEtSums'].product().size(0)):
                    l1Obj = self.handles['l1tEtSums'].product().at(0,i)
                    l1ObjType = l1Obj.getType()
                    if l1ObjType == l1Obj.kMissingEt2:
                        event.l1MET2 = PtPhiPair(l1Obj.et(),l1Obj.phi())
                    elif l1ObjType == l1Obj.kMissingEt:
                        event.l1MET = PtPhiPair(l1Obj.et(),l1Obj.phi())
                    elif l1ObjType == l1Obj.kMissingHt:
                        event.l1MHT = PtPhiPair(l1Obj.et(),l1Obj.phi())
                    elif l1ObjType == l1Obj.kTotalEt:
                        event.l1ET = PtPhiPair(l1Obj.et(),l1Obj.phi())
                    elif l1ObjType == l1Obj.kTotalHt:
                        event.l1HT = PtPhiPair(l1Obj.et(),l1Obj.phi())
            except:
                self.validL1handles = False
                pass

setattr(L1TriggerAnalyzer,"defaultConfig",cfg.Analyzer(
    L1TriggerAnalyzer, name="L1TriggerAnalyzerDefault",
    processName = "HLT"
)
)


