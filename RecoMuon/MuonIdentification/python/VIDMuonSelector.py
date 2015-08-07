import ROOT
from PhysicsTools.SelectorUtils.VIDSelectorBase import VIDSelectorBase

class VIDMuonSelector(VIDSelectorBase):
    def __init__(self,pythonpset = None):
        builder  = ROOT.MakeVersionedSelector(ROOT.reco.Muon)
        ptrmaker = ROOT.MakePtrFromCollection(ROOT.vector(ROOT.pat.Muon),
                                              ROOT.pat.Muon,
                                              ROOT.reco.Muon)
        printer  = ROOT.PrintVIDToString(ROOT.reco.Muon)
        VIDSelectorBase.__init__(self,builder,ptrmaker,printer,pythonpset)
        
