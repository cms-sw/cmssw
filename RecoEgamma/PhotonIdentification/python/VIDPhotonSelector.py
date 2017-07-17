import ROOT
from PhysicsTools.SelectorUtils.VIDSelectorBase import VIDSelectorBase

class VIDPhotonSelector(VIDSelectorBase):
    def __init__(self,pythonpset = None):
        builder  = ROOT.MakeVersionedSelector(ROOT.reco.Photon)
        ptrmaker = ROOT.MakePtrFromCollection(ROOT.vector(ROOT.pat.Photon),
                                              ROOT.pat.Photon,
                                              ROOT.reco.Photon)
        printer  = ROOT.PrintVIDToString(ROOT.reco.Photon)
        VIDSelectorBase.__init__(self,builder,ptrmaker,printer,pythonpset)
        
