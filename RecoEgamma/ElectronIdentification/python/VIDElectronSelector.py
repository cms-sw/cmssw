import ROOT
from PhysicsTools.SelectorUtils.VIDSelectorBase import VIDSelectorBase

class VIDElectronSelector(VIDSelectorBase):
    def __init__(self,pythonpset = None):
        builder  = ROOT.MakeVersionedSelector(ROOT.reco.GsfElectron)
        ptrmaker = ROOT.MakePtrFromCollection(ROOT.vector(ROOT.pat.Electron),
                                              ROOT.pat.Electron,
                                              ROOT.reco.GsfElectron)
        printer  = ROOT.PrintVIDToString(ROOT.reco.GsfElectron)
        VIDSelectorBase.__init__(self,builder,ptrmaker,printer,pythonpset)
        
