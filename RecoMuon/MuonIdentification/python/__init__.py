#definition of pythonized VIDElectron class
import os
__path__.append(os.path.dirname(os.path.abspath(__file__).rsplit('/RecoMuon/MuonIdentification/',1)[0])+'/cfipython/slc6_amd64_gcc491/RecoMuon/MuonIdentification')

import ROOT
from PhysicsTools.SelectorUtils import VIDSelectorBase

class VIDMuonSelector(VIDSelectorBase):
    def __init__(self,pythonpset = None):
        builder  = ROOT.MakeVersionedSelector(ROOT.reco.Muon)
        ptrmaker = ROOT.MakePtrFromCollection(ROOT.vector(ROOT.pat.Muon),
                                              ROOT.pat.Muon,
                                              ROOT.reco.Muon)
        printer  = ROOT.PrintVIDToString(ROOT.reco.Muon)
        VIDSelectorBase.__init__(self,builder,ptrmaker,printer,pythonpset)
        
