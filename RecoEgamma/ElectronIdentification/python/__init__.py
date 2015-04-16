#definition of pythonized VIDElectron class
import os
__path__.append(os.path.dirname(os.path.abspath(__file__).rsplit('/RecoEgamma/ElectronIdentification/',1)[0])+'/cfipython/slc6_amd64_gcc491/RecoEgamma/ElectronIdentification')

import ROOT
from PhysicsTools.SelectorUtils import VIDSelectorBase

class VIDElectronSelector(VIDSelectorBase):
    def __init__(self,pythonpset = None):
        builder  = ROOT.MakeVersionedSelector(ROOT.reco.GsfElectron)
        ptrmaker = ROOT.MakePtrFromCollection(ROOT.vector(ROOT.pat.Electron),
                                              ROOT.pat.Electron,
                                              ROOT.reco.GsfElectron)
        printer  = ROOT.PrintVIDToString(ROOT.reco.GsfElectron)
        VIDSelectorBase.__init__(self,builder,ptrmaker,printer,pythonpset)
        
