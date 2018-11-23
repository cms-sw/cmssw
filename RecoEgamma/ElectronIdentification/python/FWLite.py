import ROOT

# Python wrappers around the Electron MVAs.
# Usage example in RecoEgamma/ElectronIdentification/test

class ElectronMVAID:
    """ Electron MVA wrapper class.
    """

    def __init__(self, name, tag, categoryCuts, xmls, variablesFile, debug=False):
        self.name = name
        self.tag = tag
        self.categoryCuts = categoryCuts
        self.variablesFile = variablesFile
        self.xmls = ROOT.vector(ROOT.string)()
        for x in xmls: self.xmls.push_back(x)
        self._init = False
        self._debug = debug

    def __call__(self, ele, convs, beam_spot, rho, debug=False):
        if not self._init:
            print('Initializing ' + self.name + self.tag)
            ROOT.gSystem.Load("libRecoEgammaElectronIdentification")
            categoryCutStrings =  ROOT.vector(ROOT.string)()
            for x in self.categoryCuts : 
                categoryCutStrings.push_back(x)
            self.estimator = ROOT.ElectronMVAEstimatorRun2(
                self.tag, self.name, len(self.xmls), 
                self.variablesFile, categoryCutStrings, self.xmls, self._debug)
            self._init = True
        extra_vars = self.estimator.getExtraVars(ele, convs, beam_spot, rho[0])
        return self.estimator.mvaValue(ele, extra_vars)

# Import information needed to construct the e/gamma MVAs

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import EleMVA_6CategoriesCuts, mvaVariablesFile, EleMVA_3CategoriesCuts

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff import mvaWeightFiles as Fall17_iso_V2_weightFiles
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff import mvaWeightFiles as Fall17_noIso_V2_weightFiles
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff import mvaSpring16WeightFiles_V1 as mvaSpring16GPWeightFiles_V1
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff import mvaSpring16WeightFiles_V1 as mvaSpring16HZZWeightFiles_V1

# Dictionary with the relecant e/gmma MVAs

ElectronMVAs = {
        "Fall17IsoV2"   : ElectronMVAID("ElectronMVAEstimatorRun2","Fall17IsoV2",
                                      EleMVA_6CategoriesCuts, Fall17_iso_V2_weightFiles, mvaVariablesFile),
        "Fall17NoIsoV2" : ElectronMVAID("ElectronMVAEstimatorRun2","Fall17NoIsoV2",
                                      EleMVA_6CategoriesCuts, Fall17_noIso_V2_weightFiles, mvaVariablesFile),
        "Spring16HZZV1" : ElectronMVAID("ElectronMVAEstimatorRun2","Spring16HZZV1",
                                      EleMVA_6CategoriesCuts, mvaSpring16HZZWeightFiles_V1, mvaVariablesFile),
        "Spring16V1"    : ElectronMVAID("ElectronMVAEstimatorRun2","Spring16GeneralPurposeV1",
                                      EleMVA_3CategoriesCuts, mvaSpring16GPWeightFiles_V1, mvaVariablesFile),
}
