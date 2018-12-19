import ROOT
import ctypes
import pprint
from numpy import exp

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
        '''returns a tuple mva_value, category 
        ele: a reco::GsfElectron
        convs: conversions
        beam_spot: beam spot
        rho: energy density in the event
        debug: enable debugging mode. 

        example: 
        
            event.getByLabel(('slimmedElectrons'),                 ele_handle)
            event.getByLabel(('fixedGridRhoFastjetAll'),           rho_handle)
            event.getByLabel(('reducedEgamma:reducedConversions'), conv_handle)
            event.getByLabel(('offlineBeamSpot'),                  bs_handle)
            
            electrons = ele_handle.product()
            convs     = conv_handle.product()
            beam_spot = bs_handle.product()
            rho       = rho_handle.product()

            mva, category = electron_mva_id(electron[0], convs, beam_spot, rho)
        '''
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
        category = ctypes.c_int(0)
        mva = self.estimator.mvaValue(ele, extra_vars, category)
        return mva, category.value


class WorkingPoints(object):
    '''Working Points. Keeps track of the cuts associated to a given flavour of the MVA ID 
    for each working point and allows to test the working points'''

    def __init__(self, name, tag, working_points, logistic_transform=False):
        self.name = name 
        self.tag = tag
        self.working_points = self._reformat_cut_definitions(working_points)
        self.logistic_transform = logistic_transform

    def _reformat_cut_definitions(self, working_points):
        new_definitions = dict()
        for wpname, definitions in working_points.iteritems():
            new_definitions[wpname] = dict()
            for name, cut in definitions.cuts.iteritems():
                categ_id = int(name.lstrip('cutCategory'))
                cut = cut.replace('pt','x')
                formula = ROOT.TFormula('_'.join([self.name, wpname, name]), cut)
                new_definitions[wpname][categ_id] = formula
        return new_definitions

    def passed(self, ele, mva, category, wp):
        '''return true if ele passes wp'''
        threshold = self.working_points[wp][category].Eval(ele.pt())
        if self.logistic_transform:
            mva = 2.0/(1.0+exp(-2.0*mva))-1
        return mva > threshold


# Import information needed to construct the e/gamma MVAs

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools \
        import EleMVA_6CategoriesCuts, mvaVariablesFile, EleMVA_3CategoriesCuts

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff \
        import mvaWeightFiles as Fall17_iso_V2_weightFiles
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff \
        import mvaWeightFiles as Fall17_noIso_V2_weightFiles
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff \
        import mvaSpring16WeightFiles_V1 as mvaSpring16GPWeightFiles_V1
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff \
        import mvaSpring16WeightFiles_V1 as mvaSpring16HZZWeightFiles_V1

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff \
        import workingPoints as mvaSpring16GP_V1_workingPoints
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff \
        import workingPoints as mvaSpring16HZZ_V1_workingPoints
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff \
        import workingPoints as Fall17_iso_V2_workingPoints
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff \
        import workingPoints as Fall17_noIso_V2_workingPoints

# Dictionary with the relecant e/gmma MVAs

electron_mvas = {
    "Fall17IsoV2"   : ElectronMVAID("ElectronMVAEstimatorRun2","Fall17IsoV2",
                                    EleMVA_6CategoriesCuts, Fall17_iso_V2_weightFiles, mvaVariablesFile),
    "Fall17NoIsoV2" : ElectronMVAID("ElectronMVAEstimatorRun2","Fall17NoIsoV2",
                                    EleMVA_6CategoriesCuts, Fall17_noIso_V2_weightFiles, mvaVariablesFile),
    "Spring16HZZV1" : ElectronMVAID("ElectronMVAEstimatorRun2","Spring16HZZV1",
                                    EleMVA_6CategoriesCuts, mvaSpring16HZZWeightFiles_V1, mvaVariablesFile),
    "Spring16GPV1"    : ElectronMVAID("ElectronMVAEstimatorRun2","Spring16GeneralPurposeV1",
                                    EleMVA_3CategoriesCuts, mvaSpring16GPWeightFiles_V1, mvaVariablesFile),
    }

working_points = {
    "Fall17IsoV2"   : WorkingPoints("ElectronMVAEstimatorRun2","Fall17IsoV2",
                                    Fall17_iso_V2_workingPoints),
    "Fall17NoIsoV2" : WorkingPoints("ElectronMVAEstimatorRun2","Fall17NoIsoV2",
                                    Fall17_noIso_V2_workingPoints),
    "Spring16HZZV1" : WorkingPoints("ElectronMVAEstimatorRun2","Spring16HZZV1",
                                    mvaSpring16HZZ_V1_workingPoints, logistic_transform=True),
    "Spring16GPV1"    : WorkingPoints("ElectronMVAEstimatorRun2","Spring16GeneralPurposeV1",
                                    mvaSpring16GP_V1_workingPoints, logistic_transform=True),

    }
