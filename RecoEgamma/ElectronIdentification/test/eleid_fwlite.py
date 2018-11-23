import ROOT
ROOT.gROOT.SetBatch()

import numpy as np

class ElectronMVAID:

    def __init__(self, name, tag, flavor, *xmls):
        self.name = name
        self.tag = tag
        self.flavor = flavor
        self.sxmls = ROOT.vector(ROOT.string)()
        for x in xmls: self.sxmls.push_back(x)
        self._init = False

    def __call__(self, ele, convs, beam_spot, rho, debug=False):
        print(type(rho[0]))
        print(rho[0])
        if not self._init:
            print 'initializing electron mva id'
            ROOT.gSystem.Load("libRecoEgammaElectronIdentification")
            debug = False
            variableDefinition = 'RecoEgamma/ElectronIdentification/data/ElectronMVAEstimatorRun2Variables.txt'
            categoryCutStrings_List = [
     "pt < 10. && abs(superCluster.eta) < 0.800", # EB1_5
     "pt < 10. && abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479", # EB2_5
     "pt < 10. && abs(superCluster.eta) >= 1.479", # EE_5
     "pt >= 10. && abs(superCluster.eta) < 0.800", # EB1_10
     "pt >= 10. && abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479", # EB2_10
     "pt >= 10. && abs(superCluster.eta) >= 1.479", # EE_10
     ]
            categoryCutStrings =  ROOT.vector(ROOT.string)()
            for x in categoryCutStrings_List : 
                categoryCutStrings.push_back(x)
        # self.estimator = ROOT.ElectronMVAEstimatorRun2(self.tag, self.name, self.sxmls, len(self.sxmls), debug, variableDefinition, categoryCutStrings)
            self.estimator = ROOT.ElectronMVAEstimatorRun2(
                self.tag, self.name, len(self.sxmls), 
                variableDefinition, categoryCutStrings, debug
                )
            self.estimator.init(self.sxmls)
            self._init = True
        extra_vars = self.estimator.getExtraVars(ele, convs, beam_spot, rho[0])
        for x in extra_vars:
            print(x)
        return self.estimator.mvaValue(ele, extra_vars)

eleid_Fall17IsoV2 = ElectronMVAID(
    "ElectronMVAEstimatorRun2Fall17V2","V2","Iso",
    "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2/EB1_5.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2/EB2_5.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2/EE_5.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2/EB1_10.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2/EB2_10.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2/EE_10.weights.xml.gz"
)

from DataFormats.FWLite import Events, Handle

print 'open input file...'

events = Events('root://cms-xrd-global.cern.ch//store/mc/RunIIFall17MiniAODv2/VBFHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/549B2DC8-C443-E811-9DAF-A0369FE2C17C.root')
# events = Events('test.root')

ele_handle  = Handle('std::vector<pat::Electron>')
rho_handle  = Handle('double')
conv_handle = Handle('reco::ConversionCollection')
bs_handle   = Handle('reco::BeamSpot')

print 'start processing' 
accepted = 0
for i,event in enumerate(events): 

    event.getByLabel(('slimmedElectrons'),                 ele_handle)
    event.getByLabel(('fixedGridRhoFastjetAll'),           rho_handle)
    event.getByLabel(('reducedEgamma:reducedConversions'), conv_handle)
    event.getByLabel(('offlineBeamSpot'),                  bs_handle)

    electrons = ele_handle.product()
    convs     = conv_handle.product()
    beam_spot = bs_handle.product()
    rho       = rho_handle.product()

    if not len(electrons):
        continue
    accepted += 1
    print accepted
    mva = eleid_Fall17IsoV2(electrons[0], convs, beam_spot, rho )
    print 'mva=', mva
    if accepted==100: 
        break

