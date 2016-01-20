import ROOT
import os.path
ROOT.gSystem.Load("libEgammaAnalysisElectronTools")

class Run2ElectronCalibrator:
    def __init__(self, scales, smearings, gbrForest, isMC, isSync=False):
        self.epCombinationTool = ROOT.EpCombinationTool()
        self.epCombinationTool.init(os.path.expandvars(gbrForest[0]), gbrForest[1]) 
        self.random = ROOT.TRandom3()
        self.random.SetSeed(0) # make it really random across different jobs
        vscales = ROOT.std.vector('double')()
        for s in scales: vscales.push_back(s)
        vsmearings = ROOT.std.vector('double')()
        for s in smearings: vsmearings.push_back(s)
        self.electronEnergyCalibratorRun2 = ROOT.ElectronEnergyCalibratorRun2(self.epCombinationTool, isMC, isSync, vsmearings, vscales)
        self.electronEnergyCalibratorRun2.initPrivateRng(self.random)
 
    def correct(self,electron,run):
        if not electron.validCandidateP4Kind(): return False # these can't be calibrated
        electron.uncalibratedP4 = electron.p4()
        electron.uncalibratedP4Error = electron.p4Error(electron.candidateP4Kind())
        self.electronEnergyCalibratorRun2.calibrate(electron.physObj, int(run))
        return True

