import ROOT
import os.path
ROOT.gSystem.Load("libEgammaAnalysisElectronTools")

class Run2PhotonCalibrator:
    def __init__(self, data, isMC, isSync=False):
        self.random = ROOT.TRandom3()
        self.random.SetSeed(0) # make it really random across different jobs
        self.photonEnergyCalibratorRun2 = ROOT.PhotonEnergyCalibratorRun2(isMC, isSync, data)
        self.photonEnergyCalibratorRun2.initPrivateRng(self.random)

    def correct(self,photon,run):
        photon.uncalibratedP4 = photon.p4(photon.getCandidateP4type())
        photon.uncalibratedP4Error = photon.getCorrectedEnergyError(photon.getCandidateP4type())
        self.photonEnergyCalibratorRun2.calibrate(photon.physObj, int(run))
        return True
