import PhysicsTools.Heppy.loadlibs
import ROOT

class KalmanMuonCorrector:
    def __init__(self, calibration, isMC, isSync=False, smearMode="none"):
        self.kamuca = ROOT.KalmanMuonCalibrator(calibration)
        self.isMC = isMC
        self.isSync = isSync
        self.smearMode = smearMode
    def correct(self, mu, run):
        newPt = self.kamuca.getCorrectedPt(mu.pt(), mu.eta(), mu.phi(), mu.charge())
        newPtErr = newPt * self.kamuca.getCorrectedError(newPt, mu.eta(), mu.ptErr()/newPt)
        if self.isMC: # new we do the smearing
            if self.isSync:
                newPt = self.kamuca.smearForSync(newPt, mu.eta())
                newPtErr = newPt * self.kamuca.getCorrectedErrorAfterSmearing(newPt, mu.eta(), newPtErr/newPt)
            elif self.smearMode == "none" or self.smearMode == None:
                pass
            elif self.smearMode == "basic":
                newPt = self.kamuca.smear(newPt, mu.eta())
                newPtErr = newPt * self.kamuca.getCorrectedErrorAfterSmearing(newPt, mu.eta(), newPtErr/newPt)
            else:
                newPt = self.kamuca.smearUsingEbE(newPt, mu.eta(), newPtErr/newPt)
                newPtErr = newPt * self.kamuca.getCorrectedErrorAfterSmearing(newPt, mu.eta(), newPtErr/newPt)
        newP4 = ROOT.math.PtEtaPhiMLorentzVector(newPt, mu.eta(), mu.phi(), mu.mass())
        mu.setP4(newP4)
        mu._ptErr = newPtErr

    def correct_all(self, mus, run):
        for mu in mus:
            self.correct(mu, run)

if __name__ == '__main__':
    kamuka = KalmanMuonCorrector("MC_76X_13TeV", True)
