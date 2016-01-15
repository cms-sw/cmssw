import os 
from ROOT import heppy, TLorentzVector

class MuScleFitCorr:
    def __init__(self,isMC,isReReco,isSync=False):
        #colin need to import muscle fit inputs, if tool still in use
        path = "%s/src/CMGTools/RootTools/data/musclefit/" % os.environ['CMSSW_BASE'];
        self.isMC = isMC
        if self.isMC:
            self.corr = heppy.MuScleFitCorrector(path+"MuScleFit_2012_MC_53X_smear%s.txt" % ("ReReco" if isReReco else "Prompt"))
            self.isSync = isSync
        else:
            self.corrABC = heppy.MuScleFitCorrector(path+"MuScleFit_2012ABC_DATA%s_53X.txt" % ("_ReReco" if isReReco else ""))
            self.corrD   = heppy.MuScleFitCorrector(path+"MuScleFit_2012D_DATA%s_53X.txt"   % ("_ReReco" if isReReco else ""))
    def corrected_p4(self, mu, run):
        p4 = TLorentzVector(mu.px(), mu.py(), mu.pz(), mu.energy())
        if self.isMC:
            self.corr.applyPtCorrection(p4, mu.charge())
            self.corr.applyPtSmearing(p4, mu.charge(), self.isSync)
        else:
            corr = self.corrD if run >= 203773 else self.corrABC
            corr.applyPtCorrection(p4, mu.charge())
        ## convert to the proper C++ class (but preserve the mass!)
        return ROOT.reco.Muon.PolarLorentzVector( p4.Pt(), p4.Eta(), p4.Phi(), mu.mass() )
    def correct(self, mu, run):
        mu.setP4( self.corrected_p4(mu, run) )

if __name__ == '__main__':
    muscle = MuScleFitCorr(True, True)
