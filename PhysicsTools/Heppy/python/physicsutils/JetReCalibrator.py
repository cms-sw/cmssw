import ROOT
import os
from math import *
from PhysicsTools.HeppyCore.utils.deltar import *

class JetReCalibrator:
    def __init__(self,globalTag,jetFlavour,doResidualJECs,jecPath,upToLevel=3):
        """Create a corrector object that reads the payloads from the text dumps of a global tag under
            CMGTools/RootTools/data/jec  (see the getJec.py there to make the dumps).
           It will apply the L1,L2,L3 and possibly the residual corrections to the jets."""
        # Make base corrections
        path = jecPath #"%s/src/CMGTools/RootTools/data/jec" % os.environ['CMSSW_BASE'];
        self.L1JetPar  = ROOT.JetCorrectorParameters("%s/%s_L1FastJet_%s.txt" % (path,globalTag,jetFlavour),"");
        self.L2JetPar  = ROOT.JetCorrectorParameters("%s/%s_L2Relative_%s.txt" % (path,globalTag,jetFlavour),"");
        self.L3JetPar  = ROOT.JetCorrectorParameters("%s/%s_L3Absolute_%s.txt" % (path,globalTag,jetFlavour),"");
        self.vPar = ROOT.vector(ROOT.JetCorrectorParameters)()
        self.vPar.push_back(self.L1JetPar);
        if upToLevel >= 2: self.vPar.push_back(self.L2JetPar);
        if upToLevel >= 3: self.vPar.push_back(self.L3JetPar);
        # Add residuals if needed
        if doResidualJECs : 
            self.ResJetPar = ROOT.JetCorrectorParameters("%s/%s_L2L3Residual_%s.txt" % (path,globalTag,jetFlavour))
            self.vPar.push_back(self.ResJetPar);
        #Step3 (Construct a FactorizedJetCorrector object) 
        self.JetCorrector = ROOT.FactorizedJetCorrector(self.vPar)
        if os.path.exists("%s/%s_Uncertainty_%s.txt" % (path,globalTag,jetFlavour)):
            self.JetUncertainty = ROOT.JetCorrectionUncertainty("%s/%s_Uncertainty_%s.txt" % (path,globalTag,jetFlavour));
        else:
            print 'Missing JEC uncertainty file "%s/%s_Uncertainty_%s.txt", so jet energy uncertainties will not be available' % (path,globalTag,jetFlavour)
            self.JetUncertainty = None
    def correctAll(self,jets,rho,delta=0,metShift=[0,0]):
        """Applies 'correct' to all the jets, discard the ones that have bad corrections (corrected pt <= 0)"""
        badJets = []
        for j in jets:
            ok = self.correct(j,rho,delta,metShift)
            if not ok: badJets.append(j)
        if len(badJets) > 0:
            print "Warning: %d out of %d jets flagged bad by JEC." % (len(badJets), len(jets))
        for bj in badJets:
            jets.remove(bj)
    def correct(self,jet,rho,delta=0,metShift=[0,0]):
        """Corrects a jet energy (optionally shifting it also by delta times the JEC uncertainty)
           If a two-component list is passes as 'metShift', it will be modified adding to the first and second
           component the change to the MET along x and y due to the JEC, defined as the negative difference 
           between the new and old jet 4-vectors, for jets with corrected pt > 10."""
        self.JetCorrector.setJetEta(jet.eta())
        self.JetCorrector.setJetPt(jet.pt() * jet.rawFactor())
        self.JetCorrector.setJetA(jet.jetArea())
        self.JetCorrector.setRho(rho)
        corr = self.JetCorrector.getCorrection()
        if delta != 0:
            if not self.JetUncertainty: raise RuntimeError, "Jet energy scale uncertainty shifts requested, but not available"
            self.JetUncertainty.setJetEta(jet.eta())
            self.JetUncertainty.setJetPt(corr * jet.pt() * jet.rawFactor())
            try:
                jet.jetEnergyCorrUncertainty = self.JetUncertainty.getUncertainty(True) 
            except RuntimeError, r:
                print "Caught %s when getting uncertainty for jet of pt %.1f, eta %.2f\n" % (r,corr * jet.pt() * jet.rawFactor(),jet.eta())
                jet.jetEnergyCorrUncertainty = 0.5
        if jet.photonEnergyFraction() < 0.9 and jet.pt()*corr*jet.rawFactor() > 10:
            metShift[0] -= jet.px()*(corr*jet.rawFactor() - 1)*(1-jet.muonEnergyFraction())
            metShift[1] -= jet.py()*(corr*jet.rawFactor() - 1)*(1-jet.muonEnergyFraction()) 
        if delta != 0:
            #print "   jet with corr pt %6.2f has an uncertainty %.2f " % (jet.pt()*jet.rawFactor()*corr, jet.jetEnergyCorrUncertainty)
            corr *= max(0, 1+delta*jet.jetEnergyCorrUncertainty)
            if jet.pt()*jet.rawFactor()*corr > 10:
                metShift[0] -= jet.px()*jet.rawFactor()*corr*delta*jet.jetEnergyCorrUncertainty
                metShift[1] -= jet.py()*jet.rawFactor()*corr*delta*jet.jetEnergyCorrUncertainty
        #print "   jet with raw pt %6.2f eta %+5.3f phi %+5.3f: previous corr %.4f, my corr %.4f " % (jet.pt()*jet.rawFactor(), jet.eta(), jet.phi(), 1./jet.rawFactor(), corr)
        if corr <= 0:
            return False
        jet.setP4(jet.p4() * (corr * jet.rawFactor()))
        jet.setRawFactor(1.0/corr)
        return True

class Type1METCorrection:
    def __init__(self,globalTag,jetFlavour,doResidualJECs):
        """Create a corrector object that reads the payloads from the text dumps of a global tag under
           CMGTools/RootTools/data/jec  (see the getJec.py there to make the dumps).
           It will make the Type1 MET."""
        # Make base corrections
        path = "%s/src/CMGTools/RootTools/data/jec" % os.environ['CMSSW_BASE'];
        self.L1JetPar  = ROOT.JetCorrectorParameters("%s/%s_L1FastJet_%s.txt" % (path,globalTag,jetFlavour));
        self.L2JetPar  = ROOT.JetCorrectorParameters("%s/%s_L2Relative_%s.txt" % (path,globalTag,jetFlavour));
        self.L3JetPar  = ROOT.JetCorrectorParameters("%s/%s_L3Absolute_%s.txt" % (path,globalTag,jetFlavour));
        self.vPar = ROOT.vector(ROOT.JetCorrectorParameters)()
        self.vPar.push_back(self.L1JetPar);
        self.vPar.push_back(self.L2JetPar);
        self.vPar.push_back(self.L3JetPar);
        # Add residuals if needed
        if doResidualJECs : 
            self.ResJetPar = ROOT.JetCorrectorParameters("%s/%s_L2L3Residual_%s.txt" % (path,globalTag,jetFlavour))
            self.vPar.push_back(self.ResJetPar);
        #Step3 (Construct a FactorizedJetCorrector object) 
        self.JetCorrector = ROOT.FactorizedJetCorrector(self.vPar)
        self.JetUncertainty = ROOT.JetCorrectionUncertainty("%s/%s_Uncertainty_%s.txt" % (path,globalTag,jetFlavour));
        self.vPar1 = ROOT.vector(ROOT.JetCorrectorParameters)()
        self.vPar1.push_back(self.L1JetPar);
        self.JetCorrectorL1 = ROOT.FactorizedJetCorrector(self.vPar1)
    def getMETCorrection(self,jets,rho,muons):
        px0, py0 = 0, 0
        for j in jets:
            if j.component(4).fraction() > 0.9: continue
            jp4 = j.p4().__class__(j.p4());
            jp4 *= j.rawFactor()
            mup4 = j.p4().__class__(0,0,0,0)
            for mu in muons:
                if mu.sourcePtr().track().isNonnull() and (mu.sourcePtr().isGlobalMuon() or mu.sourcePtr().isStandAloneMuon()) and deltaR(mu.eta(),mu.phi(),j.eta(),j.phi()) < 0.5:
                    mup4 += mu.p4()
            jp4 -= mup4
            self.JetCorrector.setJetEta(j.eta())
            self.JetCorrector.setJetPt(j.pt()*j.rawFactor()) # NOTE: we don't subtract the muon here!
            self.JetCorrector.setJetA(j.jetArea())
            self.JetCorrector.setRho(rho)
            corr3 = self.JetCorrector.getCorrection()
            self.JetCorrectorL1.setJetEta(j.eta())
            self.JetCorrectorL1.setJetPt(j.pt()*j.rawFactor()) # NOTE: we don't subtract the muon here!
            self.JetCorrectorL1.setJetA(j.jetArea())
            self.JetCorrectorL1.setRho(rho)
            corr1 = self.JetCorrectorL1.getCorrection()
            if jp4.pt() * corr3 < 10:
                #print " no correction from jet of pt %8.2f / %8.2f , phi %+5.3f / %+5.3f (corr3 %5.3f, corr1 %5.3f):  %+7.3f %+7.3f" % (j.pt(),jp4.pt(),j.phi(), jp4.phi(), corr3, corr1, -jp4.X()*(corr3  -corr1),-jp4.Y()*(corr3-corr1))
                continue
            #print " correction from jet of pt %8.2f, phi %+5.3f / %+5.3f (corr3 %5.3f, corr1 %5.3f):  %+7.3f %+7.3f" % (j.pt(),j.phi(), jp4.phi(), corr3, corr1, -jp4.X()*(corr3-corr1), -jp4.Y()*(corr3-
            px0 -= jp4.X()*(corr3-corr1)
            py0 -= jp4.Y()*(corr3-corr1)
        return [px0, py0]
