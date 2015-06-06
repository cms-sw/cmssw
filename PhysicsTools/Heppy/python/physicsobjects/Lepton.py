from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
import ROOT

class Lepton( PhysicsObject):
    def ip3D(self):
        '''3D impact parameter value.'''
        return abs(self.dB(self.PV3D))


    def sip3D(self):
        '''3D impact parameter significance.'''
        return abs(self.dB(self.PV3D) / self.edB(self.PV3D))

    def absIsoFromEA(self,area = "04"):
        '''
        Calculate Isolation using the effective area approach. If fsrPhotons is set
        the list of photons is subtracted from the isolation cone. It works with one or
        two effective Areas in case one needs to do photon and neutral hadron separately
        '''
        photonIso = self.photonIso()
        if hasattr(self,'fsrPhotons'):
            for gamma in self.fsrPhotons:
                photonIso=max(photonIso-gamma.pt(),0.0)                

        offset = self.rho*getattr(self,"EffectiveArea"+area)
        return self.chargedHadronIso()+max(0.,photonIso+self.neutralHadronIso()-offset)            


    def absIso(self, dBetaFactor=0, allCharged=0):
        if dBetaFactor>0 and self.puChargedHadronIso()<0:
            raise ValueError('If you want to use dbeta corrections, you must make sure that the pu charged hadron iso is available. This should never happen') 
        neutralIso = self.neutralHadronIso()+self.photonIso()
        #Recover FSR
        if hasattr(self,'fsrPhotons'):
            for gamma in self.fsrPhotons:
                neutralIso=neutralIso-gamma.pt()
        corNeutralIso = neutralIso - dBetaFactor * self.puChargedHadronIso()
        charged = self.chargedHadronIso()
        if allCharged:
            charged = self.chargedAllIso()
        return charged + max(corNeutralIso,0)

    def relIso(self,dBetaFactor=0, allCharged=0):
        rel = self.absIso(dBetaFactor, allCharged)/self.pt()
        return rel

    def relEffAreaIso(self,rho):
        '''MIKE, missing doc'''
        return self.absEffAreaIso(rho)/self.pt()

    def lostInner(self):
        if hasattr(self.innerTrack(),"trackerExpectedHitsInner") :
		return self.innerTrack().trackerExpectedHitsInner().numberOfLostHits()
	else :	
		return self.innerTrack().hitPattern().numberOfHits(ROOT.reco.HitPattern.MISSING_INNER_HITS)	

