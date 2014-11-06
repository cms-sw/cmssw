from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
import ROOT
class Lepton( PhysicsObject):
    def sip3D(self):
        '''3D impact parameter, for H to ZZ to 4l analysis.'''
        return abs(self.dB(self.PV3D) / self.edB(self.PV3D))

    def absIsoFromEA(self,rho,eta,effectiveArea1 = None,effectiveArea2 = None):
        '''
        Calculate Isolation using the effective area approach. If fsrPhotons is set
        the list of photons is subtracted from the isolation cone. It works with one or
        two effective Areas in case one needs to do photon and neutral hadron separately
        '''
        photonIso = self.photonIso()
        if hasattr(self,'fsrPhotons'):
            for gamma in self.fsrPhotons:
                photonIso=photonIso-gamma.pt()                
        ea1 = rho
        ea2 = rho
        if effectiveArea1 is not None:
            for element in effectiveArea1:
                if abs(eta)>= element['etaMin'] and \
                   abs(eta)< element['etaMax']:
                    ea1 = ea1 * element['area']
                    break
        else:
            return self.chargedHadronIso()+max(0.,photonIso+self.neutralHadronIso()-ea1)
        if effectiveArea2 is not None:
            for element in effectiveArea2:
                if abs(eta)>= element['etaMin'] and \
                   abs(eta)< element['etaMax']:
                    ea2 = ea2 * element['area']
            return self.chargedHadronIso()+max(0.,photonIso-ea1)+max(0.,self.neutralHadronIso()-ea2)
        else:
            return self.chargedHadronIso()+max(0.,photonIso+self.neutralHadronIso()-ea1)


    def absIso(self,dBetaFactor = 0,allCharged=0):
        if dBetaFactor>0 and self.puChargedHadronIso()<0:
            raise ValueError('If you want to use dbeta corrections, you must make sure that the pu charged hadron iso is available. This should never happen') 
        neutralIso = self.neutralHadronIso()+self.photonIso()
        #Recover FSR
        if hasattr(self,'fsrPhotons'):
            for gamma in self.fsrPhotons:
                neutralIso=neutralIso-gamma.pt()
        corNeutralIso = neutralIso - dBetaFactor * self.puChargedHadronIso();
        charged = self.chargedHadronIso();
        if  allCharged:
            charged = self.chargedAllIso();
        return charged + max(corNeutralIso,0)

    def  relIso(self,dBetaFactor=0, allCharged=0):
         rel = self.absIso(dBetaFactor, allCharged)/self.pt();
         return rel


    def relIsoAllChargedDB05(self):
        '''Used in the H2TauTau analysis: rel iso, dbeta=0.5, using all charged particles.'''
        return self.relIso( 0.5, 1 )


    def relEffAreaIso(self,rho):
        '''MIKE, missing doc'''
        return 0


    def relEffAreaIso(self,rho):
        '''MIKE, missing doc'''
        return self.absEffAreaIso(rho)/self.pt()

    def lostInner(self):
        if hasattr(self.innerTrack(),"trackerExpectedHitsInner") :
		return self.innerTrack().trackerExpectedHitsInner().numberOfLostHits()
	else :	
		return self.innerTrack().hitPattern().numberOfHits(ROOT.reco.HitPattern.MISSING_INNER_HITS)	

