from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
import ROOT

class Lepton( PhysicsObject):
    def ip3D(self):
        '''3D impact parameter value.'''
        return abs(self.dB(self.PV3D))

    def sip3D(self):
        '''3D impact parameter significance.'''
        db, edb = self.dB(self.PV3D), self.edB(self.PV3D)
        return abs(db/edb) if edb > 0 else 999.

    def absIsoFromEA(self, area='04'):
        '''Calculate Isolation using the effective area approach.'''
        photonIso = self.photonIso()
        offset = self.rho*getattr(self,"EffectiveArea"+area)
        return self.chargedHadronIso()+max(0.,photonIso+self.neutralHadronIso()-offset)            

    def relIsoFromEA(self, area='04'):
        return self.absIsoFromEA(area)/self.pt()

    def relIso(self, dBetaFactor=0, allCharged=0):
        '''Relative isolation with default cone size of 0.4.'''
        rel = self.absIsoR(dBetaFactor=dBetaFactor, allCharged=allCharged)/self.pt()
        return rel

    def absIsoR(self, R=0.4, dBetaFactor=0, allCharged=False):
        '''Isolation in given cone with optional delta-beta subtraction.'''
        if dBetaFactor>0 and self.puChargedHadronIsoR(R)<0:
            raise ValueError('If you want to use dbeta corrections, you must make sure that the pu charged hadron iso is available. This should never happen') 
        neutralIso = self.neutralHadronIsoR(R) + self.photonIsoR(R)
        corNeutralIso = neutralIso - dBetaFactor * self.puChargedHadronIsoR(R)
        charged = self.chargedHadronIsoR(R)
        if allCharged:
            charged = self.chargedAllIsoR(R)
        return charged + max(corNeutralIso, 0.)

    def relIsoR(self, R=0.4, dBetaFactor=0, allCharged=False):
        return self.absIsoR(R, dBetaFactor, allCharged)/self.pt()

    def lostInner(self):
        if hasattr(self.innerTrack(),"trackerExpectedHitsInner") :
            return self.innerTrack().trackerExpectedHitsInner().numberOfLostHits()
        else :	
            return self.innerTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS)	

    def p4WithFSR(self):
        ret = self.p4()
        for p in getattr(self, 'ownFsrPhotons', getattr(self, 'fsrPhotons', [])):
            ret += p.p4()
        return ret

    def __str__(self):
        ptc = super(Lepton, self).__str__()
        return '{ptc}, iso={iso:5.2f}'.format(ptc=ptc, iso=self.relIso())
