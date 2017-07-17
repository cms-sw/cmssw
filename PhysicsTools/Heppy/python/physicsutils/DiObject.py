import math

from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Muon, Tau, Electron
from PhysicsTools.Heppy.physicsobjects.PhysicsObject import PhysicsObject
from PhysicsTools.Heppy.physicsobjects.HTauTauElectron import HTauTauElectron
from CMGTools.RootTools.utils.DeltaR import deltaR2

class DiObject( PhysicsObject ):
    '''Generic di-object class, to handle di-objects from the EDM file
    '''
    
    def __init__(self, diobject):
        '''diobject is the di-object read from the edm file'''
        self.diobject = diobject
        self.leg1Gen = None
        self.leg2Gen = None
        self.leg1DeltaR = -1
        self.leg2DeltaR = -1
        super(DiObject, self).__init__(diobject)

    def sumPt(self):
        '''pt_leg1 + pt_leg2, e.g. used for finding the best DiTau.'''
        return self.leg1().pt() + self.leg2().pt()

    def __str__(self):
        header = '{cls}: mvis={mvis}, mT={mt}, sumpT={sumpt}'.format(
            cls = self.__class__.__name__,
            mvis = self.diobject.mass(),
            mt = self.diobject.mTLeg2(),
            sumpt = self.sumPt() )
        return '\n'.join( [header,
                           '\t'+str(self.leg1()),
                           '\t'+str(self.leg2())] )



class DiMuon( DiObject ):

    def __init__(self, diobject):
        super(DiMuon, self).__init__(diobject)
        self.mu1 = Muon( diobject.leg1() )
        self.mu2 = Muon( diobject.leg2() )

    def leg1(self):
        return self.mu1

    def leg2(self):
        return self.mu2

    def __str__(self):
        return 'DiMuon: mass={mass:5.2f}, sumpt={sumpt:5.2f}, pt={pt:5.2f}'.format(
            mass = self.mass(),
            sumpt = self.sumPt(),
            pt = self.pt()
            )
    

class DiElectron( DiObject ):

    def __init__(self, diobject):
        super(DiElectron, self).__init__(diobject)
        self.ele1 = Electron( diobject.leg1() )
        self.ele2 = Electron( diobject.leg2() )

    def leg1(self):
        return self.ele1

    def leg2(self):
        return self.ele2

    def __str__(self):
        header = 'DiElectron: mvis=%3.2f, sumpT=%3.2f' \
                 % (self.diobject.mass(),
                    self.sumPt() )
        return '\n'.join( [header] )


class DiTau( DiObject ):
    def __init__(self, diobject):
        super(DiTau, self).__init__(diobject)
        
    def match(self, genParticles):
        #TODO review matching algorithm
        #TODO move matching stuff even higher?
        # print self
        genTaus = []
        ZorPhotonorHiggs = [22, 23, 25, 35, 36, 37]
        for gen in genParticles:
            # print '\t', gen
            if abs(gen.pdgId())==15 and gen.mother().pdgId() in ZorPhotonorHiggs:
                genTaus.append( gen )
        # print 'Gen taus: '
        # print '\n'.join( map( str, genTaus ) )
        if len(genTaus)!=2:
            #COLIN what about WW, ZZ? 
            return (-1, -1)
        else:
            dR2leg1Min, self.leg1Gen = ( float('inf'), None)
            dR2leg2Min, self.leg2Gen = ( float('inf'), None) 
            for genTau in genTaus:
                dR2leg1 = deltaR2(self.leg1().eta(), self.leg1().phi(),
                                  genTau.eta(), genTau.phi() )
                dR2leg2 = deltaR2(self.leg2().eta(), self.leg2().phi(),
                                  genTau.eta(), genTau.phi() )
                if dR2leg1 <  dR2leg1Min:
                    dR2leg1Min, self.leg1Gen = (dR2leg1, genTau)
                if dR2leg2 <  dR2leg2Min:
                    dR2leg2Min, self.leg2Gen = (dR2leg2, genTau)
            # print dR2leg1Min, dR2leg2Min
            # print self.leg1Gen
            # print self.leg2Gen
            self.leg1DeltaR = math.sqrt( dR2leg1Min )
            self.leg2DeltaR = math.sqrt( dR2leg2Min )
            return (self.leg1DeltaR, self.leg2DeltaR)        


class TauMuon( DiTau ):
    '''Holds a CMG TauMuon, and the 2 legs as a python Tau and Muon'''
    def __init__(self, diobject):
        super(TauMuon, self).__init__(diobject)
        self.tau = Tau( self.leg1() )
        self.mu = Muon( self.leg2() )

    def leg1(self):
        return self.tau

    def leg2(self):
        return self.mu



class TauElectron( DiTau ):
    def __init__(self, diobject):
        super(TauElectron, self).__init__(diobject)
        self.tau = Tau( diobject.leg1() )
        self.ele = HTauTauElectron( diobject.leg2() )

    def leg1(self):
        return self.tau

    def leg2(self):
        return self.ele


class MuonElectron( DiTau ):
    def __init__(self, diobject):
        super(MuonElectron, self).__init__(diobject)
        self.mu = Muon( diobject.leg1() )
        self.ele = HTauTauElectron( diobject.leg2() )

    def leg1(self):
        return self.mu

    def leg2(self):
        return self.ele


class TauTau( DiTau ):
    def __init__(self, diobject):
        super(TauTau, self).__init__(diobject)
        self.tau = Tau( diobject.leg1() )
        self.tau2 = Tau( diobject.leg2() )

    def leg1(self):
        return self.tau

    def leg2(self):
        return self.tau2
    

