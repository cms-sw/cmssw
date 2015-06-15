import math

from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Muon, Tau
# from PhysicsTools.Heppy.physicsobjects.HTauTauElectron import HTauTauElectron
from PhysicsTools.Heppy.physicsobjects.Electron import Electron
from PhysicsTools.HeppyCore.utils.deltar import deltaR2
from ROOT import TVector3

class DiObject( object ):

    def __init__(self, diobject):
        self.diobject = diobject
        #p4 = LorentzVector( 1,0,0,1)
        # self.diobject.setP4(p4)
        self.leg1Gen = None
        self.leg2Gen = None
        self.leg1DeltaR = -1
        self.leg2DeltaR = -1

    def leg1(self):
        return self.daughter(0)

    def leg2(self):
        return self.daughter(1)

    def sumPt(self):
        '''pt_leg1 + pt_leg2. used for finding the best DiTau.'''
        return self.leg1().pt() + self.leg2().pt()

    def __getattr__(self, name):
        '''all accessors  from cmg::DiObject are transferred to this class.'''
        return getattr(self.diobject, name)

    def __str__(self):
        header = '{cls}: mvis={mvis}, sumpT={sumpt}'.format(
            cls = self.__class__.__name__,
            mvis = self.diobject.mass(),
            sumpt = self.sumPt() )
        return '\n'.join( [header,
                           '\t'+str(self.leg1()),
                           '\t'+str(self.leg2())] )


class DiTau( DiObject ):
    def __init__(self, diobject):
        super(DiTau, self).__init__(diobject)

    def met(self):
        return self.daughter(2)

    def svfitMass(self):
        return self.userFloat('mass')

    def svfitMassError(self):
        return self.userFloat('massUncert')

    def svfitPt(self):
        return self.userFloat('pt')

    def svfitPtError(self):
        return self.userFloat('ptUncert')

    def svfitEta(self):
        return self.userFloat('fittedEta')

    def svfitPhi(self):
        return self.userFloat('fittedPhi')

    def pZeta(self):
        if not hasattr(self, 'pZetaVis_'):
            self.calcPZeta()
        return self.pZetaVis_ + self.pZetaMET_

    def pZetaVis(self):
        if not hasattr(self, 'pZetaVis_'):
            self.calcPZeta()
        return self.pZetaVis_

    def pZetaMET(self):
        if not hasattr(self, 'pZetaMET_'):
            self.calcPZeta()
        return self.pZetaMET_

    def pZetaDisc(self):
        if not hasattr(self, 'pZetaVis_'):
            self.calcPZeta()
        return self.pZetaMET_ - 0.5*self.pZetaVis_

    # Calculate the pzeta variables with the same algorithm
    # as previously in the C++ DiObject class
    def calcPZeta(self):
        tau1PT = TVector3(self.leg1().p4().x(), self.leg1().p4().y(), 0.)
        tau2PT = TVector3(self.leg2().p4().x(), self.leg2().p4().y(), 0.)
        metPT = TVector3(self.met().p4().x(), self.met().p4().y(), 0.)
        zetaAxis = (tau1PT.Unit() + tau2PT.Unit()).Unit()
        self.pZetaVis_ = tau1PT*zetaAxis + tau2PT*zetaAxis
        self.pZetaMET_ = metPT*zetaAxis

    def mTLeg1(self):
        if hasattr(self, 'mt1'):
            return self.mt1
        else:
            self.mt1 = self.calcMT(self.leg1(), self.met())
            return self.mt1

    def mTLeg2(self):
        if hasattr(self, 'mt2'):
            return self.mt2
        else:
            self.mt2 = self.calcMT(self.leg2(), self.met())
            return self.mt2

    # This is the default transverse mass by convention
    def mt(self):
        return self.mTLeg2()

    # Calculate the transverse mass with the same algorithm
    # as previously in the C++ DiObject class
    @staticmethod
    def calcMT(cand1, cand2):
        pt = cand1.pt() + cand2.pt()
        px = cand1.px() + cand2.px()
        py = cand1.py() + cand2.py()
        return math.sqrt( pt*pt - px*px - py*py)

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

class DiMuon( DiTau ):
    def __init__(self, diobject):
        super(DiMuon, self).__init__(diobject)
        self.mu1 = Muon( super(DiMuon, self).leg1() )
        self.mu2 = Muon( super(DiMuon, self).leg2() )

    def __str__(self):
        header = 'DiMuon: mvis=%3.2f, sumpT=%3.2f' \
                 % (self.diobject.mass(),
                    self.sumPt() )
        return '\n'.join( [header] )


class TauMuon( DiTau ):
    def __init__(self, diobject):
        super(TauMuon, self).__init__(diobject)
        self.tau = Tau( super(TauMuon, self).leg1() )
        self.mu = Muon( super(TauMuon, self).leg2() )

    def leg1(self):
        return self.tau

    def leg2(self):
        return self.mu

class TauElectron( DiTau ):
    def __init__(self, diobject):
        super(TauElectron, self).__init__(diobject)
        self.tau = Tau( super(TauElectron, self).leg1() )
        self.ele = Electron( super(TauElectron, self).leg2() )
#         self.ele = HTauTauElectron( super(TauElectron, self).leg2() )

    def leg1(self):
        return self.tau

    def leg2(self):
        return self.ele

class MuonElectron( DiTau ):
    def __init__(self, diobject):
        super(MuonElectron, self).__init__(diobject)
        self.mu = Muon( super(MuonElectron, self).leg1() )
        self.ele = Electron( super(MuonElectron, self).leg2() )
#         self.ele = HTauTauElectron( super(MuonElectron, self).leg2() )

    def leg1(self):
        return self.mu

    def leg2(self):
        return self.ele

class TauTau( DiTau ):
    def __init__(self, diobject):
        super(TauTau, self).__init__(diobject)
        self.tau  = Tau( super(TauTau, self).leg1() )
        self.tau2 = Tau( super(TauTau, self).leg2() )

    def leg1(self):
        return self.tau

    def leg2(self):
        return self.tau2

