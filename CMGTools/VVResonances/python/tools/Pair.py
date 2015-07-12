import math

from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi


class Pair(object):
    def __init__(self,leg1,leg2,pdg = 0):
        self.leg1 = leg1
        self.leg2 = leg2
        self.pdgId = pdg
        self.LV = leg1.p4()+leg2.p4()
        et1 = math.sqrt(leg1.mass()*leg1.mass()+leg1.pt()*leg1.pt())
        et2 = math.sqrt(leg2.mass()*leg2.mass()+leg2.pt()*leg2.pt())
        self.MT  =math.sqrt(self.leg1.p4().mass()*self.leg1.p4().mass()+\
            self.leg2.p4().mass()*self.leg2.p4().mass()+2*(et1*et2-self.leg1.p4().px()*self.leg2.p4().px()-self.leg1.p4().py()*self.leg2.p4().py()))

        

    def rawP4(self):
        return self.leg1.p4()+self.leg2.p4()

    def p4(self):
        return self.LV
    
    def m(self):
        return self.LV.mass()
    
    def pdgId(self):
        return self.pdgId
    
    def mt2(self):
        return self.MT*self.MT

    def mt(self):
        return self.MT

    def deltaPhi(self):
        return abs(deltaPhi(self.leg1.phi(),self.leg2.phi()))

    def deltaR(self):
        return abs(deltaR(self.leg1.eta(),self.leg1.phi(),self.leg2.eta(),self.leg2.phi()))

    def __getattr__(self, name):
        return getattr(self.LV,name)

