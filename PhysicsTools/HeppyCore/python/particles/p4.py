import math
from PhysicsTools.HeppyCore.configuration import Collider

from functools import total_ordering

class P4(object):

    def __init__(self, *args, **kwargs):
        super(P4, self).__init__(*args, **kwargs)
    
    def p4(self):
        '''4-momentum, px, py, pz, E'''
        return self._tlv

    def p3(self):
        '''3-momentum px, py, pz'''
        return self._tlv.Vect()

    def e(self):
        '''energy'''
        return self._tlv.E()

    def pt(self):
        '''transverse momentum (magnitude of p3 in transverse plane)'''
        return self._tlv.Pt()
    
    def theta(self):
        '''angle w/r to transverse plane'''
        return math.pi/2 - self._tlv.Theta()

    def eta(self):
        '''pseudo-rapidity (-ln(tan self._tlv.Theta()/2)).
        theta = 0 -> eta = +inf
        theta = pi/2 -> 0 
        theta = pi -> eta = -inf
        '''
        if self._tlv.Pt()<1e-9:
            if self._tlv.Pz()>0.:
                return float('inf')
            else:
                return -float('inf')
        else:
            return self._tlv.Eta()

    def phi(self):
        '''azymuthal angle (from x axis, in the transverse plane)'''
        return self._tlv.Phi()

    def m(self):
        '''mass'''
        return self._tlv.M()
    
    
    def sort_key(self):
        if Collider.BEAMS == 'ee':
            return self.e()
        else:
            return self.pt() 
    
    def __gt__(self, other):
        '''sorting by pT or energy depending on Collider.BEAMS'''
        return self.sort_key() > other.sort_key()  
        
    def __lt__(self, other):
        '''sorting by pT or energy depending on Collider.BEAMS'''
        return self.sort_key() < other.sort_key()  
    
    def __str__(self):
        if Collider.BEAMS == 'pp':
            return 'pt = {pt:5.1f}, e = {e:5.1f}, eta = {eta:5.2f}, phi = {phi:5.2f}, mass = {m:5.2f}'.format(
                pt = self.pt(),
                e = self.e(),
                eta = self.eta(),
                phi = self.phi(),
                m = self.m()
            )
        elif Collider.BEAMS == 'ee':
            return 'e = {e:5.1f}, theta = {theta:5.2f}, phi = {phi:5.2f}, mass = {m:5.2f}'.format(
                e = self.e(),
                eta = self.eta(),
                theta = self.theta(),
                phi = self.phi(),
                m = self.m()
            )
