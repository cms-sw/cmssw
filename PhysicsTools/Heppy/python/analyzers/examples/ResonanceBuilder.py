
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.physicsobjects.Particle import Particle

import pprint 
import itertools

mass = {23: 91., 25: 125.}

class Resonance(Particle):
    '''Resonance decaying into 2 particles.

    The interface of this class mimics the interface of the CMS Candidate class. 
    In this way Resonance objects or CMS Candidate objects can be processed 
    transparently. 
    '''

    def __init__(self, leg1, leg2, pdgid, status=3): 
        '''
        Parameters (stored as attributes):
        leg1,2 : first and second leg.
        pdgid  : pdg code of the resonance
        status : status code of the resonance
        '''
        self.leg1 = leg1 
        self.leg2 = leg2 
        self._p4 = leg1.p4() + leg2.p4()
        self._charge = leg1.charge() + leg2.charge()
        self._pdgid = pdgid
        self._status = status
    
    def p4(self):
        return self._p4

    def pt(self):
        return self._p4.pt()

    def energy(self):
        return self._p4.energy()

    def eta(self):
        return self._p4.eta()

    def phi(self):
        return self._p4.phi()

    def mass(self):
        return self._p4.mass()

    def charge(self):
        return self._charge

    def pdgId(self):
        return self._pdgid


class ResonanceBuilder(Analyzer):
    '''Builds resonances from an input collection of particles. 

    Example configuration:

    from PhysicsTools.Heppy.analyzers.examples.ResonanceBuilder import ResonanceBuilder
    dimuons = cfg.Analyzer(
       ResonanceBuilder,
       'dimuons',                            
       leg_collection = 'muons',             # input collection
       filter_func = lambda x : True,        # filtering function for input objects. here, take all.
       pdgid = 23                            # pdgid for the resonances, here Z
       )

    This analyzer puts one collection in the event:
    event.dimuons : all resonances, sorted by their distance to the nominal mass
                    corresponding to the specified pdgid
    '''
    def process(self, event):
        legs = getattr(event, self.cfg_ana.leg_collection)
        legs = [leg for leg in legs if self.cfg_ana.filter_func(leg)]
        resonances = []
        for leg1, leg2 in itertools.combinations(legs,2):
            resonances.append( Resonance(leg1, leg2, self.cfg_ana.pdgid, 3) )
        # sorting according to distance to nominal mass
        nominal_mass = mass[self.cfg_ana.pdgid]
        resonances.sort(key=lambda x: abs(x.mass()-nominal_mass))
        setattr(event, self.instance_label, resonances)
        

