from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.tlv.resonance import Resonance2 as Resonance

import pprint 
import itertools

mass = {23: 91, 25: 125}

class ResonanceBuilder(Analyzer):
    
    def process(self, event):
        legs = getattr(event, self.cfg_ana.leg_collection)
        resonances = []
        for leg1, leg2 in itertools.combinations(legs,2):
            resonances.append( Resonance(leg1, leg2, self.cfg_ana.pdgid) )
        # sorting according to distance to nominal mass
        nominal_mass = mass[self.cfg_ana.pdgid]
        resonances.sort(key=lambda x: abs(x.m()-nominal_mass))
        setattr(event, self.cfg_ana.output, resonances)
        # getting legs of best resonance
        legs = []
        if len(resonances):
            legs = resonances[0].legs
        setattr(event, '_'.join([self.cfg_ana.output, 'legs']), legs)
                
