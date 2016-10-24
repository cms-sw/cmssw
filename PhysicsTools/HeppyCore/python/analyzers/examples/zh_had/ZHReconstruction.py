from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.tlv.resonance import Resonance2 as Resonance

import pprint 
import itertools
import copy

mass = {23: 91, 25: 125}

class ZHReconstruction(Analyzer):
    
    def process(self, event):
        jets = getattr(event, self.cfg_ana.input_jets)
        bjets = [jet for jet in jets if jet.tags['b']]
        higgses = []
        for leg1, leg2 in itertools.combinations(bjets,2):
            higgses.append( Resonance(leg1, leg2, 25) )
        higgs = None
        zed = None
        if len(higgses):
            # sorting according to distance to nominal mass
            nominal_mass = mass[25]
            higgses.sort(key=lambda x: abs(x.m()-nominal_mass))
            higgs = higgses[0]
            remaining_jets = copy.copy(jets)
            remaining_jets.remove(higgs.leg1())
            remaining_jets.remove(higgs.leg2())
            assert(len(remaining_jets) == 2)
            zed = Resonance(remaining_jets[0], remaining_jets[1], 21)
        setattr(event, self.cfg_ana.output_higgs, higgs)
        setattr(event, self.cfg_ana.output_zed, zed)
        
        
