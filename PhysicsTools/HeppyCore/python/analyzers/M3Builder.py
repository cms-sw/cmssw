from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.tlv.resonance import Resonance 

import pprint 
import itertools

class M3Builder(Analyzer):
    
    def process(self, event):
        jets = getattr(event, self.cfg_ana.jets)

        m3 = None
        pt3max=0
        seljets=None
        #print jets

        if len(jets)>=3:
            for l in list(itertools.permutations(jets,3)):
                #ntag=sum([l[0].tags['b'],l[1].tags['b'],l[2].tags['b']])
                pt3=(l[0].p4()+l[1].p4()+l[2].p4()).Pt()
                if pt3>pt3max:
                    ptmax=pt3
                    seljets=l

            top_pdgid = 6
            m3 = Resonance(seljets, top_pdgid)
        setattr(event, self.instance_label, m3)



