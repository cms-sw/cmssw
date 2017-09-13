from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.genbrowser import GenBrowser
from PhysicsTools.HeppyCore.particles.pdgcodes import hasBottom

class ChargedHadronsFromB(Analyzer):
    
    def process(self, event):
        genptcs = event.gen_particles
        bquarks = []
        charged_hadrons = []
        event.hadrons_from_b = []
        for ptc in genptcs:
            if abs(ptc.pdgid())==5:
                bquarks.append(ptc)
            elif ptc.q() and ptc.status()==1:
                charged_hadrons.append(ptc)
        if len(bquarks) == 0 or len(charged_hadrons) == 0:
            return
        event.genbrowser = GenBrowser(event.gen_particles,
                                      event.gen_vertices)
        event.hadrons_from_b = []
        event.hadrons_not_from_b = []
        for hadron in charged_hadrons:
            ancestors = event.genbrowser.ancestors(hadron)
            is_from_b = False 
            for ancestor in ancestors:
                if hasBottom(ancestor.pdgid() ):
                    is_from_b = True
            if is_from_b:
                event.hadrons_from_b.append(hadron)
            else:
                event.hadrons_not_from_b.append(hadron)
            
        
