from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.genbrowser import GenBrowser
from PhysicsTools.HeppyCore.particles.pdgcodes import hasBottom

class GenAnalyzer(Analyzer):
    
    def process(self, event):
        genptcs = event.gen_particles
        charged_hadrons = [ptc for ptc in genptcs if ptc.q() and ptc.status()==1]
        event.genbrowser = GenBrowser(event.gen_particles,
                                      event.gen_vertices)
        event.hadrons_from_b = []
        for hadron in charged_hadrons:
            ancestors = event.genbrowser.ancestors(hadron)
            for ancestor in ancestors:
                if hasBottom(ancestor.pdgid() ):
                    event.hadrons_from_b.append(hadron)
                    break 
        
