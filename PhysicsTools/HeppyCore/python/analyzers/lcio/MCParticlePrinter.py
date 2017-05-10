from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class MCParticlePrinter(Analyzer):
    
    def process(self, event):
        mcparticles = event.input.getCollection('MCParticle')
        for ptc in mcparticles: 
            p4 = ptc.getLorentzVec()
            self.mainLogger.info(
                "ptc E={energy}, m={mass}".format(
                    energy = p4.E(),
                    mass = p4.M()
                    ))
                             
        
