from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class BTagging(Analyzer):
    
    def process(self, event):
        jets = getattr(event, self.cfg_ana.input_objects)
        bjets = []
        for jet in jets: 
            if self.cfg_ana.filter_func(jet):
                bjets.append(jet)    
                jet.tags['b'] = True
            else:
                jet.tags['b'] = False
        
        setattr(event, self.cfg_ana.output, bjets)
