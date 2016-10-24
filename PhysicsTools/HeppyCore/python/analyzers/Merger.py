from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
import copy
import itertools


class Merger(Analyzer):
    '''Merges collections of particle-like objects into a single collection 
    
        
        Example: 
    
        from PhysicsTools.HeppyCore.analyzers.Merger import Merger
        merge_particles = cfg.Analyzer(
            Merger,
            instance_label = 'leptons', 
            inputs = ['electrons','muons'],
            output = 'leptons', 
        )
    
        inputs: names of the collections of input
        output: collection of all particle-like objects in the input collections
        '''        
    def process(self, event):
        inputs = [getattr(event, name) for name in self.cfg_ana.inputs]
        output = list(ptc for ptc in itertools.chain(*inputs))
        if hasattr(self.cfg_ana, 'sort_key'):
            output.sort(key=self.cfg_ana.sort_key,
                        reverse=True)
        setattr(event, self.cfg_ana.output, output)
        
        
        
  
