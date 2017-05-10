from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class Masker(Analyzer):
    '''Returns in output all objects that are in input and not in mask. 

    Example: 

    from PhysicsTools.HeppyCore.analyzers.Masker import Masker
    particles_not_zed = cfg.Analyzer(
      Masker,
      output = 'particles_not_zed',
      input = 'gen_particles_stable',
      mask = 'zeds',
    )

    '''
    def process(self, event):
        inputs = getattr(event, self.cfg_ana.input)
        masks = getattr(event, self.cfg_ana.mask)
        output = [obj for obj in inputs if obj not in masks]
        setattr(event, self.cfg_ana.output, output)
