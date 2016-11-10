from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

import copy
import PhysicsTools.HeppyCore.statistics.rrandom as random

class GaussianSmearer(Analyzer):
    '''Smears the 4-momentum of a collection of particles according to a gaussian model,
    and then applies an acceptance model
    
    Example: 
    
    from PhysicsTools.HeppyCore.analyzers.GaussianSmearer import Smearer     
    def accept_electron(ele):
      return abs(ele.eta()) < 2.5 and ele.e() > 5.
    electrons = cfg.Analyzer(
      Smearer,
      'electrons',
      output = 'electrons',
      input_objects = 'sim_leptons',
      accept=accept_electron, 
      mu_sigma=(1, 0.1)
    )
    
    output: name of the collection created in the event to hold the smeared particles
    input_objects: the collection of particles to be smeared
    accept: function object returning True if a particle is accepted and
      False otherwise
    mu_sigma: mean and width of the gaussian model (response and resolution)
    '''
    
    def process(self, event):
        input_objects = getattr(event, self.cfg_ana.input_objects)
        output = []
        for obj in input_objects:
            smeared = self.smear(obj)
            if self.cfg_ana.accept(smeared): 
                output.append(smeared)
        setattr(event, self.cfg_ana.output, output)

    def smear(self, obj):
        mu, sigma = self.cfg_ana.mu_sigma
        smear_factor = random.gauss(mu, sigma) 
        smeared = copy.deepcopy(obj)
        smeared._tlv *= smear_factor
        return smeared
