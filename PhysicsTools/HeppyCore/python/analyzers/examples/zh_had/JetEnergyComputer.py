from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
import copy

class JetEnergyComputer(Analyzer):
    '''Use the initial p4 to constrain the energy of the 4 jets,
    in ee -> 4 jet final states.
    
    from PhysicsTools.HeppyCore.analyzers.examples.zh_had.JetEnergyComputer import JetEnergyComputer
    compute_jet_energy = cfg.Analyzer(
      JetEnergyComputer,
      output_jets='rescaled_jets',
      input_jets='jets',
      sqrts=Collider.SQRTS
    )

    * output_jets: output jets with a rescaled energy.
    note that only the jet p4 is copied when creating a rescaled jet
    
    * input_jets: collection of jets to be rescaled
    
    * sqrts: center-of-mass energy of the collision
    
    '''
    
    def process(self, event):
        sqrts = self.cfg_ana.sqrts
        jets = getattr(event, self.cfg_ana.input_jets)
        assert(len(jets) == 4)
        # here solve the equation to get the energy scale factor for each jet.
        scale_factors = [1] * 4
        output = []
        for jet, factor in zip(jets, scale_factors):
            # the jets should not be deepcopied
            # as they are heavy objects containing
            # in particular a list of consistuent particles 
            scaled_jet = copy.copy(jet)
            scaled_jet._tlv = copy.deepcopy(jet._tlv)
            scaled_jet._tlv *= factor
            output.append(scaled_jet)
        setattr(event, self.cfg_ana.output_jets, output)
