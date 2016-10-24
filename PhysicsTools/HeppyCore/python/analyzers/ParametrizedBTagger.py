from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class ParametrizedBTagger(Analyzer):

    def process(self, event):
        jets = getattr(event, self.cfg_ana.input_jets)
        for jet in jets:
            is_bjet = False 
            if jet.match and \
               jet.match.match and \
               abs(jet.match.match.pdgid())== 5:
                is_bjet = True
            is_b_tagged = self.cfg_ana.roc.is_b_tagged(is_bjet)
            jet.tags['b'] = is_b_tagged
