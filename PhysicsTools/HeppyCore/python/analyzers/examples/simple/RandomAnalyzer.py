from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

import PhysicsTools.HeppyCore.statistics.rrandom as random

class RandomAnalyzer(Analyzer):

    def process(self, event):
        event.var_random = random.uniform(0,1)
