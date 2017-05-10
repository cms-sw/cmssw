from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.exceptions import UserStop

class Stopper(Analyzer):

    def process(self, event):
        if event.iEv == self.cfg_ana.iEv:
            raise UserStop('stopping at event {iEv}'.format(iEv=event.iEv))
                             
        
