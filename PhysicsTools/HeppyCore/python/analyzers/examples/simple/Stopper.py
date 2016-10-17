from heppy.framework.analyzer import Analyzer
from heppy.framework.exceptions import UserStop

class Stopper(Analyzer):

    def process(self, event):
        if event.iEv == self.cfg_ana.iEv:
            raise UserStop('stopping at event {iEv}'.format(iEv=event.iEv))
                             
        
