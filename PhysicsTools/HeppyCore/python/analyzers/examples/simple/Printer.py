from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class Printer(Analyzer):

    def beginLoop(self, setup):
        super(Printer, self).beginLoop(setup)
        
    def process(self, event):
        self.mainLogger.info(
            "event {iEv}, var1 {var1}".format(
                iEv = event.iEv, var1 = event.input.var1
            ))
                             
        
