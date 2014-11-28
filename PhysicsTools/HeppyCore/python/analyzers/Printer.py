from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class Printer(Analyzer):

    def beginLoop(self, setup):
        super(Printer, self).beginLoop(setup)
        self.firstEvent = True
        
    def process(self, event):
        if self.firstEvent:
            event.input.Print()
            self.firstEvent = False
        print "printing event", event.iEv, 'var1', event.input.var1
        
