from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class Printer(Analyzer):
       
    def process(self, event):
        print "printing event", event.iEv, 'input', event.input
        
