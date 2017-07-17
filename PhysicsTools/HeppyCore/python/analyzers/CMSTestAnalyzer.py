from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class CMSTestAnalyzer(Analyzer):
       
    def process(self, event):
        evid = event.input.eventAuxiliary().id()
        print 'run/lumi/event:', evid.run(), evid.luminosityBlock(), evid.event()
