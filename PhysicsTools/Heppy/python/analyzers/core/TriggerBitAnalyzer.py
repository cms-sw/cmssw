import ROOT
from ROOT.heppy import TriggerBitChecker

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import NTupleVariable
        
class TriggerBitAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TriggerBitAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.processName = getattr(self.cfg_ana,"processName","HLT")
        self.outprefix   = getattr(self.cfg_ana,"outprefix",  self.processName)

    def declareHandles(self):
        super(TriggerBitAnalyzer, self).declareHandles()
        self.handles['TriggerResults'] = AutoHandle( ('TriggerResults','',self.processName), 'edm::TriggerResults' )

    def beginLoop(self, setup):
        super(TriggerBitAnalyzer,self).beginLoop(setup)
        self.triggerBitCheckers = []
        for T, TL in self.cfg_ana.triggerBits.iteritems():
                trigVec = ROOT.vector(ROOT.string)()
                for TP in TL:
                    trigVec.push_back(TP)
                outname="%s_%s"%(self.outprefix,T)
                if not hasattr(setup ,"globalVariables") :
                        setup.globalVariables = []
                setup.globalVariables.append( NTupleVariable(outname, eval("lambda ev: ev.%s" % outname), help="OR of %s"%TL) )
                self.triggerBitCheckers.append( (T, TriggerBitChecker(trigVec)) )

    def process(self, event):
        self.readCollections( event.input )
        triggerResults = self.handles['TriggerResults'].product()
        for T,TC in self.triggerBitCheckers:
            outname="%s_%s"%(self.outprefix,T)
            setattr(event,outname, TC.check(event.input.object(), triggerResults))

        return True

