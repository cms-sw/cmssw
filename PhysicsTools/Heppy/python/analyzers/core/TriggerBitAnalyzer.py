import ROOT

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import NTupleVariable
import PhysicsTools.HeppyCore.framework.config as cfg
        
class TriggerBitAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TriggerBitAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.processName = getattr(self.cfg_ana,"processName","HLT")
        self.fallbackName = getattr(self.cfg_ana,"fallbackProcessName",None)
        self.outprefix   = getattr(self.cfg_ana,"outprefix",  self.processName)
        self.unrollbits = ( hasattr(self.cfg_ana,"unrollbits") and self.cfg_ana.unrollbits )
        self.saveIsUnprescaled = getattr(self.cfg_ana,"saveIsUnprescaled",False)
        self.force1prescale = False
        if self.cfg_comp.isMC and self.saveIsUnprescaled:
            print 'Cannot save prescale information in MC: will put everything to unprescaled'
            self.saveIsUnprescaled = False
            self.force1prescale = True

    def declareHandles(self):
        super(TriggerBitAnalyzer, self).declareHandles()
        fallback = ('TriggerResults','',self.fallbackName) if self.fallbackName else None
        self.handles['TriggerResults'] = AutoHandle( ('TriggerResults','',self.processName), 'edm::TriggerResults', fallbackLabel=(('TriggerResults','',self.fallbackName) if self.fallbackName else None) )
        if self.saveIsUnprescaled: self.handles["TriggerPrescales"] = AutoHandle( ('patTrigger','',self.processName), 'pat::PackedTriggerPrescales', fallbackLabel=(('patTrigger','',self.fallbackName) if self.fallbackName else None) )

    def beginLoop(self, setup):
        super(TriggerBitAnalyzer,self).beginLoop(setup)
        self.triggerBitCheckers = []
        if self.unrollbits :
            self.allPaths = set()
            self.triggerBitCheckersSingleBits = []

        for T, TL in self.cfg_ana.triggerBits.iteritems():
                trigVec = ROOT.vector(ROOT.string)()
                for TP in TL:
                    trigVec.push_back(TP)
                    if self.unrollbits :
                        if TP not in self.allPaths :
                            self.allPaths.update([TP])
                            trigVecBit = ROOT.vector(ROOT.string)()
                            trigVecBit.push_back(TP)
                            outname="%s_BIT_%s"%(self.outprefix,TP)
                            if not hasattr(setup ,"globalVariables") :
                                setup.globalVariables = []
                            if outname[-1] == '*' :
                                outname=outname[0:-1]
                            setup.globalVariables.append( NTupleVariable(outname, eval("lambda ev: ev.%s" % outname), help="Trigger bit  %s"%TP) )
                            if self.saveIsUnprescaled or self.force1prescale: setup.globalVariables.append( NTupleVariable(outname+'_isUnprescaled', eval("lambda ev: ev.%s_isUnprescaled" % outname), help="Trigger bit  %s isUnprescaled flag"%TP) )
                            self.triggerBitCheckersSingleBits.append( (TP, ROOT.heppy.TriggerBitChecker(trigVecBit)) )

                outname="%s_%s"%(self.outprefix,T)  
                if not hasattr(setup ,"globalVariables") :
                        setup.globalVariables = []
                setup.globalVariables.append( NTupleVariable(outname, eval("lambda ev: ev.%s" % outname), help="OR of %s"%TL) )
                if self.saveIsUnprescaled or self.force1prescale: setup.globalVariables.append( NTupleVariable(outname+'_isUnprescaled', eval("lambda ev: ev.%s_isUnprescaled" % outname), help="OR of %s is Unprescaled flag"%TL) )
                self.triggerBitCheckers.append( (T, ROOT.heppy.TriggerBitChecker(trigVec)) )
                

    def process(self, event):
        self.readCollections( event.input )
        triggerResults = self.handles['TriggerResults'].product()
        if self.saveIsUnprescaled: triggerPrescales = self.handles["TriggerPrescales"].product()
        for T,TC in self.triggerBitCheckers:
            outname="%s_%s"%(self.outprefix,T)
            setattr(event,outname, TC.check(event.input.object(), triggerResults))
            if self.saveIsUnprescaled: setattr(event,outname+'_isUnprescaled', TC.check_unprescaled(event.input.object(), triggerResults, triggerPrescales))
            if self.force1prescale: setattr(event,outname+'_isUnprescaled', True)
        if self.unrollbits :
            for TP,TC in self.triggerBitCheckersSingleBits:
               outname="%s_BIT_%s"%(self.outprefix,TP)
               if outname[-1] == '*' :
                  outname=outname[0:-1]
               setattr(event,outname, TC.check(event.input.object(), triggerResults))
               if self.saveIsUnprescaled: setattr(event,outname+'_isUnprescaled', TC.check_unprescaled(event.input.object(), triggerResults, triggerPrescales))
               if self.force1prescale: setattr(event,outname+'_isUnprescaled', True)

        return True


setattr(TriggerBitAnalyzer,"defaultConfig",cfg.Analyzer(
    TriggerBitAnalyzer, name="TriggerFlags",
    processName = 'HLT',
    triggerBits = {
        # "<name>" : [ 'HLT_<Something>_v*', 'HLT_<SomethingElse>_v*' ] 
}
)
)
setattr(TriggerBitAnalyzer,"defaultEventFlagsConfig",cfg.Analyzer(
    TriggerBitAnalyzer, name="EventFlags",
    processName = 'PAT',
    fallbackProcessName = 'RECO',
    outprefix   = 'Flag',
    triggerBits = {
        "HBHENoiseFilter" : [ "Flag_HBHENoiseFilter" ],
        "CSCTightHaloFilter" : [ "Flag_CSCTightHaloFilter" ],
        "hcalLaserEventFilter" : [ "Flag_hcalLaserEventFilter" ],
        "EcalDeadCellTriggerPrimitiveFilter" : [ "Flag_EcalDeadCellTriggerPrimitiveFilter" ],
        "goodVertices" : [ "Flag_goodVertices" ],
        "trackingFailureFilter" : [ "Flag_trackingFailureFilter" ],
        "eeBadScFilter" : [ "Flag_eeBadScFilter" ],
        "ecalLaserCorrFilter" : [ "Flag_ecalLaserCorrFilter" ],
        "trkPOGFilters" : [ "Flag_trkPOGFilters" ],
        "trkPOG_manystripclus53X" : [ "Flag_trkPOG_manystripclus53X" ],
        "trkPOG_toomanystripclus53X" : [ "Flag_trkPOG_toomanystripclus53X" ],
        "trkPOG_logErrorTooManyClusters" : [ "Flag_trkPOG_logErrorTooManyClusters" ],
        "METFilters" : [ "Flag_METFilters" ],
    }
)
)
