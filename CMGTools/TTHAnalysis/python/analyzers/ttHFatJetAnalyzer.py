import random
import math
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import JetReCalibrator

from PhysicsTools.HeppyCore.utils.deltar import *
import PhysicsTools.HeppyCore.framework.config as cfg

class ttHFatJetAnalyzer( Analyzer ):
    """Taken from RootTools.JetAnalyzer, simplified, modified, added corrections    """
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(ttHFatJetAnalyzer,self).__init__(cfg_ana, cfg_comp, looperName)
        # -- this part needs some updates for 7.0.X (but AK7 is correct)
        #mcGT   = cfg_ana.mcGT   if hasattr(cfg_ana,'mcGT')   else "PHYS14_25_V1"
        #dataGT = cfg_ana.dataGT if hasattr(cfg_ana,'dataGT') else "GR_70_V2_AN1"
        #self.shiftJEC = self.cfg_ana.shiftJEC if hasattr(self.cfg_ana, 'shiftJEC') else 0
        #self.doJEC = self.cfg_ana.recalibrateJets or (self.shiftJEC != 0)
        #if self.doJEC:
        #  if self.cfg_comp.isMC:
        #    self.jetReCalibrator = JetReCalibrator(mcGT,"AK7PFchs", False,cfg_ana.jecPath)
        #  else:
        #    self.jetReCalibrator = JetReCalibrator(dataGT,"AK7PFchs", True,cfg_ana.jecPath)
        self.jetLepDR = self.cfg_ana.jetLepDR  if hasattr(self.cfg_ana, 'jetLepDR') else 0.5
        self.lepPtMin = self.cfg_ana.minLepPt  if hasattr(self.cfg_ana, 'minLepPt') else -1


    def declareHandles(self):
        super(ttHFatJetAnalyzer, self).declareHandles()
        self.handles['jets'] = AutoHandle( self.cfg_ana.jetCol, 'std::vector<pat::Jet>' )
    
    def beginLoop(self, setup):
        super(ttHFatJetAnalyzer,self).beginLoop(setup)
        
    def process(self, event):
        self.readCollections( event.input )

        ## Read jets, if necessary recalibrate and shift MET
        allJets = map(Jet, self.handles['jets'].product()) 

        ## Apply jet selection
        event.fatJets     = []
        event.fatJetsNoID = []
        for jet in allJets:
            if self.testJetNoID( jet ): 
                event.fatJetsNoID.append(jet) 
                if self.testJetID (jet ):
                    event.fatJets.append(jet)

        ## Associate jets to leptons
        leptons = event.inclusiveLeptons if hasattr(event, 'inclusiveLeptons') else event.selectedLeptons
        jlpairs = matchObjectCollection( leptons, allJets, self.jetLepDR**2)

        for jet in allJets:
            jet.leptons = [l for l in jlpairs if jlpairs[l] == jet ]

        for lep in leptons:
            jet = jlpairs[lep]
            if jet is None:
                lep.fatjet = lep.jet
            else:
                lep.fatjet = jet   
                

    def testJetID(self, jet):
        #jet.puJetIdPassed = jet.puJetId() 
        jet.pfJetIdPassed = jet.jetID('POG_PFID_Loose') 
        if self.cfg_ana.relaxJetId:
            return True
        else:
            return jet.pfJetIdPassed
            #return jet.pfJetIdPassed and (jet.puJetIdPassed or not(self.doPuId)) 
        
    def testJetNoID( self, jet ):
        return jet.pt() > self.cfg_ana.jetPt and \
               abs( jet.eta() ) < self.cfg_ana.jetEta;
 
