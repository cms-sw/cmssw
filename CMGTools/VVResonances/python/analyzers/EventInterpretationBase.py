import random
import math
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.utils.deltar import *
import PhysicsTools.HeppyCore.framework.config as cfg
from CMGTools.VVResonances.tools.PyJetToolbox import *




class EventInterpretationBase( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(EventInterpretationBase,self).__init__(cfg_ana, cfg_comp, looperName)
        self.selectFat = self.cfg_ana.selectFat
        self.selectPair = self.cfg_ana.selectPair
        self.isMC =cfg_comp.isMC
        if hasattr(cfg_ana,'matchDR'):
            self.matchDR = cfg_ana.matchDR
        else:
            self.matchDR = 0.2
            
    def declareHandles(self):
        super(EventInterpretationBase, self).declareHandles()
        self.handles['packed'] = AutoHandle( 'packedPFCandidates', 'std::vector<pat::PackedCandidate>' )

    def removeLeptonFootPrint(self,leptons,cands):
        toRemove=[]
        for lepton in leptons:
            for p in range(0,lepton.numberOfSourceCandidatePtrs()):
                toRemove.append(cands[lepton.sourceCandidatePtr(p).key()])
        return list(set(cands)-set(toRemove))


    def matchSubJets(self,jets,genquarks):
        for j in jets:
            for s in j.subjets:
                s.mcquark = None
                s.matched=0
                for g in genquarks:
                    if deltaR(s.eta(),s.phi(),g.eta(),g.phi())<self.matchDR:
                        s.matched=1
                        s.mcquark = g
                        break;
                    
    def makeFatJets(self,cands):
        toolboxFat  = PyJetToolbox(cands)
        toolboxFat.setInterface(True,self.cfg_ana.ktPowerFat,self.cfg_ana.rFat)
        toolboxFat.setMassDrop(self.cfg_ana.massdrop)
        toolboxFat.setSubjets(True,'inc',self.cfg_ana.subjets)
        toolboxFat.setPruning(self.cfg_ana.prunning)
        toolboxFat.setNtau(True)
        toolboxFat.setSoftDrop(self.cfg_ana.softdrop)
        # Lets cluster !! Fat jets first
        fatJets=toolboxFat.inclusiveJets(100.0,True)
        return filter(self.selectFat,fatJets)

    def makeSatelliteJets(self,cands):
        toolbox  = PyJetToolbox(cands)
        toolbox.setInterface(True,self.cfg_ana.ktPower,self.cfg_ana.r)
        toolbox.setMassDrop(False)
        toolbox.setSubjets(False,'inc',2)
        toolbox.setPruning(False)
        toolbox.setNtau(False)
        toolbox.setSoftDrop(False)
        return toolbox.inclusiveJets(30.0,False)

    def removeJetFootPrint(self,jets,cands):
        toRemove=[]
        for j in jets:
            toRemove.extend(j.constituents)
        return list(set(cands)-set(toRemove))


    def vbfTopology(self,obj):
        if len(obj['satelliteJets'])<2:
            obj['vbfDEta'] = -1.0
            obj['vbfMJJ'] = -1.0
        else:            
            j1 = obj['satelliteJets'][0]
            j2 = obj['satelliteJets'][1]
            obj['vbfDEta'] = abs(j1.eta()-j2.eta())
            obj['vbfMJJ'] = (j1.p4()+j2.p4()).M()
    
    def beginLoop(self, setup):
        super(EventInterpretationBase,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        #read the packed candidates
        event.packedCandidatesForJets = self.handles['packed'].product()
            

        


                
                
