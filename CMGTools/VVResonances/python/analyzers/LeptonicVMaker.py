import random
import math
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.utils.deltar import *
import PhysicsTools.HeppyCore.framework.config as cfg
from  itertools import combinations
from CMGTools.VVResonances.tools.Pair import *


class LeptonicVMaker( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(LeptonicVMaker,self).__init__(cfg_ana, cfg_comp, looperName)
        self.zMassLimits = cfg_ana.zMassLimits
        self.wMTLimits   = cfg_ana.wMTLimits
        

    def declareHandles(self):
        super(LeptonicVMaker, self).declareHandles()


    def makeDiLeptons(self,leptonList):
        output=[]
        for l1,l2 in combinations(leptonList,2):
            if  (l1.pdgId() == -l2.pdgId()):
                pair = Pair(l1,l2,23)
                m=pair.p4().mass()
                if m>self.zMassLimits[0] and  m<self.zMassLimits[1]:
#                    print 'New Z with mass ',m
                    output.append(pair)
        return output            

    def makeLeptonsMET(self,leptonList,MET):
        output=[]
        for l1 in leptonList:
            pair = Pair(l1,MET,l1.charge()*24)
            mt=pair.mt()
            if mt>self.wMTLimits[0] and  mt<self.wMTLimits[1]:
#                    print 'New W with mt ',mt

                    output.append(pair)
        return output            

    
    def beginLoop(self, setup):
        super(LeptonicVMaker,self).beginLoop(setup)
        
    def process(self, event):
        self.readCollections( event.input )
        

        #make all first
        event.allLL=self.makeDiLeptons(event.selectedLeptons)
        event.LL = event.allLL
        event.allLNu=self.makeLeptonsMET(event.selectedLeptons,event.met)
        
        
        #now make Z first . for the remaining leptons after Z make W
        leptonsSet = set(event.selectedLeptons)
        used = []
        for z in event.LL:
            used.extend([z.leg1,z.leg2])
        usedSet = set(used)

        remaining = leptonsSet-usedSet
        event.LNu = self.makeLeptonsMET(list(remaining),event.met)
        return True



        
            

        


                
                
