from math import *
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi
from PhysicsTools.Heppy.physicsobjects.PhysicsObject import PhysicsObject

from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

import ROOT
from ROOT import heppy

import os
import itertools
import numpy
        
class GenFSRAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(GenFSRAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

        self.file = ROOT.TFile("fsrTree.root", 'recreate' )
        self.tree = ROOT.TTree("tree","tree")
        self.vars={}

        self.vars['mZ']=numpy.zeros(1,float)
        self.tree.Branch('mZ',self.vars['mZ'],'mZ/D')

        self.vars['mllgamma']=numpy.zeros(1,float)
        self.tree.Branch('mllgamma',self.vars['mllgamma'],'mllgamma/D')

        self.vars['mll']=numpy.zeros(1,float)
        self.tree.Branch('mll',self.vars['mll'],'mll/D')

        self.vars['mStar']=numpy.zeros(1,float)
        self.tree.Branch('mStar',self.vars['mStar'],'mStar/D')

        self.vars['P']=numpy.zeros(1,float)
        self.tree.Branch('P',self.vars['P'],'P/D')

        self.vars['E']=numpy.zeros(1,float)
        self.tree.Branch('E',self.vars['E'],'E/D')

        self.vars['ET']=numpy.zeros(1,float)
        self.tree.Branch('ET',self.vars['ET'],'ET/D')

        self.vars['cosTheta']=numpy.zeros(1,float)
        self.tree.Branch('cosTheta',self.vars['cosTheta'],'cosTheta/D')

        self.vars['DR']=numpy.zeros(1,float)
        self.tree.Branch('DR',self.vars['DR'],'DR/D')



    def declareHandles(self):
        pass
    
    def beginLoop(self, setup):
        super(GenFSRAnalyzer,self).beginLoop(setup)


    def fetchFinalStateLeptons(self,particle):
        finalState=[]
        for i in range(0,particle.numberOfDaughters()):
            d=particle.daughter(i)
            if d.status()==1 and abs(d.pdgId()) in [11,13]:
                finalState.append(d)
            else:
                finalState=finalState+self.fetchFinalStateLeptons(d)
        return finalState
    
    def process(self, event):


        for l in event.genParticles:
            if abs(l.pdgId()) in [11,13] and l.status()==2:
                hasFSR=False
                for d in range(0,l.numberOfDaughters()):
                    daughter=l.daughter(d)
                    if daughter.pdgId()==22:
                        hasFSR=True
                if hasFSR:        
                    self.vars['mZ'][0] = l.mother(0).mother(0).mass()
                    self.vars['mStar'][0] =(l.daughter(0).p4()+l.daughter(1).p4()).M()
                    self.vars['P'][0] =l.daughter(0).p()
                    self.vars['E'][0] =l.daughter(1).energy()
                    self.vars['ET'][0] =l.daughter(1).pt()
                    self.vars['cosTheta'][0] =l.daughter(0).p4().Vect().Dot(l.daughter(1).p4().Vect())/(l.daughter(0).p()*l.daughter(1).energy())
                    self.vars['DR'][0] =deltaR(l.daughter(0).eta(),l.daughter(0).phi(),l.daughter(1).eta(),l.daughter(1).phi())

                    finalStateLeptons=self.fetchFinalStateLeptons(l.mother(0).mother(0))
                    if len(finalStateLeptons)<2:
                        continue;
                    print 'Final state leptons',len(finalStateLeptons)
                    self.vars['mll'][0] =(finalStateLeptons[0].p4()+finalStateLeptons[1].p4()).M() 
                    self.vars['mllgamma'][0] =(finalStateLeptons[0].p4()+finalStateLeptons[1].p4()+l.daughter(1).p4()).M() 
                    self.tree.Fill()
        return False

    def write(self,setup):
        super(GenFSRAnalyzer, self).write(setup)
        self.file.cd()
        self.tree.Write()
        self.file.Close()
        
