from math import *
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi
from PhysicsTools.Heppy.physicsobjects.PhysicsObject import PhysicsObject


from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle


from ROOT import heppy

import os
import itertools
        
class FSRPhotonMaker( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(FSRPhotonMaker,self).__init__(cfg_ana,cfg_comp,looperName)
        self.leptonTag  = cfg_ana.leptons
        self.electronID = cfg_ana.electronID
        self.IsolationComputer = heppy.IsolationComputer(0.3)

    def declareHandles(self):
        super(FSRPhotonMaker, self).declareHandles()
        self.handles['pf'] = AutoHandle( "packedPFCandidates",'std::vector<pat::PackedCandidate>')

    def beginLoop(self, setup):
        super(FSRPhotonMaker,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        pf = map( PhysicsObject, self.handles['pf'].product() )
        leptons = getattr(event,self.leptonTag)
        self.IsolationComputer.setPackedCandidates(self.handles['pf'].product())


        #first trim the photons that are only near leptons
        direct=[]
        forIso=[]


        for p in pf:
            if p.pdgId() != 22 or not( p.pt() > 2.0 and abs(p.eta()) < 2.4 ):
                continue
            scVetoed = False
            for l in leptons:
                if abs(l.pdgId())==11 and self.electronID(l):
                    #print "Testing photon pt %5.1f eta %+7.4f phi %+7.4f vs ele pt %.1f eta %+7.4f  phi %+7.4f sc  eta %+7.4f  phi %+7.4f" % ( p.pt(), p.eta(), p.phi(), l.pt(), l.eta(), l.phi(), l.superCluster().eta(), l.superCluster().phi() )
                    #print "Testing                                                       deta %+7.4f dphi %+7.4f sc deta %+7.4f dphi %+7.4f" % ( abs(p.eta()-l.eta()), deltaPhi(p.phi(),l.phi()), abs(p.eta()-l.superCluster().eta()), deltaPhi(p.phi(),l.superCluster().phi()))
                    if self.cfg_ana.electronVeto == "superclusterEta":
                        if (abs(p.eta()-l.superCluster().eta())<0.05 and abs(deltaPhi(p.phi(),l.superCluster().phi()))<2) or deltaR(p.eta(),p.phi(),l.superCluster().eta(),l.superCluster().phi())<0.15: 
                            scVetoed = True; break
                    elif self.cfg_ana.electronVeto == "electronEta":
                        if (abs(p.eta()-l.eta())<0.05 and abs(deltaPhi(p.phi(),l.phi()))<2) or deltaR(p.eta(),p.phi(),l.eta(),l.phi())<0.15: 
                            scVetoed = True; break
                    else: 
                        raise RuntimeError, "electronVeto option %r not implemented" % self.cfg_ana.electronVeto
            if scVetoed: continue
            okNoIso = False; okIfIso = False
            for l in leptons:
                DR =deltaR(l.eta(),l.phi(),p.eta(),p.phi())     
                if DR<0.07 and p.pt()>2.0:
                    direct.append(p)
                    okNoIso = True
                    break;
                elif  DR<0.5 and p.pt()>4.0:   
                    okIfIso = True
            if okIfIso and not okNoIso:
                forIso.append(p)
        isolatedPhotons=[]        
        for g in forIso:
            g.absIsoCH = self.IsolationComputer.chargedAbsIso(g.physObj,0.3,0.0001,0.2)
            g.absIsoPU = self.IsolationComputer.puAbsIso(g.physObj,0.3,0.0001,0.2)
            g.absIsoNH = self.IsolationComputer.neutralHadAbsIsoRaw(g.physObj,0.3,0.01,0.5)
            g.absIsoPH = self.IsolationComputer.photonAbsIsoRaw(g.physObj,0.3,0.01,0.5)
            g.relIso   = (g.absIsoCH+g.absIsoPU+g.absIsoNH+g.absIsoPH)/g.pt()
            if g.relIso<1.0:
                isolatedPhotons.append(g)

        event.fsrPhotons = isolatedPhotons+direct
                    
        # save all, for debugging
        event.fsrPhotonsNoIso = forIso + direct
        for fsr in event.fsrPhotonsNoIso:
            closest = min(leptons, key = lambda l : deltaR(fsr.eta(),fsr.phi(),l.eta(),l.phi()))
            fsr.globalClosestLepton = closest

        return True
        
