import operator 
import itertools
import copy
from math import *

from ROOT import std 
from ROOT import TLorentzVector, TVector3, TVectorD

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

from PhysicsTools.HeppyCore.utils.deltar import deltaR

from ROOT.heppy import Megajet
from ROOT.heppy import ReclusterJets

import ROOT

import os

class RazorAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(RazorAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 

    def declareHandles(self):
        super(RazorAnalyzer, self).declareHandles()
       #genJets                                                                                                                                                                     
        self.handles['genJets'] = AutoHandle( 'slimmedGenJets','std::vector<reco::GenJet>')

    def beginLoop(self, setup):
        super(RazorAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')

    def computeMR(self, ja, jb):
        A = ja.P();
        B = jb.P();
        az = ja.Pz();
        bz = jb.Pz();
        mr = sqrt((A+B)*(A+B)-(az+bz)*(az+bz));
        return mr

    def computeMTR(self, ja, jb, met):
        
        mtr = met.Vect().Mag()*(ja.Pt()+jb.Pt()) - met.Vect().Dot(ja.Vect()+jb.Vect());
        mtr = sqrt(mtr/2.);
        return mtr;

    def computeR(self, ja, jb, met):

        mr = self.computeMR(ja,jb)
        mtr = self.computeMTR(ja,jb,met)
        r = 999999. if mr <= 0 else mtr/mr 
        return r


    def makeRAZOR(self, event):
#        print '==> INSIDE THE PRINT MT2'
#        print 'MET=',event.met.pt() 

        import array
        import numpy


## ===> hadronic RAZOR 

        (met, metphi)  = event.met.pt(), event.met.phi()
        metp4 = ROOT.TLorentzVector()
        metp4.SetPtEtaPhiM(met,0,metphi,0)

        objects40jc = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<2.5 ]

#### get megajets (association method: default 1 = minimum sum of the invariant masses of the two megajets)

        if len(objects40jc)>=2:

            pxvec  = ROOT.std.vector(float)()
            pyvec  = ROOT.std.vector(float)()
            pzvec  = ROOT.std.vector(float)()
            Evec  = ROOT.std.vector(float)()
            grouping  = ROOT.std.vector(int)()
            
            for jet in objects40jc:
                pxvec.push_back(jet.px())
                pyvec.push_back(jet.py())
                pzvec.push_back(jet.pz())
                Evec.push_back(jet.energy())

            megajet = Megajet(pxvec, pyvec, pzvec, Evec, 1)

            pseudoJet1px = megajet.getAxis1()[0] * megajet.getAxis1()[3]
            pseudoJet1py = megajet.getAxis1()[1] * megajet.getAxis1()[3]
            pseudoJet1pz = megajet.getAxis1()[2] * megajet.getAxis1()[3]
            pseudoJet1energy = megajet.getAxis1()[4]

            pseudoJet2px = megajet.getAxis2()[0] * megajet.getAxis2()[3]
            pseudoJet2py = megajet.getAxis2()[1] * megajet.getAxis2()[3]
            pseudoJet2pz = megajet.getAxis2()[2] * megajet.getAxis2()[3]
            pseudoJet2energy = megajet.getAxis2()[4]

            pseudoJet1pt2 = pseudoJet1px*pseudoJet1px + pseudoJet1py*pseudoJet1py
            pseudoJet2pt2 = pseudoJet2px*pseudoJet2px + pseudoJet2py*pseudoJet2py

            if pseudoJet1pt2 >= pseudoJet2pt2:
                event.pseudoJet1_had = ROOT.TLorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy)
                event.pseudoJet2_had = ROOT.TLorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy)
            else:
                event.pseudoJet2_had = ROOT.TLorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy)
                event.pseudoJet1_had = ROOT.TLorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy)

            event.mr_had = self.computeMR(event.pseudoJet1_had, event.pseudoJet2_had)
            event.mtr_had = self.computeMTR(event.pseudoJet1_had, event.pseudoJet2_had, metp4)
            event.r_had = self.computeR(event.pseudoJet1_had, event.pseudoJet2_had, metp4)

#### do same things for GEN

        if self.cfg_comp.isMC:
            (genmet, genmetphi)  = event.met.genMET().pt(), event.met.genMET().phi()
            genmetp4 = ROOT.TLorentzVector()
            genmetp4.SetPtEtaPhiM(genmet,0,genmetphi,0)

            allGenJets = [ x for x in self.handles['genJets'].product() ] 
            objects40jc_Gen = [ j for j in allGenJets if j.pt() > 40 and abs(j.eta())<2.5 ]

            if len(objects40jc_Gen)>=2:
     
                pxvec  = ROOT.std.vector(float)()
                pyvec  = ROOT.std.vector(float)()
                pzvec  = ROOT.std.vector(float)()
                Evec  = ROOT.std.vector(float)()
                grouping  = ROOT.std.vector(int)()
                
                for jet in objects40jc_Gen:
                    pxvec.push_back(jet.px())
                    pyvec.push_back(jet.py())
                    pzvec.push_back(jet.pz())
                    Evec.push_back(jet.energy())
     
                megajet = Megajet(pxvec, pyvec, pzvec, Evec, 1)
     
                pseudoJet1px = megajet.getAxis1()[0] * megajet.getAxis1()[3]
                pseudoJet1py = megajet.getAxis1()[1] * megajet.getAxis1()[3]
                pseudoJet1pz = megajet.getAxis1()[2] * megajet.getAxis1()[3]
                pseudoJet1energy = megajet.getAxis1()[4]
     
                pseudoJet2px = megajet.getAxis2()[0] * megajet.getAxis2()[3]
                pseudoJet2py = megajet.getAxis2()[1] * megajet.getAxis2()[3]
                pseudoJet2pz = megajet.getAxis2()[2] * megajet.getAxis2()[3]
                pseudoJet2energy = megajet.getAxis2()[4]
     
                pseudoJet1pt2 = pseudoJet1px*pseudoJet1px + pseudoJet1py*pseudoJet1py
                pseudoJet2pt2 = pseudoJet2px*pseudoJet2px + pseudoJet2py*pseudoJet2py
     
                if pseudoJet1pt2 >= pseudoJet2pt2:
                    pseudoJet1_gen = ROOT.TLorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy)
                    pseudoJet2_gen = ROOT.TLorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy)
                else:
                    pseudoJet2_gen = ROOT.TLorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy)
                    pseudoJet1_gen = ROOT.TLorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy)
     
                event.mr_gen = self.computeMR(pseudoJet1_gen, pseudoJet2_gen)
                event.mtr_gen = self.computeMTR(pseudoJet1_gen, pseudoJet2_gen, genmetp4)
                event.r_gen = self.computeR(pseudoJet1_gen, pseudoJet2_gen, genmetp4)
        else:
            event.mr_gen = -999
            event.mtr_gen = -999
            event.r_gen = -999

            
## ===> full RAZOR (jets + leptons)                                                                                                                                                                                             
        objects10lc = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta())<2.5 ]
        if hasattr(event, 'selectedIsoCleanTrack'):
            objects10lc = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta())<2.5 ] + [ t for t in event.selectedIsoCleanTrack ]

        objects40j10lc = objects40jc + objects10lc

        objects40j10lc.sort(key = lambda obj : obj.pt(), reverse = True)

        if len(objects40j10lc)>=2:

            pxvec  = ROOT.std.vector(float)()
            pyvec  = ROOT.std.vector(float)()
            pzvec  = ROOT.std.vector(float)()
            Evec  = ROOT.std.vector(float)()
            grouping  = ROOT.std.vector(int)()

            for obj in objects40j10lc:
                pxvec.push_back(obj.px())
                pyvec.push_back(obj.py())
                pzvec.push_back(obj.pz())
                Evec.push_back(obj.energy())

            #for obj in objects_fullmt2:
            #    print "pt: ", obj.pt(), ", eta: ", obj.eta(), ", phi: ", obj.phi(), ", mass: ", obj.mass()

            #### get megajets  (association method: default 1 = minimum sum of the invariant masses of the two megajets)

            megajet = Megajet(pxvec, pyvec, pzvec, Evec, 1)

            pseudoJet1px = megajet.getAxis1()[0] * megajet.getAxis1()[3]
            pseudoJet1py = megajet.getAxis1()[1] * megajet.getAxis1()[3]
            pseudoJet1pz = megajet.getAxis1()[2] * megajet.getAxis1()[3]
            pseudoJet1energy = megajet.getAxis1()[4]

            pseudoJet2px = megajet.getAxis2()[0] * megajet.getAxis2()[3]
            pseudoJet2py = megajet.getAxis2()[1] * megajet.getAxis2()[3]
            pseudoJet2pz = megajet.getAxis2()[2] * megajet.getAxis2()[3]
            pseudoJet2energy = megajet.getAxis2()[4]

            pseudoJet1pt2 = pseudoJet1px*pseudoJet1px + pseudoJet1py*pseudoJet1py
            pseudoJet2pt2 = pseudoJet2px*pseudoJet2px + pseudoJet2py*pseudoJet2py

            if pseudoJet1pt2 >= pseudoJet2pt2:
                event.pseudoJet1 = ROOT.TLorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy)
                event.pseudoJet2 = ROOT.TLorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy)
            else:
                event.pseudoJet2 = ROOT.TLorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy)
                event.pseudoJet1 = ROOT.TLorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy)

            ###

            event.mr = self.computeMR(event.pseudoJet1, event.pseudoJet2)
            event.mtr = self.computeMTR(event.pseudoJet1, event.pseudoJet2, metp4)
            event.r = self.computeR(event.pseudoJet1, event.pseudoJet2, metp4)



#### do the razor with one or two b jets (medium CSV)                                                                                                                                                                                                         
        if len(event.bjetsMedium)>=2:

            bJet1 = ROOT.TLorentzVector(event.bjetsMedium[0].px(), event.bjetsMedium[0].py(), event.bjetsMedium[0].pz(), event.bjetsMedium[0].energy())
            bJet2 = ROOT.TLorentzVector(event.bjetsMedium[1].px(), event.bjetsMedium[1].py(), event.bjetsMedium[1].pz(), event.bjetsMedium[1].energy())

            event.mr_bb  = self.computeMR(bJet1, bJet2)
            event.mtr_bb = self.computeMTR(bJet1, bJet2, metp4)
            event.r_bb   = self.computeR(bJet1, bJet2, metp4)

#            print 'MR(2b)',event.mr_bb                                                                                                                                                                                                                 
        if len(event.bjetsMedium)==1:

            objects40jcCSV = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<2.5 and j.p4()!=event.bjetsMedium[0].p4() ]
            objects40jcCSV.sort(key = lambda l : l.btag('combinedInclusiveSecondaryVertexV2BJetTags'), reverse = True)

            if len(objects40jcCSV)>0:
                
                bJet1 = ROOT.TLorentzVector(event.bjetsMedium[0].px(), event.bjetsMedium[0].py(), event.bjetsMedium[0].pz(), event.bjetsMedium[0].energy())
                bJet2 = ROOT.TLorentzVector(objects40jcCSV[0].px(), objects40jcCSV[0].py(), objects40jcCSV[0].pz(), objects40jcCSV[0].energy())

                event.mr_bb = self.computeMR(bJet1, bJet2)
                event.mtr_bb = self.computeMTR(bJet1, bJet2, metp4)
                event.r_bb = self.computeR(bJet1, bJet2, metp4)
##                print 'MRbb(1b)',event.mr_bb

## ===> leptonic MR 
        if not self.cfg_ana.doOnlyDefault:
            if len(event.selectedLeptons)>=2:

                lep1 = ROOT.TLorentzVector(event.selectedLeptons[0].px(), event.selectedLeptons[0].py(), event.selectedLeptons[0].pz(), event.selectedLeptons[0].energy())
                lep2 = ROOT.TLorentzVector(event.selectedLeptons[1].px(), event.selectedLeptons[1].py(), event.selectedLeptons[1].pz(), event.selectedLeptons[1].energy())

                event.mr_lept = self.computeMR(lep1, lep2)
                event.mtr_lept = self.computeMTR(lep1, lep2, metp4)
                event.r_lept = self.computeR(lep1, lep2, metp4)



###

    def process(self, event):
        self.readCollections( event.input )

        event.mr_gen=-999
        event.mtr_gen=-999
        event.r_gen=-999

        event.mr_bb=-999
        event.mtr_bb=-999
        event.r_bb=-999

        event.mr_lept=-999        
        event.mtr_lept=-999        
        event.r_lept=-999        

        event.mr_had=-999
        event.mtr_had=-999
        event.r_had=-999

        event.mr=-999
        event.mtr=-999
        event.r=-999

        event.pseudoJet1 = ROOT.TLorentzVector( 0, 0, 0, 0 )
        event.pseudoJet2 = ROOT.TLorentzVector( 0, 0, 0, 0 )
        
        ###

        self.makeRAZOR(event)

#        print 'variables computed: MR=',event.mr_had,'R=',event.r,'MTR=',event.mtr
#        print 'pseudoJet1 px=',event.pseudoJet1.px(),' py=',event.pseudoJet1.py(),' pz=',event.pseudoJet1.pz()
#        print 'pseudoJet2 px=',event.pseudoJet2.px(),' py=',event.pseudoJet2.py(),' pz=',event.pseudoJet2.pz()   

        return True
