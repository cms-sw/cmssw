import operator 
import itertools
import copy
from math import *

from ROOT import std 
from ROOT import TLorentzVector, TVectorD

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

import PhysicsTools.HeppyCore.framework.config as cfg

from PhysicsTools.HeppyCore.utils.deltar import deltaR

from ROOT.heppy import Hemisphere
from ROOT.heppy import ReclusterJets

from ROOT.heppy import Davismt2
davismt2 = Davismt2()

from ROOT.heppy import mt2w_bisect 
mt2wSNT = mt2w_bisect.mt2w()

import ROOT

import os


class MT2Analyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(MT2Analyzer,self).__init__(cfg_ana,cfg_comp,looperName) 
        self.jetPt = cfg_ana.jetPt

    def declareHandles(self):
        super(MT2Analyzer, self).declareHandles()
       #genJets                                                                                                                                                                     
        self.handles['genJets'] = AutoHandle( 'slimmedGenJets','std::vector<reco::GenJet>')
        self.handles['met'] = AutoHandle( self.cfg_ana.metCollection, 'std::vector<pat::MET>' )

    def beginLoop(self, setup):
        super(MT2Analyzer,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')

    def computeMT2(self, visaVec, visbVec, metVec):
        
        import array
        import numpy

        metVector = array.array('d',[0.,metVec.px(), metVec.py()])
        visaVector = array.array('d',[0.,visaVec.px(), visaVec.py()])
        visbVector = array.array('d',[0.,visbVec.px(), visbVec.py()])

        davismt2.set_momenta(visaVector,visbVector,metVector);
        davismt2.set_mn(0);

        return davismt2.get_mt2()

    def getMT2AKT(self, event, TMPobjects40jc, met , collectionPostFix, postFix):

#### get hemispheres via AntiKT -1 antikt, 1 kt, 0 CA
        if len(TMPobjects40jc)>=2:

            objects  = ROOT.std.vector(ROOT.reco.Particle.LorentzVector)()
            for jet in TMPobjects40jc:
                objects.push_back(jet.p4())

            hemisphereViaKt = ReclusterJets(objects, 1.,50.0)
            groupingViaKt=hemisphereViaKt.getGroupingExclusive(2)

            if len(groupingViaKt)>=2:
                setattr(event, "pseudoViaKtJet1"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector(groupingViaKt[0]) )
                setattr(event, "pseudoViaKtJet2"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector(groupingViaKt[1]) )
                setattr(event, "mt2ViaAKt"+collectionPostFix+postFix, self.computeMT2(getattr(event,'pseudoViaKtJet1'+collectionPostFix+postFix), getattr(event,'pseudoViaKtJet2'+collectionPostFix+postFix), met) )
                return self.computeMT2(getattr(event,'pseudoViaKtJet1'+collectionPostFix+postFix), getattr(event,'pseudoViaKtJet2'+collectionPostFix+postFix), met)

            if not self.cfg_ana.doOnlyDefault:
                hemisphereViaAKt = ReclusterJets(objects, -1.,50.0)
                groupingViaAKt=hemisphereViaAKt.getGroupingExclusive(2)

                if len(groupingViaAKt)>=2:
                    setattr(event, "pseudoViaAKtJet1"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector(groupingViaAKt[0]) )
                    setattr(event, "pseudoViaAKtJet2"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector(groupingViaAKt[1]) )
                    setattr(event, "mt2ViaAKt"+collectionPostFix+postFix, self.computeMT2(getattr(event,'pseudoViaAKtJet1'+collectionPostFix+postFix), getattr(event,'pseudoViaAKtJet2'+collectionPostFix+postFix), met) )
                    return self.computeMT2(getattr(event,'pseudoViaAKtJet1'+collectionPostFix+postFix), getattr(event,'pseudoViaAKtJet2'+collectionPostFix+postFix), met)

    def getMT2Hemi(self, event, TMPobjects40jc, met, collectionPostFix, postFix):

        if len(TMPobjects40jc)>=2:

            pxvec  = ROOT.std.vector(float)()
            pyvec  = ROOT.std.vector(float)()
            pzvec  = ROOT.std.vector(float)()
            Evec  = ROOT.std.vector(float)()
            grouping  = ROOT.std.vector(int)()
            
            for jet in TMPobjects40jc:
                pxvec.push_back(jet.px())
                pyvec.push_back(jet.py())
                pzvec.push_back(jet.pz())
                Evec.push_back(jet.energy())

            hemisphere = Hemisphere(pxvec, pyvec, pzvec, Evec, 2, 3)
            grouping=hemisphere.getGrouping()

            pseudoJet1px = 0 
            pseudoJet1py = 0 
            pseudoJet1pz = 0
            pseudoJet1energy = 0 
            multPSJ1 = 0

            pseudoJet2px = 0 
            pseudoJet2py = 0 
            pseudoJet2pz = 0
            pseudoJet2energy = 0 
            multPSJ2 = 0
                
            for index in range(0, len(pxvec)):
                if(grouping[index]==1):
                    pseudoJet1px += pxvec[index]
                    pseudoJet1py += pyvec[index]
                    pseudoJet1pz += pzvec[index]
                    pseudoJet1energy += Evec[index]
                    multPSJ1 += 1
                if(grouping[index]==2):
                    pseudoJet2px += pxvec[index]
                    pseudoJet2py += pyvec[index]
                    pseudoJet2pz += pzvec[index]
                    pseudoJet2energy += Evec[index]                    
                    multPSJ2 += 1

            pseudoJet1pt2 = pseudoJet1px*pseudoJet1px + pseudoJet1py*pseudoJet1py
            pseudoJet2pt2 = pseudoJet2px*pseudoJet2px + pseudoJet2py*pseudoJet2py

            if pseudoJet1pt2 >= pseudoJet2pt2:
                setattr(event, "pseudoJet1"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy ))
                setattr(event, "pseudoJet2"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy ))
                setattr(event, "multPseudoJet1"+collectionPostFix+postFix, multPSJ1 )
                setattr(event, "multPseudoJet2"+collectionPostFix+postFix, multPSJ2 )
            else:
                setattr(event, "pseudoJet2"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector( pseudoJet1px, pseudoJet1py, pseudoJet1pz, pseudoJet1energy ))
                setattr(event, "pseudoJet1"+collectionPostFix+postFix, ROOT.reco.Particle.LorentzVector( pseudoJet2px, pseudoJet2py, pseudoJet2pz, pseudoJet2energy ))
                setattr(event, "multPseudoJet1"+collectionPostFix+postFix, multPSJ2 )
                setattr(event, "multPseudoJet2"+collectionPostFix+postFix, multPSJ1 )

            setattr(event, "mt2"+collectionPostFix+postFix, self.computeMT2(getattr(event,'pseudoJet1'+collectionPostFix+postFix), getattr(event,'pseudoJet2'+collectionPostFix+postFix), met) )
            return self.computeMT2(getattr(event,'pseudoJet1'+collectionPostFix+postFix), getattr(event,'pseudoJet2'+collectionPostFix+postFix), met)


    def makeMT2(self, event):
#        print '==> INSIDE THE PRINT MT2'
#        print 'MET=',event.met.pt()

        self.met = ROOT.pat.MET(self.handles['met'].product()[0])

        import array
        import numpy

        objects40jc = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<2.5 ]
        objectsXjc = [ j for j in event.cleanJets if j.pt() > self.jetPt and abs(j.eta())<2.5 ]

        setattr(event, "mt2ViaKt"+self.cfg_ana.collectionPostFix+"had", -999)
        setattr(event, "mt2ViaKt"+self.cfg_ana.collectionPostFix+"_Xj_had", -999)
        setattr(event, "pseudoViaKtJet1"+self.cfg_ana.collectionPostFix+"_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoViaKtJet2"+self.cfg_ana.collectionPostFix+"_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoViaKtJet1"+self.cfg_ana.collectionPostFix+"_Xj_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoViaKtJet2"+self.cfg_ana.collectionPostFix+"_Xj_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))

        setattr(event, "mt2ViaAKt"+self.cfg_ana.collectionPostFix+"had", -999)
        setattr(event, "mt2ViaAKt"+self.cfg_ana.collectionPostFix+"_Xj_had", -999)
        setattr(event, "pseudoViaAKtJet1"+self.cfg_ana.collectionPostFix+"_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoViaAKtJet2"+self.cfg_ana.collectionPostFix+"_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoViaAKtJet1"+self.cfg_ana.collectionPostFix+"_Xj_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoViaAKtJet2"+self.cfg_ana.collectionPostFix+"_Xj_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))

#### get hemispheres via AntiKT -1 antikt, 1 kt, 0 CA
        if len(objects40jc)>=2:

            self.mt2ViaKt_had=self.getMT2AKT(event, objects40jc, self.met, self.cfg_ana.collectionPostFix, "_had")

        if len(objectsXjc)>=2:

            self.mt2ViaKt_Xj_had=self.getMT2AKT(event, objectsXjc, self.met, self.cfg_ana.collectionPostFix, "_Xj_had")

## ===> hadronic MT2 (as used in the SUS-13-019)
#### get hemispheres (seed 2: max inv mass, association method: default 3 = minimal lund distance)

        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_had", -999)
        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_Xj_had", -999)

        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_Xj_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_Xj_had", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))

        if len(objects40jc)>=2:

            self.mt2_had = self.getMT2Hemi(event,objects40jc, self.met, self.cfg_ana.collectionPostFix, "_had")

        if len(objectsXjc)>=2:

            self.mt2_Xj_had = self.getMT2Hemi(event,objectsXjc, self.met, self.cfg_ana.collectionPostFix, "_Xj_had")

#### do same things for GEN

        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_gen", -999)
        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_Xj_gen", -999)

        if self.cfg_comp.isMC and self.met.genMET():
            allGenJets = [ x for x in self.handles['genJets'].product() ] 
            objects40jc_Gen = [ j for j in allGenJets if j.pt() > 40 and abs(j.eta())<2.5 ]
            objectsXjc_Gen = [ j for j in allGenJets if j.pt() > self.jetPt and abs(j.eta())<2.5 ]

            if len(objects40jc_Gen)>=2:
                self.mt2_gen = self.getMT2Hemi(event,objects40jc_Gen, self.met.genMET(), self.cfg_ana.collectionPostFix, "_gen")

            if len(objectsXjc_Gen)>=2:
                self.mt2_Xj_gen = self.getMT2Hemi(event,objectsXjc_Gen, self.met.genMET(), self.cfg_ana.collectionPostFix, "_Xj_gen")

        else:
            self.mt2_gen = -999.

            
## ===> full MT2 (jets + leptons)
                                                                                                                                                                                             
        objects10lc = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta())<2.5 ]
        if hasattr(event, 'selectedIsoCleanTrack'):
            objects10lc = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta())<2.5 ] + [ t for t in event.selectedIsoCleanTrack ]

        objects40j10lc = objects40jc + objects10lc
        objects40j10lc.sort(key = lambda obj : obj.pt(), reverse = True)

        objectsXj10lc = objectsXjc + objects10lc
        objectsXj10lc.sort(key = lambda obj : obj.pt(), reverse = True)

        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"", -999)
        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_Xj", -999)

        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_Xj", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_Xj", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))

        if len(objects40j10lc)>=2:

            self.mt2 = self.getMT2Hemi(event,objects40j10lc,self.met,self.cfg_ana.collectionPostFix,"") # no postfit since this is the nominal MT2

        if len(objectsXj10lc)>=2:

            self.mt2_Xj = self.getMT2Hemi(event,objectsXj10lc,self.met,self.cfg_ana.collectionPostFix,"_Xj") # no postfit since this is the nominal MT2

## ===> full gamma_MT2

        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_gamma", -999)

        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_gamma", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_gamma", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
            
        if hasattr(event, 'gamma_met'):

            gamma_objects40jc = [ j for j in event.gamma_cleanJets if j.pt() > 40 and abs(j.eta())<2.5 ]
            
            gamma_objects40j10lc = gamma_objects40jc + objects10lc
            
            gamma_objects40j10lc.sort(key = lambda obj : obj.pt(), reverse = True)
            
##        if len(gamma_objects40j10lc)>=2:
            if len(gamma_objects40jc)>=2:
                
                self.gamma_mt2 = self.getMT2Hemi(event,gamma_objects40jc,event.gamma_met,self.cfg_ana.collectionPostFix,"_gamma")

        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_Xj_gamma", -999)
        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_Xj_gamma", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_Xj_gamma", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
            
        if hasattr(event, 'gamma_met'):

            gamma_objectsXjc = [ j for j in event.gamma_cleanJets if j.pt() > self.jetPt and abs(j.eta())<2.5 ]
            
            gamma_objectsXj10lc = gamma_objectsXjc + objects10lc
            
            gamma_objectsXj10lc.sort(key = lambda obj : obj.pt(), reverse = True)
            
            if len(gamma_objectsXjc)>=2:
                
                self.gamma_mt2_Xj = self.getMT2Hemi(event,gamma_objectsXjc,event.gamma_met,self.cfg_ana.collectionPostFix,"_Xj_gamma")



## ===> zll_MT2
        
        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_zll", -999)
        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_zll", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_zll", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
                
        if hasattr(event, 'zll_met'):

            csLeptons = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta()) < 2.5 ]
            
            if len(csLeptons)==2 and len(objects40jc)>=2:
            
                self.zll_mt2 = self.getMT2Hemi(event,objects40jc,event.zll_met,self.cfg_ana.collectionPostFix,"_zll")

        setattr(event, "mt2"+self.cfg_ana.collectionPostFix+"_Xj_zll", -999)
        setattr(event, "pseudoJet1"+self.cfg_ana.collectionPostFix+"_Xj_zll", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))
        setattr(event, "pseudoJet2"+self.cfg_ana.collectionPostFix+"_Xj_zll", ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 ))

        if hasattr(event, 'zll_met'):

            csLeptons = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta()) < 2.5 ]
            
            if len(csLeptons)==2 and len(objectsXjc)>=2:
            
                self.zll_mt2_Xj = self.getMT2Hemi(event,objectsXjc,event.zll_met,self.cfg_ana.collectionPostFix,"_Xj_zll")


#### do the mt2 with one or two b jets (medium CSV)                                                                                                                                                                                                         
        if len(event.bjetsMedium)>=2:

           event.mt2bb = self.computeMT2(event.bjetsMedium[0], event.bjetsMedium[1], self.met)
           event.mt2bb_Xj = self.computeMT2(event.bjetsMedium[0], event.bjetsMedium[1], self.met)
#            print 'MT2bb(2b)',event.mt2bb                                                                                                                                                                                                                 
        if len(event.bjetsMedium)==1:

            objects40jcCSV = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<2.5 and j.p4()!=event.bjetsMedium[0].p4() ]
            objects40jcCSV.sort(key = lambda l : l.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags'), reverse = True)

            objectsXjcCSV = [ j for j in event.cleanJets if j.pt() > self.jetPt and abs(j.eta())<2.5 and j.p4()!=event.bjetsMedium[0].p4() ]
            objectsXjcCSV.sort(key = lambda l : l.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags'), reverse = True)

            if len(objects40jcCSV)>0:
                self.mt2bb = self.computeMT2(event.bjetsMedium[0], objects40jcCSV[0], self.met)
                setattr(event, "mt2bb"+self.cfg_ana.collectionPostFix, self.mt2bb)

            if len(objectsXjcCSV)>0:
                self.mt2bb_Xj = self.computeMT2(event.bjetsMedium[0], objectsXjcCSV[0], self.met)
                setattr(event, "mt2bb_Xj"+self.cfg_ana.collectionPostFix, self.mt2bb_Xj)

##                print 'MT2bb(1b)',event.mt2bb

## ===> leptonic MT2 (as used in the SUS-13-025 )                                                                                                                                                                                                           
        if not self.cfg_ana.doOnlyDefault:
            if len(event.selectedLeptons)>=2:
                self.mt2lep = self.computeMT2(event.selectedLeptons[0], event.selectedLeptons[1], self.met)
                setattr(event, "mt2lep"+self.cfg_ana.collectionPostFix, self.mt2lep)

###

    def process(self, event):
        self.readCollections( event.input )

        event.mt2bb=-999
        event.mt2bb_Xj=-999
        event.mt2lept=-999        

        event.multPseudoJet1_had=0
        event.multPseudoJet2_had=0
        
        event.multPseudoJet1_Xj_had=0
        event.multPseudoJet2_Xj_had=0
 
        ###

        self.makeMT2(event)

#        print 'variables computed: MT=',event.mtw,'MT2=',event.mt2,'MT2W=',event.mt2w
#        print 'pseudoJet1 px=',event.pseudoJet1.px(),' py=',event.pseudoJet1.py(),' pz=',event.pseudoJet1.pz()
#        print 'pseudoJet2 px=',event.pseudoJet2.px(),' py=',event.pseudoJet2.py(),' pz=',event.pseudoJet2.pz()   

        return True



setattr(MT2Analyzer,"defaultConfig", cfg.Analyzer(
    class_object = MT2Analyzer,
    metCollection     = "slimmedMETs",
    collectionPostFix = "",
    doOnlyDefault = True,
    )
)
