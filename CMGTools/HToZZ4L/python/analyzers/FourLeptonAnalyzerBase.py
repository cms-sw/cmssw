from math import *
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi
from PhysicsTools.HeppyCore.framework.event import *

from CMGTools.HToZZ4L.tools.DiObject import DiObject
from CMGTools.HToZZ4L.tools.DiObjectPair import DiObjectPair
from CMGTools.HToZZ4L.tools.OverlapCleaner import OverlapCleaner
from CMGTools.HToZZ4L.tools.CutFlowMaker  import CutFlowMaker

import os
import itertools
import collections
import ROOT

class EventBox(object):
    def __init__(self):
        pass

    def __str__(self):

        header = 'EVENT BOX ---> {type} <------ EVENT BOX'.format( type=self.__class__.__name__)
        varlines = []
        for var,value in sorted(vars(self).iteritems()):
            tmp = value
            # check for recursivity
            recursive = False
            if hasattr(value, '__getitem__'):
                if (len(value)>0 and value[0].__class__ == value.__class__):
                    recursive = True
            if isinstance( value, collections.Iterable ) and \
                   not isinstance(value, (str,unicode)) and \
                   not isinstance(value, TChain) and \
                   not recursive :
                tmp = map(str, value)
            varlines.append( '\t{var:<15}:   {value}'.format(var=var, value=tmp) )
        all = [ header ]
        all.extend(varlines)
        return '\n'.join( all )



        
class FourLeptonAnalyzerBase( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(FourLeptonAnalyzerBase,self).__init__(cfg_ana,cfg_comp,looperName)
        self._MEMs = ROOT.MEMCalculatorsWrapper(13.0)

    def declareHandles(self):
        super(FourLeptonAnalyzerBase, self).declareHandles()

    def beginLoop(self, setup):
        super(FourLeptonAnalyzerBase,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )



    def leptonID_tight(self,lepton):
        return lepton.tightId()

    def leptonID_loose(self,lepton):
        return True    


    def leptonID(self,lepton):
        return self.leptonID_tight(lepton)


    def muonIsolation(self,lepton):
        return lepton.absIsoWithFSR(R=0.4,puCorr="deltaBeta")/lepton.pt()<0.4

    def electronIsolation(self,lepton):
        return lepton.absIsoWithFSR(R=0.4,puCorr="rhoArea")/lepton.pt()<0.5

    def diLeptonMass(self,dilepton):
        return dilepton.M()>12.0 and dilepton.M()<120.

    def fourLeptonMassZ1Z2(self,fourLepton):
        return self.diLeptonMass(fourLepton.leg1) and self.diLeptonMass(fourLepton.leg2)

    def fourLeptonMassZ1(self,fourLepton):
        return fourLepton.leg1.M()>40.0 and fourLepton.leg1.M()<120. # usually implied in fourLeptonMassZ1Z2 but sometimes needed independently

    def stupidCut(self,fourLepton):
        #if not 4mu/4e  pass 
        if abs(fourLepton.leg1.leg1.pdgId())!=abs(fourLepton.leg2.leg1.pdgId()):
            return True

        #print "Nominal, mZ1 %.3f, mZ2 %.3f: %s" % (fourLepton.leg1.M(),fourLepton.leg2.M(),fourLepton)
        #find Alternative pairing.Do not forget FSR
        leptons  = [ fourLepton.leg1.leg1, fourLepton.leg1.leg2, fourLepton.leg2.leg1, fourLepton.leg2.leg2 ]
        quads = []
        for quad in self.findOSSFQuads(leptons, fourLepton.daughterPhotons()): # only re-search for FSR from already-attached photons
            # skip self
            if fourLepton.leg1.leg1 == quad.leg1.leg1 and fourLepton.leg1.leg2 == quad.leg1.leg2 and fourLepton.leg2.leg1 == quad.leg2.leg1:
                continue

            # we used to skip those that fail cuts except Z2 mass
            ### if not self.fourLeptonIsolation(quad) or not self.fourLeptonMassZ1(quad) or not self.qcdSuppression(quad):
            ###    continue
            # however:
            # - we've now decided in the sync that we don't re-check for isolation on the alternate pairing
            # - QCD suppression does not depend on photons, and so it doesn't depend on the pairing
            # - if the new pairing has a better Z1 mass than the original one, then it passes the Z1 mass cut

            # skip those that have a worse Z1 mass than the nominal
            if abs(fourLepton.leg1.M()-91.1876) < abs(quad.leg1.M()-91.1876):
                continue
            #print "Found alternate, mZ1 %.3f, mZ2 %.3f: %s" % (quad.leg1.M(),quad.leg2.M(),quad)
            quads.append(quad)
        if len(quads) == 0:
            #print "No alternates to ",fourLepton
            return True
        bestByZ1 = min(quads, key = lambda quad : abs(quad.leg1.M()-91.1876))
        #print "Best alternate, mZ1 %.3f, mZ2 %.3f: %s" % (bestByZ1.leg1.M(),bestByZ1.leg2.M(),bestByZ1)
        return bestByZ1.leg2.M() > 12.




    def fourLeptonPtThresholds(self, fourLepton):
        leading_pt = fourLepton.sortedPtLeg(0).pt() 
        subleading_pt = fourLepton.sortedPtLeg(1).pt() 
        return leading_pt>20  and subleading_pt>10


    def fourLeptonIsolation(self,fourLepton):
        ##First ! attach the FSR photons of this candidate to the leptons!
        



        leptons = fourLepton.daughterLeptons()
        photons = fourLepton.daughterPhotons()

        


        for l in leptons:
            l.fsrPhotons=[]
            for g in photons:
                if deltaR(g.eta(),g.phi(),l.eta(),l.phi())<0.4:
                    l.fsrPhotons.append(g)
            if abs(l.pdgId())==11:
                if not self.electronIsolation(l):
                    return False
            if abs(l.pdgId())==13:
                if not self.muonIsolation(l):
                    return False
        return True        

    def ghostSuppression(self, fourLepton):
        leptons = fourLepton.daughterLeptons()
        for l1,l2 in itertools.combinations(leptons,2):
            if deltaR(l1.eta(),l1.phi(),l2.eta(),l2.phi())<0.02:
                return False
        return True    



    def qcdSuppression(self, fourLepton):
        return fourLepton.minOSPairMass()>4.0

        
    def zSorting(self,Z1,Z2):
        return abs(Z1.M()-91.1876) <= abs(Z2.M()-91.1876)

    def findOSSFQuads(self, leptons,photons):
        '''Make combinatorics and make permulations of four leptons
           Cut the permutations by asking Z1 nearest to Z and also 
           that plus is the first
           Include FSR if in cfg file
        '''
        out = []
        for l1, l2,l3,l4 in itertools.permutations(leptons, 4):
            if (l1.pdgId()+l2.pdgId())!=0: 
                continue;
            if (l3.pdgId()+l4.pdgId())!=0:
                continue;
            if (l1.pdgId()<l2.pdgId())!=0: 
                continue;
            if (l3.pdgId()<l4.pdgId())!=0: 
                continue;

            quadObject =DiObjectPair(l1, l2,l3,l4)
            self.attachFSR(quadObject,photons)
            if not self.zSorting(quadObject.leg1,quadObject.leg2):
                continue
            out.append(quadObject)

        return out



    def attachFSR(self,quad,photons):
        #first attach photons to the closest leptons
        quad.allPhotonsForFSR = photons # record this for later
        
        legs=[quad.leg1.leg1,quad.leg1.leg2,quad.leg2.leg1,quad.leg2.leg2]

        assocPhotons=[]
        for g in photons:
            for l in legs:
                DR=deltaR(l.eta(),l.phi(),g.eta(),g.phi())
                if DR>0.5:
                    continue;
                if self.cfg_ana.attachFsrToGlobalClosestLeptonOnly:
                    if l != g.globalClosestLepton:
                        continue
                if hasattr(g,'DR'):
                    if DR<g.DR:
                        g.DR=DR
                        g.nearestLepton = l
                else:        
                    g.DR=DR
                    g.nearestLepton = l
            if hasattr(g,'DR'):
                assocPhotons.append(g)

        
        #Now we have the association . Check criteria
        #First on Z1
        z1Photons=[]
        z2Photons=[]

        z1Above4=False
        z2Above4=False
        for g in assocPhotons:
            if g.nearestLepton in [quad.leg1.leg1,quad.leg1.leg2]:
                mll = quad.leg1.M()
                mllg = (quad.leg1.leg1.p4()+quad.leg1.leg2.p4()+g.p4()).M()
                if mllg<4 or mllg>100:
                    continue
                if abs(mllg-91.1876)>abs(mll-91.1876):
                    continue
                z1Photons.append(g)
                if g.pt()>4:
                    z1Above4 = True

            if g.nearestLepton in [quad.leg2.leg1,quad.leg2.leg2]:
                mll = quad.leg2.M()
                mllg = (quad.leg2.leg1.p4()+quad.leg2.leg2.p4()+g.p4()).M()
                if mllg<4 or mllg>100:
                    continue
                if abs(mllg-91.1876)>abs(mll-91.1876):
                    continue
                z2Photons.append(g)
                if g.pt()>4:
                    z2Above4 = True
                
            

        if len(z1Photons)>0:
            if z1Above4: #Take the highest pt
                fsr = max(z1Photons,key=lambda x: x.pt())
                quad.leg1.setFSR(fsr)
            else:    #Take the smallest DR
                fsr = min(z1Photons,key=lambda x: x.DR)
                quad.leg1.setFSR(fsr)
        if len(z2Photons)>0:
            if z2Above4: #Take the highest pt
                fsr = max(z2Photons,key=lambda x: x.pt())
                quad.leg2.setFSR(fsr)
            else:    #Take the smallest DR
                fsr = min(z2Photons,key=lambda x: x.DR)
                quad.leg2.setFSR(fsr)

        quad.updateP4()        
        #cleanup for next combination!        
        for g in assocPhotons:
            del g.DR
            del g.nearestLepton
            
                
    def fillMEs(self,quad):
        legs = [ quad.leg1.leg1, quad.leg1.leg2, quad.leg2.leg1, quad.leg2.leg2 ]
        lvs  = [ ROOT.TLorentzVector(l.px(),l.py(),l.pz(),l.energy()) for l in legs ]

        if hasattr(quad.leg1,'fsrPhoton'):
            photon = ROOT.TLorentzVector(quad.leg1.fsrPhoton.px(),quad.leg1.fsrPhoton.py(),quad.leg1.fsrPhoton.pz(),quad.leg1.fsrPhoton.energy())
            if quad.leg1.fsrDR1() < quad.leg1.fsrDR2():
                lvs[0] = lvs[0]+photon
            else:
                lvs[1]=lvs[1]+photon

        if hasattr(quad.leg2,'fsrPhoton'):
            photon = ROOT.TLorentzVector(quad.leg2.fsrPhoton.px(),quad.leg2.fsrPhoton.py(),quad.leg2.fsrPhoton.pz(),quad.leg2.fsrPhoton.energy())
            if quad.leg2.fsrDR1() < quad.leg2.fsrDR2():
                lvs[2] = lvs[2]+photon
            else:
                lvs[3]=lvs[3]+photon


        ids  = [ l.pdgId() for l in legs ]
        quad.melaAngles = self._MEMs.computeAngles(lvs[0],ids[0], lvs[1],ids[1], lvs[2],ids[2], lvs[3],ids[3])
        self._MEMs.computeAll(lvs[0],ids[0], lvs[1],ids[1], lvs[2],ids[2], lvs[3],ids[3])
        quad.KD = self._MEMs.getKD()
        return True

