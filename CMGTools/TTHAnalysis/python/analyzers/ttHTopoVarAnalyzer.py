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

from PhysicsTools.HeppyCore.utils.deltar import deltaR

from ROOT.heppy import mt2w_bisect 
mt2wSNT = mt2w_bisect.mt2w()

import ROOT

import os

def mtw(x1,x2):
    return sqrt(2*x1.pt()*x2.pt()*(1-cos(x1.phi()-x2.phi())))

class ttHTopoVarAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHTopoVarAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 
        self.jetPt = cfg_ana.jetPt

    def declareHandles(self):
        super(ttHTopoVarAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(ttHTopoVarAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')

    def makeMinMT(self,event):

#        objectsb40jc = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<2.5 and j.btagWP("CSVv2IVFM")]
#
#        if len(objectsb40jc)>0:
#            for bjet in objectsb40jc:
#                mtTemp = mtw(bjet, event.met)
#                event.minMTBMet = min(event.minMTBMet,mtTemp)

        objectsbXjc = [ j for j in event.cleanJets if j.pt() > self.jetPt and abs(j.eta())<2.5 and j.btagWP("CSVv2IVFM")]

        if len(objectsbXjc)>0:
            for bjet in objectsbXjc:
                mtTemp = mtw(bjet, event.met)
                event.minMTBMet = min(event.minMTBMet,mtTemp)


    def makeMinMTGamma(self,event):

#        gamma_objectsb40jc = [ j for j in event.gamma_cleanJets if j.pt() > 40 and abs(j.eta())<2.5 and j.btagWP("CSVv2IVFM")]
#
#        if len(gamma_objectsb40jc)>0:
#            for bjet in gamma_objectsb40jc:
#                mtTemp = mtw(bjet, event.gamma_met)
#                event.gamma_minMTBMet = min(event.gamma_minMTBMet,mtTemp)

        gamma_objectsbXjc = [ j for j in event.gamma_cleanJets if j.pt() > self.jetPt and abs(j.eta())<2.5 and j.btagWP("CSVv2IVFM")]

        if len(gamma_objectsbXjc)>0:
            for bjet in gamma_objectsbXjc:
                mtTemp = mtw(bjet, event.gamma_met)
                event.gamma_minMTBMet = min(event.gamma_minMTBMet,mtTemp)


    def makeMT2W(self, event):
#        print '==> INSIDE THE PRINT MT2'
#        print 'MET=',event.met.pt()

        import array
        import numpy

## ===> hadronic MT2w (as used in the SUS-13-011) below just a placeHolder to be coded properly

        if not self.cfg_ana.doOnlyDefault:
            if len(event.selectedLeptons)>=1:

                metVector = TVectorD(3,array.array('d',[0.,event.met.px(), event.met.py()]))
                lVector = TVectorD(3,array.array('d',[0.,event.selectedLeptons[0].px(), event.selectedLeptons[0].py()]))
                #placeholder for visaVector and visbVector  need to get the jets
                visaVector = TVectorD(3,array.array('d',[0.,event.selectedLeptons[0].px(), event.selectedLeptons[0].py()]))
                visbVector = TVectorD(3,array.array('d',[0.,event.selectedLeptons[0].px(), event.selectedLeptons[0].py()]))

                metVector =numpy.asarray(metVector,dtype='double')
                lVector =numpy.asarray(lVector,dtype='double')
                visaVector =numpy.asarray(visaVector,dtype='double')
                visbVector =numpy.asarray(visbVector,dtype='double')

                mt2wSNT.set_momenta(lVector, visaVector,visbVector,metVector);
                event.mt2w = mt2wSNT.get_mt2w() 



    def process(self, event):
        self.readCollections( event.input )

#        event.mt2w=-999

        ###

        event.minMTBMet=999999
        self.makeMinMT(event)

        event.gamma_minMTBMet=999999
        self.makeMinMTGamma(event)

        event.zll_minMTBMet=999999
        csLeptons = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta()) < 2.5 ]
        if len(csLeptons)==2:
            event.zll_minMTBMet=event.minMTBMet
#        print 'variables computed: MT=',event.mtw,'MT2=',event.mt2,'MT2W=',event.mt2w
#        print 'pseudoJet1 px=',event.pseudoJet1.px(),' py=',event.pseudoJet1.py(),' pz=',event.pseudoJet1.pz()
#        print 'pseudoJet2 px=',event.pseudoJet2.px(),' py=',event.pseudoJet2.py(),' pz=',event.pseudoJet2.pz()   

        return True
