import random
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.HeppyCore.utils.deltar import * 
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import JetReCalibrator
import PhysicsTools.HeppyCore.framework.config as cfg

import operator 
import itertools
import copy
from ROOT import TLorentzVector, TVectorD
import ROOT
import math

from copy import deepcopy

class METAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(METAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(METAnalyzer, self).declareHandles()
        self.handles['met'] = AutoHandle( self.cfg_ana.metCollection, 'std::vector<pat::MET>' )
        self.handles['nopumet'] = AutoHandle( self.cfg_ana.noPUMetCollection, 'std::vector<pat::MET>' )
        self.handles['cmgCand'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['vertices'] =  AutoHandle( "offlineSlimmedPrimaryVertices", 'std::vector<reco::Vertex>', fallbackLabel="offlinePrimaryVertices" )

    def beginLoop(self, setup):
        super(METAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    def makeTkMETs(self, event):

        charged = []
        chargedchs = []
        chargedPVLoose = []
        chargedPVTight = []

        pfcands = self.handles['cmgCand'].product()

        for i in xrange(pfcands.size()):

## ===> require the Track Candidate charge and with a  minimum dz 
            
            if (pfcands.at(i).charge()!=0):

                if abs(pfcands.at(i).dz())<=self.cfg_ana.dzMax:
                    charged.append(pfcands.at(i))

                if pfcands.at(i).fromPV()>0:
                    chargedchs.append(pfcands.at(i))

                if pfcands.at(i).fromPV()>1:
                    chargedPVLoose.append(pfcands.at(i))

                if pfcands.at(i).fromPV()>2:
                    chargedPVTight.append(pfcands.at(i))

        import ROOT
        setattr(event, "tkMet"+self.cfg_ana.collectionPostFix, \
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in charged])) , -1.*(sum([x.py() for x in charged])), 0, math.hypot((sum([x.px() for x in charged])),(sum([x.py() for x in charged]))) ))
        setattr(event, "tkMetchs"+self.cfg_ana.collectionPostFix, \ 
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedchs])) , -1.*(sum([x.py() for x in chargedchs])), 0, math.hypot((sum([x.px() for x in chargedchs])),(sum([x.py() for x in chargedchs]))) ))
        setattr(event, "tkMetPVLoose"+self.cfg_ana.collectionPostFix, \ 
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedPVLoose])) , -1.*(sum([x.py() for x in chargedPVLoose])), 0, math.hypot((sum([x.px() for x in chargedPVLoose])),(sum([x.py() for x in chargedPVLoose]))) ))
        setattr(event, "tkMetPVTight"+self.cfg_ana.collectionPostFix, \ 
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedPVTight])) , -1.*(sum([x.py() for x in chargedPVTight])), 0, math.hypot((sum([x.px() for x in chargedPVTight])),(sum([x.py() for x in chargedPVTight]))) ))
##        print 'tkmet',self.tkMet.pt(),'tkmetphi',self.tkMet.phi()


    def makeMETNoMu(self, event):
        self.metNoMu = copy.deepcopy(self.met)
        if self.cfg_ana.doMetNoPU: self.metNoMuNoPU = copy.deepcopy(self.metNoPU)

        mupx = 0
        mupy = 0
        #sum muon momentum                                                                                                                                                                                                                            
        for mu in event.selectedMuons:
            mupx += mu.px()
            mupy += mu.py()

        #subtract muon momentum and construct met                                                                                                                                                                                                     
        px,py = self.metNoMu.px()-mupx, self.metNoMu.py()-mupy
        self.metNoMu.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        px,py = self.metNoMuNoPU.px()-mupx, self.metNoMuNoPU.py()-mupy
        self.metNoMuNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))

    def makeMETNoPhoton(self, event):
        self.metNoPhoton = copy.deepcopy(self.met)

        phopx = 0
        phopy = 0
        #sum photon momentum                                                                                                                                                                                                                            
        for pho in event.selectedPhotons:
            phopx += pho.px()
            phopy += pho.py()

        #subtract photon momentum and construct met                                                                                                                                                                                                     
        px,py = self.metNoPhoton.px()-phopx, self.metNoPhoton.py()-phopy
        self.metNoPhoton.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        if self.cfg_ana.doMetNoPU: 
          self.metNoPhotonNoPU = copy.deepcopy(self.metNoPU)
          px,py = self.metNoPhotonNoPU.px()-phopx, self.metNoPhotonNoPU.py()-phopy
          self.metNoPhotonNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))


    def makeMETs(self, event):
        self.met = self.handles['met'].product()[0]
        if self.cfg_ana.doMetNoPU: self.metNoPU = self.handles['nopumet'].product()[0]

        #Shifted METs
        #Uncertainties defined in https://github.com/cms-sw/cmssw/blob/CMSSW_7_2_X/DataFormats/PatCandidates/interface/MET.h#L168
        for i in range(event.met.METUncertaintySize):
            m = ROOT.pat.MET(event.met)
            px  = m.shiftedPx(i);
            py  = m.shiftedPy(i);
            m.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
            setattr(event, "met_shifted_{0}".format(i), m)
        event.met_sig = event.met.significance()
        event.met_sumet = event.met.sumEt()
        #event.met_sigm = event.met.getSignificanceMatrix()

        ###https://github.com/cms-sw/cmssw/blob/CMSSW_7_2_X/DataFormats/PatCandidates/interface/MET.h
        self.metraw = self.met.shiftedPt(12, 0)
        self.metType1chs = self.met.shiftedPt(12, 1)

        if self.cfg_ana.recalibrate and hasattr(event, 'deltaMetFromJetSmearing'): #FIXME!!!!!!!!!
            px,py = self.met.px()+event.deltaMetFromJetSmearing[0], self.met.py()+event.deltaMetFromJetSmearing[1]
            self.met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
            if self.cfg_ana.doMetNoPU: 
              px,py = self.metNoPU.px()+event.deltaMetFromJetSmearing[0], self.metNoPU.py()+event.deltaMetFromJetSmearing[1]
              self.metNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        if self.cfg_ana.recalibrate and hasattr(event, 'deltaMetFromJEC') and event.deltaMetFromJEC[0] != 0 and event.deltaMetFromJEC[1] != 0:
            px,py = self.met.px()+event.deltaMetFromJEC[0], self.met.py()+event.deltaMetFromJEC[1]
            self.met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
            if self.cfg_ana.doMetNoPU: 
              px,py = self.metNoPU.px()+event.deltaMetFromJEC[0], self.metNoPU.py()+event.deltaMetFromJEC[1]
              self.metNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        
        if self.cfg_ana.doMetNoMu and hasattr(event, 'selectedMuons'):
            self.makeMETNoMu(event)

        if self.cfg_ana.doMetNoPhoton and hasattr(event, 'selectedPhotons'):
            self.makeMETNoPhoton(event)

    def process(self, event):
        self.readCollections( event.input)
        self.counters.counter('events').inc('all events')

        self.makeMETs(event)

        if self.cfg_ana.doTkMet: 
            self.makeTkMETs(event);



        return True


setattr(METAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = METAnalyzer,
    metCollection     = "slimmedMETs",
    noPUMetCollection = "slimmedMETs",
    recalibrate = True,
    doTkMet = False,
    doMetNoPU = True,  
    doMetNoMu = False,  
    doMetNoPhoton = False,  
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    collectionPostFix = "",
    )
)
