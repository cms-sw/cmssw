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
        self.handles['met'] = AutoHandle( 'slimmedMETs', 'std::vector<pat::MET>' )
        self.handles['nopumet'] = AutoHandle( 'slimmedMETs', 'std::vector<pat::MET>' )
        self.handles['cmgCand'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['vertices'] =  AutoHandle( "offlineSlimmedPrimaryVertices", 'std::vector<reco::Vertex>', fallbackLabel="offlinePrimaryVertices" )

    def beginLoop(self, setup):
        super(METAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    def makeTkMETs(self, event):
        event.tkMet = 0  

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
        event.tkMet = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in charged])) , -1.*(sum([x.py() for x in charged])), 0, math.hypot((sum([x.px() for x in charged])),(sum([x.py() for x in charged]))) )
        event.tkMetchs = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedchs])) , -1.*(sum([x.py() for x in chargedchs])), 0, math.hypot((sum([x.px() for x in chargedchs])),(sum([x.py() for x in chargedchs]))) )
        event.tkMetPVLoose = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedPVLoose])) , -1.*(sum([x.py() for x in chargedPVLoose])), 0, math.hypot((sum([x.px() for x in chargedPVLoose])),(sum([x.py() for x in chargedPVLoose]))) )
        event.tkMetPVTight = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedPVTight])) , -1.*(sum([x.py() for x in chargedPVTight])), 0, math.hypot((sum([x.px() for x in chargedPVTight])),(sum([x.py() for x in chargedPVTight]))) )
##        print 'tkmet',event.tkMet.pt(),'tkmetphi',event.tkMet.phi()


    def makeMETNoMu(self, event):
        event.metNoMu = copy.deepcopy(event.met)
        event.metNoMuNoPU = copy.deepcopy(event.metNoPU)

        mupx = 0
        mupy = 0
        #sum muon momentum                                                                                                                                                                                                                            
        for mu in event.selectedMuons:
            mupx += mu.px()
            mupy += mu.py()

        #subtract muon momentum and construct met                                                                                                                                                                                                     
        px,py = event.metNoMu.px()-mupx, event.metNoMu.py()-mupy
        event.metNoMu.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        px,py = event.metNoMuNoPU.px()-mupx, event.metNoMuNoPU.py()-mupy
        event.metNoMuNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))

    def makeMETNoPhoton(self, event):
        event.metNoPhoton = copy.deepcopy(event.met)
        event.metNoPhotonNoPU = copy.deepcopy(event.metNoPU)

        phopx = 0
        phopy = 0
        #sum photon momentum                                                                                                                                                                                                                            
        for pho in event.selectedPhotons:
            phopx += pho.px()
            phopy += pho.py()

        #subtract photon momentum and construct met                                                                                                                                                                                                     
        px,py = event.metNoPhoton.px()-phopx, event.metNoPhoton.py()-phopy
        event.metNoPhoton.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        px,py = event.metNoPhotonNoPU.px()-phopx, event.metNoPhotonNoPU.py()-phopy
        event.metNoPhotonNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))


    def makeMETs(self, event):
        event.met = self.handles['met'].product()[0]
        event.metNoPU = self.handles['nopumet'].product()[0]

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
        event.metraw = event.met.shiftedPt(12, 0)
        event.metType1chs = event.met.shiftedPt(12, 1)

        if self.cfg_ana.recalibrate and hasattr(event, 'deltaMetFromJetSmearing'):
            px,py = event.met.px()+event.deltaMetFromJetSmearing[0], event.met.py()+event.deltaMetFromJetSmearing[1]
            event.met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
            px,py = event.metNoPU.px()+event.deltaMetFromJetSmearing[0], event.metNoPU.py()+event.deltaMetFromJetSmearing[1]
            event.metNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        if self.cfg_ana.recalibrate and hasattr(event, 'deltaMetFromJEC') and event.deltaMetFromJEC[0] != 0 and event.deltaMetFromJEC[1] != 0:
            px,py = event.met.px()+event.deltaMetFromJEC[0], event.met.py()+event.deltaMetFromJEC[1]
            event.met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
            px,py = event.metNoPU.px()+event.deltaMetFromJEC[0], event.metNoPU.py()+event.deltaMetFromJEC[1]
            event.metNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        
        if self.cfg_ana.doMetNoMu and hasattr(event, 'selectedMuons'):
            self.makeMETNoMu(event)

        if self.cfg_ana.doMetNoPhoton and hasattr(event, 'selectedPhotons'):
            self.makeMETNoPhoton(event)

    def process(self, event):
        self.readCollections( event.input)
        self.counters.counter('events').inc('all events')

        self.makeMETs(event)
        event.tkMet = 0 

        if self.cfg_ana.doTkMet: 
            self.makeTkMETs(event);



        return True


setattr(METAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = METAnalyzer,
    recalibrate = True,
    doTkMet = False,
    doMetNoMu = False,  
    doMetNoPhoton = False,  
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    )
)
