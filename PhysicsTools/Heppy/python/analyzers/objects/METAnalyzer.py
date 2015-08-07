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
        self.mchandles['packedGen'] = AutoHandle( 'packedGenParticles', 'std::vector<pat::PackedGenParticle>' )

    def beginLoop(self, setup):
        super(METAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    def applyDeltaMet(self, met, deltaMet):
        px,py = self.met.px()+deltaMet[0], self.met.py()+deltaMet[1]
        met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))

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
        setattr(event, "tkMetPVchs"+self.cfg_ana.collectionPostFix, \
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedchs])) , -1.*(sum([x.py() for x in chargedchs])), 0, math.hypot((sum([x.px() for x in chargedchs])),(sum([x.py() for x in chargedchs]))) ))
        setattr(event, "tkMetPVLoose"+self.cfg_ana.collectionPostFix, \
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedPVLoose])) , -1.*(sum([x.py() for x in chargedPVLoose])), 0, math.hypot((sum([x.px() for x in chargedPVLoose])),(sum([x.py() for x in chargedPVLoose]))) ))
        setattr(event, "tkMetPVTight"+self.cfg_ana.collectionPostFix, \
          ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in chargedPVTight])) , -1.*(sum([x.py() for x in chargedPVTight])), 0, math.hypot((sum([x.px() for x in chargedPVTight])),(sum([x.py() for x in chargedPVTight]))) ))
##        print 'tkmet',self.tkMet.pt(),'tkmetphi',self.tkMet.phi()


        event.tkSumEt = sum([x.pt() for x in charged])
        event.tkPVchsSumEt = sum([x.pt() for x in chargedchs])
        event.tkPVLooseSumEt = sum([x.pt() for x in chargedPVLoose])
        event.tkPVTightSumEt = sum([x.pt() for x in chargedPVTight])


    def makeGenTkMet(self, event):
        genCharged = [ x for x in self.mchandles['packedGen'].product() if x.charge() != 0 and abs(x.eta()) < 2.4 ]
        event.tkGenMet = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in genCharged])) , -1.*(sum([x.py() for x in genCharged])), 0, math.hypot((sum([x.px() for x in genCharged])),(sum([x.py() for x in genCharged]))) )

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
        px,py = self.metNoMu.px()+mupx, self.metNoMu.py()+mupy
        self.metNoMu.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        px,py = self.metNoMuNoPU.px()+mupx, self.metNoMuNoPU.py()+mupy
        self.metNoMuNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        setattr(event, "metNoMu"+self.cfg_ana.collectionPostFix, self.metNoMu)
        if self.cfg_ana.doMetNoPU: setattr(event, "metNoMuNoPU"+self.cfg_ana.collectionPostFix, self.metNoMuNoPU)


    def makeMETNoEle(self, event):
        self.metNoEle = copy.deepcopy(self.met)
        if self.cfg_ana.doMetNoPU: self.metNoEleNoPU = copy.deepcopy(self.metNoPU)

        elepx = 0
        elepy = 0
        #sum electron momentum                                                                                                                                                                                                                            
        for ele in event.selectedElectrons:
            elepx += ele.px()
            elepy += ele.py()

        #subtract electron momentum and construct met                                                                                                                                                                                                     
        px,py = self.metNoEle.px()+elepx, self.metNoEle.py()+elepy
        self.metNoEle.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))

        px,py = self.metNoEleNoPU.px()+elepx, self.metNoEleNoPU.py()+elepy
        self.metNoEleNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        setattr(event, "metNoEle"+self.cfg_ana.collectionPostFix, self.metNoEle)
        if self.cfg_ana.doMetNoPU: setattr(event, "metNoEleNoPU"+self.cfg_ana.collectionPostFix, self.metNoEleNoPU)

    def makeMETNoPhoton(self, event):
        self.metNoPhoton = copy.deepcopy(self.met)

        phopx = 0
        phopy = 0
        #sum photon momentum                                                                                                                                                                                                                            
        for pho in event.selectedPhotons:
            phopx += pho.px()
            phopy += pho.py()

        #subtract photon momentum and construct met
        px,py = self.metNoPhoton.px()+phopx, self.metNoPhoton.py()+phopy
        self.metNoPhoton.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
        setattr(event, "metNoPhoton"+self.cfg_ana.collectionPostFix, self.metNoPhoton)
        if self.cfg_ana.doMetNoPU: 
          self.metNoPhotonNoPU = copy.deepcopy(self.metNoPU)
          px,py = self.metNoPhotonNoPU.px()+phopx, self.metNoPhotonNoPU.py()+phopy
          self.metNoPhotonNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
          setattr(event, "metNoPhotonNoPU"+self.cfg_ana.collectionPostFix, self.metNoPhotonNoPU)


    def makeMETs(self, event):
        import ROOT
        if self.cfg_ana.copyMETsByValue:
          self.met = ROOT.pat.MET(self.handles['met'].product()[0])
          if self.cfg_ana.doMetNoPU: self.metNoPU = ROOT.pat.MET(self.handles['nopumet'].product()[0])
        else:
          self.met = self.handles['met'].product()[0]
          if self.cfg_ana.doMetNoPU: self.metNoPU = self.handles['nopumet'].product()[0]

        #Shifted METs
        #Uncertainties defined in https://github.com/cms-sw/cmssw/blob/CMSSW_7_2_X/DataFormats/PatCandidates/interface/MET.h#L168
        #event.met_shifted = []
        if not self.cfg_ana.copyMETsByValue:
          for i in range(self.met.METUncertaintySize):
              m = ROOT.pat.MET(self.met)
              px  = m.shiftedPx(i);
              py  = m.shiftedPy(i);
              m.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, math.hypot(px,py)))
              #event.met_shifted += [m]
              setattr(event, "met{0}_shifted_{1}".format(self.cfg_ana.collectionPostFix, i), m)

        self.met_sig = self.met.significance()
        self.met_sumet = self.met.sumEt()


        ###https://github.com/cms-sw/cmssw/blob/CMSSW_7_2_X/DataFormats/PatCandidates/interface/MET.h
        if not self.cfg_ana.copyMETsByValue:
          self.metraw = self.met.shiftedPt(12, 0)
          self.metType1chs = self.met.shiftedPt(12, 1)
          setattr(event, "metraw"+self.cfg_ana.collectionPostFix, self.metraw)
          setattr(event, "metType1chs"+self.cfg_ana.collectionPostFix, self.metType1chs)

        if self.cfg_ana.recalibrate and hasattr(event, 'deltaMetFromJetSmearing'+self.cfg_ana.jetAnalyzerCalibrationPostFix):
          deltaMetSmear = getattr(event, 'deltaMetFromJetSmearing'+self.cfg_ana.jetAnalyzerCalibrationPostFix)
          self.applyDeltaMet(self.met, deltaMetSmear)
          if self.cfg_ana.doMetNoPU:
            self.applyDeltaMet(self.metNoPU, deltaMetSmear) 
        if self.cfg_ana.recalibrate and hasattr(event, 'deltaMetFromJEC'+self.cfg_ana.jetAnalyzerCalibrationPostFix):
          deltaMetJEC = getattr(event, 'deltaMetFromJEC'+self.cfg_ana.jetAnalyzerCalibrationPostFix)
#          print 'before JEC', self.cfg_ana.collectionPostFix, self.met.px(),self.met.py(), 'deltaMetFromJEC'+self.cfg_ana.jetAnalyzerCalibrationPostFix, deltaMetJEC
          self.applyDeltaMet(self.met, deltaMetJEC)
          if self.cfg_ana.doMetNoPU: 
            self.applyDeltaMet(self.metNoPU, deltaMetJEC)
#          print 'after JEC', self.cfg_ana.collectionPostFix, self.met.px(),self.met.py(), 'deltaMetFromJEC'+self.cfg_ana.jetAnalyzerCalibrationPostFix, deltaMetJEC

        setattr(event, "met"+self.cfg_ana.collectionPostFix, self.met)
        if self.cfg_ana.doMetNoPU: setattr(event, "metNoPU"+self.cfg_ana.collectionPostFix, self.metNoPU)
        setattr(event, "met_sig"+self.cfg_ana.collectionPostFix, self.met_sig)
        setattr(event, "met_sumet"+self.cfg_ana.collectionPostFix, self.met_sumet)
        
        if self.cfg_ana.doMetNoMu and hasattr(event, 'selectedMuons'):
            self.makeMETNoMu(event)

        if self.cfg_ana.doMetNoEle and hasattr(event, 'selectedElectrons'):
            self.makeMETNoEle(event)

        if self.cfg_ana.doMetNoPhoton and hasattr(event, 'selectedPhotons'):
            self.makeMETNoPhoton(event)

    def process(self, event):
        self.readCollections( event.input)
        self.counters.counter('events').inc('all events')

        self.makeMETs(event)

        if self.cfg_ana.doTkMet: 
            self.makeTkMETs(event);

            if self.cfg_comp.isMC and hasattr(event, 'genParticles'):
                self.makeGenTkMet(event)

        return True


setattr(METAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = METAnalyzer,
    metCollection     = "slimmedMETs",
    noPUMetCollection = "slimmedMETs",
    copyMETsByValue = False,
    recalibrate = True,
    jetAnalyzerCalibrationPostFix = "",
    doTkMet = False,
    doMetNoPU = True,  
    doMetNoMu = False,  
    doMetNoEle = False,  
    doMetNoPhoton = False,  
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    collectionPostFix = "",
    )
)
