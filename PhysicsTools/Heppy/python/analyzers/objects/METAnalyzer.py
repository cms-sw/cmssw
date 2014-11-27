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
        
class METAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(METAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(METAnalyzer, self).declareHandles()
        self.handles['met'] = AutoHandle( 'slimmedMETs', 'std::vector<pat::MET>' )
        self.handles['nopumet'] = AutoHandle( 'slimmedMETs', 'std::vector<pat::MET>' )
        self.handles['cmgCand1'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand2'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand3'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand4'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand5'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand6'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand7'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['cmgCand8'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
        self.handles['vertices'] =  AutoHandle( "offlineSlimmedPrimaryVertices", 'std::vector<reco::Vertex>', fallbackLabel="offlinePrimaryVertices" )

    def beginLoop(self):
        super(METAnalyzer,self).beginLoop()
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    def makeTkMETs(self, event):
        event.tkMet = 0  

        charged = []
        pfcands = self.handles['cmgCand'].product()

        for i in xrange(pfcands.size()):

## ===> require the Track Candidate charge and with a  minimum dz 
            
            if (pfcands.at(i).charge()!=0 and (abs(pfcands.at(i).dz())<=self.cfg_ana.dzMax )):
                
                charged.append(pfcands.at(i))
                
        import ROOT
        event.tkMet = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in charged])) , -1.*(sum([x.py() for x in charged])), 0, 0 )
##        print 'tkmet',event.tkMet.pt(),'tkmetphi',event.tkMet.phi()


    def makeMETs(self, event):
        event.met = self.handles['met'].product()[0]
        event.metNoPU = self.handles['nopumet'].product()[0]
        if hasattr(event, 'deltaMetFromJetSmearing'):
            import ROOT
            px,py = event.met.px()+event.deltaMetFromJetSmearing[0], event.met.py()+event.deltaMetFromJetSmearing[1]
            event.met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
            px,py = event.metNoPU.px()+event.deltaMetFromJetSmearing[0], event.metNoPU.py()+event.deltaMetFromJetSmearing[1]
            event.metNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
        if hasattr(event, 'deltaMetFromJEC') and event.deltaMetFromJEC[0] != 0 and event.deltaMetFromJEC[1] != 0:
            import ROOT
            px,py = event.met.px()+event.deltaMetFromJEC[0], event.met.py()+event.deltaMetFromJEC[1]
            event.met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
            px,py = event.metNoPU.px()+event.deltaMetFromJEC[0], event.metNoPU.py()+event.deltaMetFromJEC[1]
            event.metNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))


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
    doTkMet = False,
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    )
)
