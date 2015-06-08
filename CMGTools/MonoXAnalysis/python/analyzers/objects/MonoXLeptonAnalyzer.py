from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Electron import Electron
from PhysicsTools.Heppy.physicsobjects.Muon import Muon
#from CMGTools.TTHAnalysis.tools.EfficiencyCorrector import EfficiencyCorrector

from PhysicsTools.HeppyCore.utils.deltar import bestMatch
from PhysicsTools.Heppy.physicsutils.RochesterCorrections import rochcor
from PhysicsTools.Heppy.physicsutils.MuScleFitCorrector   import MuScleFitCorr
from PhysicsTools.Heppy.physicsutils.ElectronCalibrator import EmbeddedElectronCalibrator
#from CMGTools.TTHAnalysis.electronCalibrator import ElectronCalibrator
import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.utils.deltar import * 
from PhysicsTools.Heppy.physicsutils.genutils import *


from ROOT import heppy
cmgMuonCleanerBySegments = heppy.CMGMuonCleanerBySegmentsAlgo()

class MonoXLeptonAnalyzer( Analyzer ):

    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(MonoXLeptonAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        # Isolation cut
        if hasattr(cfg_ana, 'veto_electron_isoCut'):
            self.eleIsoCut = cfg_ana.veto_electron_isoCut
        else:
            self.eleIsoCut = lambda ele : (
                    ele.relIso03 <= (self.cfg_ana.veto_electron_relIso[0] if ele.isEB() else self.cfg_ana.veto_electron_relIso[1]) and 
                    ele.absIso03 <  getattr(self.cfg_ana,'veto_electron_absIso',9e99))
        if hasattr(cfg_ana, 'veto_muon_isoCut'):
            self.muIsoCut = cfg_ana.veto_muon_isoCut
        else:
            self.muIsoCut = lambda mu : (
                    mu.relIso03 <= self.cfg_ana.veto_muon_relIso and 
                    mu.absIso03 <  getattr(self.cfg_ana,'veto_muon_absIso',9e99))


    #----------------------------------------
    # DECLARATION OF HANDLES OF LEPTONS STUFF   
    #----------------------------------------
        

    def beginLoop(self, setup):
        super(MonoXLeptonAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    #------------------
    # MAKE LEPTON LISTS
    #------------------

    
    def makeVetoLeptons(self, event):
        ### selected leptons = subset of inclusive leptons passing some basic id definition and pt requirement
        ### other    leptons = subset of inclusive leptons failing some basic id definition and pt requirement
        event.vetoLeptons = []
        event.vetoMuons = []
        event.vetoElectrons = []

        #the selected leptons are the ones with the corrections / calibrations applied
        # make veto leptons 
        for mu in event.selectedMuons:
            if (mu.pt() > self.cfg_ana.veto_muon_pt and abs(mu.eta()) < self.cfg_ana.veto_muon_eta and 
                abs(mu.dxy()) < self.cfg_ana.veto_muon_dxy and abs(mu.dz()) < self.cfg_ana.veto_muon_dz and
                self.muIsoCut(mu)):
                    mu.vetoIdMonoX = True
                    event.vetoLeptons.append(mu)
                    event.vetoMuons.append(mu)
            else:
                    mu.vetoIdMonoX = False
        vetoMuons = event.vetoLeptons[:]
        for ele in event.selectedElectrons:
            if ( ele.pt() > self.cfg_ana.veto_electron_pt and abs(ele.eta())<self.cfg_ana.veto_electron_eta and 
                 (abs(ele.dxy()) < (self.cfg_ana.veto_electron_dxy[0] if ele.isEB() else self.cfg_ana.veto_electron_dxy[1]) and 
                  abs(ele.dz()) < (self.cfg_ana.veto_electron_dz[0]  if ele.isEB() else self.cfg_ana.veto_electron_dxy[1])) and 
                 self.eleIsoCut(ele) and 
                 ele.passConversionVeto() and ele.lostInner() <= (self.cfg_ana.veto_electron_lostHits[0] if ele.isEB() else self.cfg_ana.veto_electron_lostHits[1]) and
                 ( True if getattr(self.cfg_ana,'notCleaningElectrons',False) else (bestMatch(ele, vetoMuons)[1] > (self.cfg_ana.min_dr_electron_muon**2)) )):
                    event.vetoLeptons.append(ele)
                    event.vetoElectrons.append(ele)
                    ele.vetoIdMonoX = True
            else:
                    ele.vetoIdMonoX = False

        event.vetoLeptons.sort(key = lambda l : l.pt(), reverse = True)
        event.vetoMuons.sort(key = lambda l : l.pt(), reverse = True)
        event.vetoElectrons.sort(key = lambda l : l.pt(), reverse = True)

        for lepton in event.vetoLeptons:
            if hasattr(self,'efficiency'):
                self.efficiency.attachToObject(lepton)


    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        #call the leptons functions
        self.makeVetoLeptons(event)

        return True

#A default config
setattr(MonoXLeptonAnalyzer,"defaultConfig",cfg.Analyzer(
    verbose=False,
    class_object=MonoXLeptonAnalyzer,
    # veto muon selection
    veto_muon_pt     = 10,
    veto_muon_eta    = 2.4,
    veto_muon_dxy    = 0.2,
    veto_muon_dz     = 0.5,
    veto_muon_relIso = 0.4,
    # veto electron selection
    veto_electron_pt     = 10,
    veto_electron_eta    = 2.5,
    veto_electron_dxy    = [0.0250, 0.2232],
    veto_electron_dz     = [0.5863, 0.9513],
    veto_electron_relIso = [0.3313, 0.3816],
    veto_electron_lostHits = [2.0, 3.0],
    # minimum deltaR between a loose electron and a loose muon (on overlaps, discard the electron)
    min_dr_electron_muon = 0.02,
    )
)
