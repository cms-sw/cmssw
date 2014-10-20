from PhysicsTools.Heppy.analyzer.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzer.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Electron import Electron
from PhysicsTools.Heppy.physicsobjects.Muon import Muon
from CMGTools.TTHAnalysis.tools.EfficiencyCorrector import EfficiencyCorrector

from PhysicsTools.HeppyCore.utils.deltar import bestMatch
from PhysicsTools.Heppy.physicsutils.RochesterCorrections import rochcor
from PhysicsTools.Heppy.physicsutils.MuScleFitCorrector   import MuScleFitCorr
from PhysicsTools.Heppy.physicsutils.ElectronCalibrator import EmbeddedElectronCalibrator
from CMGTools.TTHAnalysis.electronCalibrator import ElectronCalibrator

from ROOT import CMGMuonCleanerBySegmentsAlgo
cmgMuonCleanerBySegments = CMGMuonCleanerBySegmentsAlgo()

class LeptonAnalyzer( Analyzer ):

    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(LeptonAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        if self.cfg_ana.doMuScleFitCorrections and self.cfg_ana.doMuScleFitCorrections != "none":
            if self.cfg_ana.doMuScleFitCorrections not in [ "none", "prompt", "prompt-sync", "rereco", "rereco-sync" ]:
                raise RuntimeError, 'doMuScleFitCorrections must be one of "none", "prompt", "prompt-sync", "rereco", "rereco-sync"'
            rereco = ("prompt" not in self.cfg_ana.doMuScleFitCorrections)
            sync   = ("sync"       in self.cfg_ana.doMuScleFitCorrections)
            self.muscleCorr = MuScleFitCorr(cfg_comp.isMC, rereco, sync)
            if hasattr(self.cfg_ana, "doRochesterCorrections") and self.cfg_ana.doRochesterCorrections:
                raise RuntimeError, "You can't run both Rochester and MuScleFit corrections!"
        else:
            self.cfg_ana.doMuScleFitCorrections = False
        if self.cfg_ana.doElectronScaleCorrections == "embedded":
            self.electronEnergyCalibrator = EmbeddedElectronCalibrator()
        else:
            self.electronEnergyCalibrator = ElectronCalibrator(cfg_comp.isMC)
        if hasattr(cfg_comp,'efficiency'):
            self.efficiency= EfficiencyCorrector(cfg_comp.efficiency)
    #----------------------------------------
    # DECLARATION OF HANDLES OF LEPTONS STUFF   
    #----------------------------------------
        

    def declareHandles(self):
        super(LeptonAnalyzer, self).declareHandles()

        #leptons
        self.handles['muons'] = AutoHandle(self.cfg_ana.muons,"std::vector<pat::Muon>")            
        self.handles['electrons'] = AutoHandle(self.cfg_ana.electrons,"std::vector<pat::Electron>")            
    
        #rho for muons
        self.handles['rhoMu'] = AutoHandle( self.cfg_ana.rhoMuon, 'double')
        #rho for electrons
        self.handles['rhoEle'] = AutoHandle( self.cfg_ana.rhoElectron, 'double')

    def beginLoop(self):
        super(LeptonAnalyzer,self).beginLoop()
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    #------------------
    # MAKE LEPTON LISTS
    #------------------

    
    def makeLeptons(self, event):
        ### inclusive leptons = all leptons that could be considered somewhere in the analysis, with minimal requirements (used e.g. to match to MC)
        event.inclusiveLeptons = []
        ### selected leptons = subset of inclusive leptons passing some basic id definition and pt requirement
        ### other    leptons = subset of inclusive leptons failing some basic id definition and pt requirement
        event.selectedLeptons = []
        event.selectedMuons = []
        event.selectedElectrons = []
        event.otherLeptons = []

        #muons
        allmuons = self.makeAllMuons(event)

        for mu in allmuons:
            # inclusive, very loose, selection
            if (mu.track().isNonnull() and mu.muonID(self.cfg_ana.inclusive_muon_id) and 
                    mu.pt()>self.cfg_ana.inclusive_muon_pt and abs(mu.eta())<self.cfg_ana.inclusive_muon_eta and 
                    abs(mu.dxy())<self.cfg_ana.inclusive_muon_dxy and abs(mu.dz())<self.cfg_ana.inclusive_muon_dz):
                event.inclusiveLeptons.append(mu)
                # basic selection
                if (mu.muonID(self.cfg_ana.loose_muon_id) and 
                        mu.pt() > self.cfg_ana.loose_muon_pt and abs(mu.eta()) < self.cfg_ana.loose_muon_eta and 
                        abs(mu.dxy()) < self.cfg_ana.loose_muon_dxy and abs(mu.dz()) < self.cfg_ana.loose_muon_dz and
                        mu.relIso03 < self.cfg_ana.loose_muon_relIso and 
                        mu.absIso03 < (self.cfg_ana.loose_muon_absIso if hasattr(self.cfg_ana,'loose_muon_absIso') else 9e99)):
                    mu.looseIdSusy = True
                    event.selectedLeptons.append(mu)
                    event.selectedMuons.append(mu)
                else:
                    mu.looseIdSusy = False
                    event.otherLeptons.append(mu)

        #electrons        
        allelectrons = self.makeAllElectrons(event)

        looseMuons = event.selectedLeptons[:]
        for ele in allelectrons:
            ## remove muons if muForEleCrossCleaning is not empty
            ## apply selection
            if ( ele.electronID(self.cfg_ana.inclusive_electron_id) and
                    ele.pt()>self.cfg_ana.inclusive_electron_pt and abs(ele.eta())<self.cfg_ana.inclusive_electron_eta and 
                    abs(ele.dxy())<self.cfg_ana.inclusive_electron_dxy and abs(ele.dz())<self.cfg_ana.inclusive_electron_dz and 
                    ele.gsfTrack().trackerExpectedHitsInner().numberOfLostHits()<=self.cfg_ana.inclusive_electron_lostHits ):
                event.inclusiveLeptons.append(ele)
                # basic selection
                if (ele.electronID(self.cfg_ana.loose_electron_id) and
                         ele.pt()>self.cfg_ana.loose_electron_pt and abs(ele.eta())<self.cfg_ana.loose_electron_eta and 
                         abs(ele.dxy()) < self.cfg_ana.loose_electron_dxy and abs(ele.dz())<self.cfg_ana.loose_electron_dz and 
                         ele.relIso03 <= self.cfg_ana.loose_electron_relIso and
                         ele.absIso03 < (self.cfg_ana.loose_electron_absIso if hasattr(self.cfg_ana,'loose_electron_absIso') else 9e99) and
                         ele.gsfTrack().trackerExpectedHitsInner().numberOfLostHits() <= self.cfg_ana.loose_electron_lostHits and
                         ( True if (hasattr(self.cfg_ana,'notCleaningElectrons') and self.cfg_ana.notCleaningElectrons) else (bestMatch(ele, looseMuons)[1] > self.cfg_ana.min_dr_electron_muon) )):
                    event.selectedLeptons.append(ele)
                    event.selectedElectrons.append(ele)
                    ele.looseIdSusy = True
                else:
                    event.otherLeptons.append(ele)
                    ele.looseIdSusy = False

        event.otherLeptons.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedLeptons.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedMuons.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedElectrons.sort(key = lambda l : l.pt(), reverse = True)
        event.inclusiveLeptons.sort(key = lambda l : l.pt(), reverse = True)

        for lepton in event.selectedLeptons:
            if hasattr(self,'efficiency'):
                self.efficiency.attachToObject(lepton)

    def makeAllMuons(self, event):
        """
               make a list of all muons, and apply basic corrections to them
        """
        # Start from all muons
        allmuons = map( Muon, self.handles['muons'].product() )

        # Muon scale and resolution corrections (if enabled)
        if self.cfg_ana.doMuScleFitCorrections:
            for mu in allmuons:
                self.muscleCorr.correct(mu, event.run)
        elif self.cfg_ana.doRochesterCorrections:
            for mu in allmuons:
                corp4 = rochcor.corrected_p4(mu, event.run) 
                mu.setP4( corp4 )

        # Clean up dulicate muons (note: has no effect unless the muon id is removed)
        if self.cfg_ana.doSegmentBasedMuonCleaning:
            isgood = cmgMuonCleanerBySegments.clean( self.handles['muons'].product() )
            newmu = []
            for i,mu in enumerate(allmuons):
                if isgood[i]: newmu.append(mu)
            allmuons = newmu

        # Attach the vertex to them, for dxy/dz calculation
        for mu in allmuons:
            mu.associatedVertex = event.goodVertices[0]

        # Compute relIso in 0.3 and 0.4 cones
        for mu in allmuons:
            mu.absIso03 = (mu.pfIsolationR03().sumChargedHadronPt + max( mu.pfIsolationR03().sumNeutralHadronEt +  mu.pfIsolationR03().sumPhotonEt -  mu.pfIsolationR03().sumPUPt/2,0.0))
            mu.absIso04 = (mu.pfIsolationR04().sumChargedHadronPt + max( mu.pfIsolationR04().sumNeutralHadronEt +  mu.pfIsolationR04().sumPhotonEt -  mu.pfIsolationR04().sumPUPt/2,0.0))
            mu.relIso03 = mu.absIso03/mu.pt()
            mu.relIso04 = mu.absIso04/mu.pt()
 
        return allmuons

    def makeAllElectrons(self, event):
        """
               make a list of all electrons, and apply basic corrections to them
        """
        allelectrons = map( Electron, self.handles['electrons'].product() )

        ## Duplicate removal for fast sim (to be checked if still necessary in latest greatest 5.3.X releases)
        allelenodup = []
        for e in allelectrons:
            dup = False
            for e2 in allelenodup:
                if abs(e.pt()-e2.pt()) < 1e-6 and abs(e.eta()-e2.eta()) < 1e-6 and abs(e.phi()-e2.phi()) < 1e-6 and e.charge() == e2.charge():
                    dup = True
                    break
            if not dup: allelenodup.append(e)
        allelectrons = allelenodup

        # fill EA for rho-corrected isolation
        for ele in allelectrons:
          ele.rho = float(self.handles['rhoEle'].product()[0])
          SCEta = abs(ele.superCluster().eta())
          if (abs(SCEta) >= 0.0   and abs(SCEta) < 1.0   ) : ele.EffectiveArea = 0.13 # 0.130;
          if (abs(SCEta) >= 1.0   and abs(SCEta) < 1.479 ) : ele.EffectiveArea = 0.14 # 0.137;
          if (abs(SCEta) >= 1.479 and abs(SCEta) < 2.0   ) : ele.EffectiveArea = 0.07 # 0.067;
          if (abs(SCEta) >= 2.0   and abs(SCEta) < 2.2   ) : ele.EffectiveArea = 0.09 # 0.089;
          if (abs(SCEta) >= 2.2   and abs(SCEta) < 2.3   ) : ele.EffectiveArea = 0.11 # 0.107;
          if (abs(SCEta) >= 2.3   and abs(SCEta) < 2.4   ) : ele.EffectiveArea = 0.11 # 0.110;
          if (abs(SCEta) >= 2.4)                           : ele.EffectiveArea = 0.14 # 0.138;

        # Electron scale calibrations
        if self.cfg_ana.doElectronScaleCorrections:
            for ele in allelectrons:
                self.electronEnergyCalibrator.correct(ele, event.run)

        # Attach the vertex
        for ele in allelectrons:
            ele.associatedVertex = event.goodVertices[0]

        # Compute relIso with R=0.3 and R=0.4 cones
        for ele in allelectrons:
            if self.cfg_ana.ele_isoCorr=="rhoArea" :
                 ele.absIso03 = (ele.chargedHadronIso(0.3) + max(ele.neutralHadronIso(0.3)+ele.photonIso(0.3)-ele.rho*ele.EffectiveArea,0))
                 ele.absIso04 = (ele.chargedHadronIso(0.4) + max(ele.neutralHadronIso(0.4)+ele.photonIso(0.4)-ele.rho*ele.EffectiveArea,0))
            elif self.cfg_ana.ele_isoCorr=="deltaBeta" :
                 ele.absIso03 = (ele.pfIsolationVariables().sumChargedHadronPt + max( ele.pfIsolationVariables().sumNeutralHadronEt + ele.pfIsolationVariables().sumPhotonEt - ele.pfIsolationVariables().sumPUPt/2,0.0))
                 ele.absIso04 = 0.
            else :
                 raise RuntimeError, "Unsupported ele_isoCorr name '" + str(self.cfg_ana.ele_isoCorr) +  "'! For now only 'rhoArea' and 'deltaBeta' are supported."
            ele.relIso03 = ele.absIso03/ele.pt()
            ele.relIso04 = ele.absIso04/ele.pt()

        # Set tight MVA id
        for ele in allelectrons:
            if self.cfg_ana.ele_tightId=="MVA" :
                 ele.tightIdResult = ele.electronID("POG_MVA_ID_Trig_full5x5")
            elif self.cfg_ana.ele_tightId=="Cuts_2012" :
                 ele.tightIdResult = -1 + 1*ele.electronID("POG_Cuts_ID_2012_Veto") + 1*ele.electronID("POG_Cuts_ID_2012_Loose") + 1*ele.electronID("POG_Cuts_ID_2012_Medium") + 1*ele.electronID("POG_Cuts_ID_2012_Tight")
            else :
                 raise RuntimeError, "Unsupported ele_tightId name '" + str(self.cfg_ana.ele_tightId) +  "'! For now only 'MVA' and 'Cuts_2012' are supported."

        
        return allelectrons 

    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        #call the leptons functions
        self.makeLeptons(event)

        return True
