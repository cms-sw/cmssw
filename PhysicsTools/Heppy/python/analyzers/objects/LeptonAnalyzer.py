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
	#FIXME: only Embedded works
        self.electronEnergyCalibrator = EmbeddedElectronCalibrator()
#        if hasattr(cfg_comp,'efficiency'):
#            self.efficiency= EfficiencyCorrector(cfg_comp.efficiency)
        # Isolation cut
        if hasattr(cfg_ana, 'loose_electron_isoCut'):
            self.eleIsoCut = cfg_ana.loose_electron_isoCut
        else:
            self.eleIsoCut = lambda ele : (
                    ele.relIso03 <= self.cfg_ana.loose_electron_relIso and 
                    ele.absIso03 <  getattr(self.cfg_ana,'loose_electron_absIso',9e99))
        if hasattr(cfg_ana, 'loose_muon_isoCut'):
            self.muIsoCut = cfg_ana.loose_muon_isoCut
        else:
            self.muIsoCut = lambda mu : (
                    mu.relIso03 <= self.cfg_ana.loose_muon_relIso and 
                    mu.absIso03 <  getattr(self.cfg_ana,'loose_muon_absIso',9e99))



        self.eleEffectiveArea = getattr(cfg_ana, 'ele_effectiveAreas', "Phys14_25ns_v1")
        self.muEffectiveArea  = getattr(cfg_ana, 'mu_effectiveAreas',  "Phys14_25ns_v1")
        # MiniIsolation
        self.doMiniIsolation = getattr(cfg_ana, 'doMiniIsolation', False)
        if self.doMiniIsolation:
            self.miniIsolationPUCorr = self.cfg_ana.miniIsolationPUCorr
            self.miniIsolationVetoLeptons = self.cfg_ana.miniIsolationVetoLeptons
            if self.miniIsolationVetoLeptons not in [ None, 'any', 'inclusive' ]:
                raise RuntimeError, "miniIsolationVetoLeptons should be None, or 'any' (all reco leptons), or 'inclusive' (all inclusive leptons)"
            if self.miniIsolationPUCorr == "weights":
                self.IsolationComputer = heppy.IsolationComputer(0.4)
            else:
                self.IsolationComputer = heppy.IsolationComputer()

        self.doIsoAnnulus = getattr(cfg_ana, 'doIsoAnnulus', False)
        if self.doIsoAnnulus:
            if not self.doMiniIsolation:
                self.IsolationComputer = heppy.IsolationComputer()
            
        self.doIsolationScan = getattr(cfg_ana, 'doIsolationScan', False)
        if self.doIsolationScan:
            if self.doMiniIsolation:
                assert (self.miniIsolationPUCorr!="weights")
                assert (self.miniIsolationVetoLeptons==None)
            else:
                self.IsolationComputer = heppy.IsolationComputer()
            

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

        if self.doMiniIsolation or self.doIsolationScan:
            self.handles['packedCandidates'] = AutoHandle( self.cfg_ana.packedCandidates, 'std::vector<pat::PackedCandidate>')
    def beginLoop(self, setup):
        super(LeptonAnalyzer,self).beginLoop(setup)
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

        if self.doMiniIsolation or self.doIsolationScan:
            self.IsolationComputer.setPackedCandidates(self.handles['packedCandidates'].product())
        if self.doMiniIsolation:
            if self.miniIsolationVetoLeptons == "any":
                for lep in self.handles['muons'].product(): 
                    self.IsolationComputer.addVeto(lep)
                for lep in self.handles['electrons'].product(): 
                    self.IsolationComputer.addVeto(lep)

        #muons
        allmuons = self.makeAllMuons(event)

        #electrons        
        allelectrons = self.makeAllElectrons(event)

        #make inclusive leptons
        inclusiveMuons = []
        inclusiveElectrons = []
        for mu in allmuons:
            if (mu.track().isNonnull() and mu.muonID(self.cfg_ana.inclusive_muon_id) and 
                    mu.pt()>self.cfg_ana.inclusive_muon_pt and abs(mu.eta())<self.cfg_ana.inclusive_muon_eta and 
                    abs(mu.dxy())<self.cfg_ana.inclusive_muon_dxy and abs(mu.dz())<self.cfg_ana.inclusive_muon_dz):
                inclusiveMuons.append(mu)
        for ele in allelectrons:
            if ( ele.electronID(self.cfg_ana.inclusive_electron_id) and
                    ele.pt()>self.cfg_ana.inclusive_electron_pt and abs(ele.eta())<self.cfg_ana.inclusive_electron_eta and 
                    abs(ele.dxy())<self.cfg_ana.inclusive_electron_dxy and abs(ele.dz())<self.cfg_ana.inclusive_electron_dz and 
                    ele.lostInner()<=self.cfg_ana.inclusive_electron_lostHits ):
                inclusiveElectrons.append(ele)
        event.inclusiveLeptons = inclusiveMuons + inclusiveElectrons
 
        if self.doMiniIsolation:
            if self.miniIsolationVetoLeptons == "inclusive":
                for lep in event.inclusiveLeptons: 
                    self.IsolationComputer.addVeto(lep)
            for lep in event.inclusiveLeptons:
                self.attachMiniIsolation(lep)
        
        if self.doIsoAnnulus:
            for lep in event.inclusiveLeptons:
                self.attachIsoAnnulus04(lep)

        if self.doIsolationScan:
            for lep in event.inclusiveLeptons:
                self.attachIsolationScan(lep)

        # make loose leptons (basic selection)
        for mu in inclusiveMuons:
                if (mu.muonID(self.cfg_ana.loose_muon_id) and 
                        mu.pt() > self.cfg_ana.loose_muon_pt and abs(mu.eta()) < self.cfg_ana.loose_muon_eta and 
                        abs(mu.dxy()) < self.cfg_ana.loose_muon_dxy and abs(mu.dz()) < self.cfg_ana.loose_muon_dz and
                        self.muIsoCut(mu)):
                    mu.looseIdSusy = True
                    event.selectedLeptons.append(mu)
                    event.selectedMuons.append(mu)
                else:
                    mu.looseIdSusy = False
                    event.otherLeptons.append(mu)
        looseMuons = event.selectedLeptons[:]
        for ele in inclusiveElectrons:
               if (ele.electronID(self.cfg_ana.loose_electron_id) and
                         ele.pt()>self.cfg_ana.loose_electron_pt and abs(ele.eta())<self.cfg_ana.loose_electron_eta and 
                         abs(ele.dxy()) < self.cfg_ana.loose_electron_dxy and abs(ele.dz())<self.cfg_ana.loose_electron_dz and 
                         self.eleIsoCut(ele) and 
                         ele.lostInner() <= self.cfg_ana.loose_electron_lostHits and
                         ( True if getattr(self.cfg_ana,'notCleaningElectrons',False) else (bestMatch(ele, looseMuons)[1] > (self.cfg_ana.min_dr_electron_muon**2)) )):
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

        # Attach EAs for isolation:
        for mu in allmuons:
          mu.rho = float(self.handles['rhoMu'].product()[0])
          if self.muEffectiveArea == "Data2012":
              if   aeta < 1.0  : mu.EffectiveArea03 = 0.382;
              elif aeta < 1.47 : mu.EffectiveArea03 = 0.317;
              elif aeta < 2.0  : mu.EffectiveArea03 = 0.242;
              elif aeta < 2.2  : mu.EffectiveArea03 = 0.326;
              elif aeta < 2.3  : mu.EffectiveArea03 = 0.462;
              else             : mu.EffectiveArea03 = 0.372;
              if   aeta < 1.0  : mu.EffectiveArea04 = 0.674;
              elif aeta < 1.47 : mu.EffectiveArea04 = 0.565;
              elif aeta < 2.0  : mu.EffectiveArea04 = 0.442;
              elif aeta < 2.2  : mu.EffectiveArea04 = 0.515;
              elif aeta < 2.3  : mu.EffectiveArea04 = 0.821;
              else             : mu.EffectiveArea04 = 0.660;
          elif self.muEffectiveArea == "Phys14_25ns_v1":
              aeta = abs(mu.eta())
              if   aeta < 0.800: mu.EffectiveArea03 = 0.0913
              elif aeta < 1.300: mu.EffectiveArea03 = 0.0765
              elif aeta < 2.000: mu.EffectiveArea03 = 0.0546
              elif aeta < 2.200: mu.EffectiveArea03 = 0.0728
              else:              mu.EffectiveArea03 = 0.1177
              if   aeta < 0.800: mu.EffectiveArea04 = 0.1564
              elif aeta < 1.300: mu.EffectiveArea04 = 0.1325
              elif aeta < 2.000: mu.EffectiveArea04 = 0.0913
              elif aeta < 2.200: mu.EffectiveArea04 = 0.1212
              else:              mu.EffectiveArea04 = 0.2085
          else: raise RuntimeError,  "Unsupported value for mu_effectiveAreas: can only use Data2012 (rho: ?) and Phys14_v1 (rho: fixedGridRhoFastjetAll)"
        # Attach the vertex to them, for dxy/dz calculation
        for mu in allmuons:
            mu.associatedVertex = event.goodVertices[0] if len(event.goodVertices)>0 else event.vertices[0]
            mu.setTrackForDxyDz(self.cfg_ana.muon_dxydz_track)

        # Set tight id if specified
        if hasattr(self.cfg_ana, "mu_tightId"):
            for mu in allmuons:
               mu.tightIdResult = mu.muonID(self.cfg_ana.mu_tightId)
 
        # Compute relIso in 0.3 and 0.4 cones
        for mu in allmuons:
            if self.cfg_ana.mu_isoCorr=="rhoArea" :
                mu.absIso03 = (mu.pfIsolationR03().sumChargedHadronPt + max( mu.pfIsolationR03().sumNeutralHadronEt +  mu.pfIsolationR03().sumPhotonEt - mu.rho * mu.EffectiveArea03,0.0))
                mu.absIso04 = (mu.pfIsolationR04().sumChargedHadronPt + max( mu.pfIsolationR04().sumNeutralHadronEt +  mu.pfIsolationR04().sumPhotonEt - mu.rho * mu.EffectiveArea04,0.0))
            elif self.cfg_ana.mu_isoCorr=="deltaBeta" :
                mu.absIso03 = (mu.pfIsolationR03().sumChargedHadronPt + max( mu.pfIsolationR03().sumNeutralHadronEt +  mu.pfIsolationR03().sumPhotonEt -  mu.pfIsolationR03().sumPUPt/2,0.0))
                mu.absIso04 = (mu.pfIsolationR04().sumChargedHadronPt + max( mu.pfIsolationR04().sumNeutralHadronEt +  mu.pfIsolationR04().sumPhotonEt -  mu.pfIsolationR04().sumPUPt/2,0.0))
            else :
                raise RuntimeError, "Unsupported mu_isoCorr name '" + str(self.cfg_ana.mu_isoCorr) +  "'! For now only 'rhoArea' and 'deltaBeta' are supported."
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
          if self.eleEffectiveArea == "Data2012":
              # https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaEARhoCorrection?rev=14
              SCEta = abs(ele.superCluster().eta())
              if   SCEta < 1.0  : ele.EffectiveArea03 = 0.13 # 0.130;
              elif SCEta < 1.479: ele.EffectiveArea03 = 0.14 # 0.137;
              elif SCEta < 2.0  : ele.EffectiveArea03 = 0.07 # 0.067;
              elif SCEta < 2.2  : ele.EffectiveArea03 = 0.09 # 0.089;
              elif SCEta < 2.3  : ele.EffectiveArea03 = 0.11 # 0.107;
              elif SCEta < 2.4  : ele.EffectiveArea03 = 0.11 # 0.110;
              else              : ele.EffectiveArea03 = 0.14 # 0.138;
              if   SCEta < 1.0  : ele.EffectiveArea04 = 0.208;
              elif SCEta < 1.479: ele.EffectiveArea04 = 0.209;
              elif SCEta < 2.0  : ele.EffectiveArea04 = 0.115;
              elif SCEta < 2.2  : ele.EffectiveArea04 = 0.143;
              elif SCEta < 2.3  : ele.EffectiveArea04 = 0.183;
              elif SCEta < 2.4  : ele.EffectiveArea04 = 0.194;
              else              : ele.EffectiveArea04 = 0.261;
          elif self.eleEffectiveArea == "Phys14_25ns_v1":
              aeta = abs(ele.eta())
              if   aeta < 0.800: ele.EffectiveArea03 = 0.1013
              elif aeta < 1.300: ele.EffectiveArea03 = 0.0988
              elif aeta < 2.000: ele.EffectiveArea03 = 0.0572
              elif aeta < 2.200: ele.EffectiveArea03 = 0.0842
              else:              ele.EffectiveArea03 = 0.1530
              if   aeta < 0.800: ele.EffectiveArea04 = 0.1830 
              elif aeta < 1.300: ele.EffectiveArea04 = 0.1734 
              elif aeta < 2.000: ele.EffectiveArea04 = 0.1077 
              elif aeta < 2.200: ele.EffectiveArea04 = 0.1565 
              else:              ele.EffectiveArea04 = 0.2680
          elif self.eleEffectiveArea == "Spring15_50ns_v1":
              aeta = abs(ele.eta())
              ## ----- https://github.com/ikrav/cmssw/blob/egm_id_747_v2/RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_50ns.txt
              if   aeta < 0.800: ele.EffectiveArea03 = 0.0973
              elif aeta < 1.300: ele.EffectiveArea03 = 0.0954
              elif aeta < 2.000: ele.EffectiveArea03 = 0.0632
              elif aeta < 2.200: ele.EffectiveArea03 = 0.0727
              else:              ele.EffectiveArea03 = 0.1337
              # warning: EAs not computed for cone DR=0.4 yet. Do not correct
              ele.EffectiveArea04 = 0.0
          elif self.eleEffectiveArea == "Spring15_25ns_v1":
              aeta = abs(ele.eta())
              ## ----- https://github.com/ikrav/cmssw/blob/egm_id_747_v2/RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt
              if   aeta < 1.000: ele.EffectiveArea03 = 0.1752
              elif aeta < 1.479: ele.EffectiveArea03 = 0.1862
              elif aeta < 2.000: ele.EffectiveArea03 = 0.1411
              elif aeta < 2.200: ele.EffectiveArea03 = 0.1534
              elif aeta < 2.300: ele.EffectiveArea03 = 0.1903
              elif aeta < 2.400: ele.EffectiveArea03 = 0.2243
              else:              ele.EffectiveArea03 = 0.2687
              # warning: EAs not computed for cone DR=0.4 yet. Do not correct
              ele.EffectiveArea04 = 0.0
          else: raise RuntimeError,  "Unsupported value for ele_effectiveAreas: can only use Data2012 (rho: ?) and Phys14_v1 (rho: fixedGridRhoFastjetAll)"

        # Electron scale calibrations
        if self.cfg_ana.doElectronScaleCorrections:
            for ele in allelectrons:
                self.electronEnergyCalibrator.correct(ele, event.run)

        # Attach the vertex
        for ele in allelectrons:
            ele.associatedVertex = event.goodVertices[0] if len(event.goodVertices)>0 else event.vertices[0]

        # Compute relIso with R=0.3 and R=0.4 cones
        for ele in allelectrons:
            if self.cfg_ana.ele_isoCorr=="rhoArea" :
                 ele.absIso03 = (ele.chargedHadronIsoR(0.3) + max(ele.neutralHadronIsoR(0.3)+ele.photonIsoR(0.3)-ele.rho*ele.EffectiveArea03,0))
                 ele.absIso04 = (ele.chargedHadronIsoR(0.4) + max(ele.neutralHadronIsoR(0.4)+ele.photonIsoR(0.4)-ele.rho*ele.EffectiveArea04,0))
            elif self.cfg_ana.ele_isoCorr=="deltaBeta" :
                 ele.absIso03 = (ele.chargedHadronIsoR(0.3) + max( ele.neutralHadronIsoR(0.3)+ele.photonIsoR(0.3) - ele.puChargedHadronIsoR(0.3)/2, 0.0))
                 ele.absIso04 = (ele.chargedHadronIsoR(0.4) + max( ele.neutralHadronIsoR(0.4)+ele.photonIsoR(0.4) - ele.puChargedHadronIsoR(0.4)/2, 0.0))
            else :
                 raise RuntimeError, "Unsupported ele_isoCorr name '" + str(self.cfg_ana.ele_isoCorr) +  "'! For now only 'rhoArea' and 'deltaBeta' are supported."
            ele.relIso03 = ele.absIso03/ele.pt()
            ele.relIso04 = ele.absIso04/ele.pt()

        # Set tight MVA id
        for ele in allelectrons:
            if self.cfg_ana.ele_tightId=="MVA" :
                 ele.tightIdResult = ele.electronID("POG_MVA_ID_Trig_full5x5")
            elif self.cfg_ana.ele_tightId=="Cuts_2012" :
                 ele.tightIdResult = -1 + 1*ele.electronID("POG_Cuts_ID_2012_Veto_full5x5") + 1*ele.electronID("POG_Cuts_ID_2012_Loose_full5x5") + 1*ele.electronID("POG_Cuts_ID_2012_Medium_full5x5") + 1*ele.electronID("POG_Cuts_ID_2012_Tight_full5x5")
            elif self.cfg_ana.ele_tightId=="Cuts_PHYS14_25ns_v1_ConvVetoDxyDz" :
                 ele.tightIdResult = -1 + 1*ele.electronID("POG_Cuts_ID_PHYS14_25ns_v1_ConvVetoDxyDz_Veto_full5x5") + 1*ele.electronID("POG_Cuts_ID_PHYS14_25ns_v1_ConvVetoDxyDz_Loose_full5x5") + 1*ele.electronID("POG_Cuts_ID_PHYS14_25ns_v1_ConvVetoDxyDz_Medium_full5x5") + 1*ele.electronID("POG_Cuts_ID_PHYS14_25ns_v1_ConvVetoDxyDz_Tight_full5x5")

            else :
                 try:
                     ele.tightIdResult = ele.electronID(self.cfg_ana.ele_tightId)
                 except RuntimeError:
                     raise RuntimeError, "Unsupported ele_tightId name '" + str(self.cfg_ana.ele_tightId) +  "'! For now only 'MVA' and 'Cuts_2012' are supported, in addition to what provided in Electron.py."

        
        return allelectrons 

    def attachMiniIsolation(self, mu):
        mu.miniIsoR = 10.0/min(max(mu.pt(), 50),200) 
        # -- version with increasing cone at low pT, gives slightly better performance for tight cuts and low pt leptons
        # mu.miniIsoR = 10.0/min(max(mu.pt(), 50),200) if mu.pt() > 20 else 4.0/min(max(mu.pt(),10),20) 
        what = "mu" if (abs(mu.pdgId()) == 13) else ("eleB" if mu.isEB() else "eleE")
        if what == "mu":
            mu.miniAbsIsoCharged = self.IsolationComputer.chargedAbsIso(mu.physObj, mu.miniIsoR, {"mu":0.0001,"eleB":0,"eleE":0.015}[what], 0.0);
        else:
            mu.miniAbsIsoCharged = self.IsolationComputer.chargedAbsIso(mu.physObj, mu.miniIsoR, {"mu":0.0001,"eleB":0,"eleE":0.015}[what], 0.0,self.IsolationComputer.selfVetoNone);

        if self.miniIsolationPUCorr == None: puCorr = self.cfg_ana.mu_isoCorr if what=="mu" else self.cfg_ana.ele_isoCorr
        else: puCorr = self.miniIsolationPUCorr

        if puCorr == "weights":
            if what == "mu":
                mu.miniAbsIsoNeutral = self.IsolationComputer.neutralAbsIsoWeighted(mu.physObj, mu.miniIsoR, 0.01, 0.5);
            else:
                mu.miniAbsIsoNeutral = ( self.IsolationComputer.photonAbsIsoWeighted(    mu.physObj, mu.miniIsoR, 0.08 if what == "eleE" else 0.0, 0.0, self.IsolationComputer.selfVetoNone) + 
                                         self.IsolationComputer.neutralHadAbsIsoWeighted(mu.physObj, mu.miniIsoR, 0.0, 0.0, self.IsolationComputer.selfVetoNone) )
        else:
            if what == "mu":
                mu.miniAbsIsoNeutral = self.IsolationComputer.neutralAbsIsoRaw(mu.physObj, mu.miniIsoR, 0.01, 0.5);
            else:
                mu.miniAbsIsoPho  = self.IsolationComputer.photonAbsIsoRaw(    mu.physObj, mu.miniIsoR, 0.08 if what == "eleE" else 0.0, 0.0, self.IsolationComputer.selfVetoNone) 
                mu.miniAbsIsoNHad = self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, mu.miniIsoR, 0.0, 0.0, self.IsolationComputer.selfVetoNone) 
                mu.miniAbsIsoNeutral = mu.miniAbsIsoPho + mu.miniAbsIsoNHad  
                # -- version relying on PF candidate vetos; apparently less performant, and the isolation computed at RECO level doesn't have them 
                #mu.miniAbsIsoPhoSV  = self.IsolationComputer.photonAbsIsoRaw(    mu.physObj, mu.miniIsoR, 0.0, 0.0) 
                #mu.miniAbsIsoNHadSV = self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, mu.miniIsoR, 0.0, 0.0) 
                #mu.miniAbsIsoNeutral = mu.miniAbsIsoPhoSV + mu.miniAbsIsoNHadSV  
            if puCorr == "rhoArea":
                mu.miniAbsIsoNeutral = max(0.0, mu.miniAbsIsoNeutral - mu.rho * mu.EffectiveArea03 * (mu.miniIsoR/0.3)**2)
            elif puCorr == "deltaBeta":
                if what == "mu":
                    mu.miniAbsIsoPU = self.IsolationComputer.puAbsIso(mu.physObj, mu.miniIsoR, 0.01, 0.5);
                else:
                    mu.miniAbsIsoPU = self.IsolationComputer.puAbsIso(mu.physObj, mu.miniIsoR, 0.015 if what == "eleE" else 0.0, 0.0,self.IsolationComputer.selfVetoNone);
                mu.miniAbsIsoNeutral = max(0.0, mu.miniAbsIsoNeutral - 0.5*mu.miniAbsIsoPU)
            elif puCorr != 'raw':
                raise RuntimeError, "Unsupported miniIsolationCorr name '" + puCorr +  "'! For now only 'rhoArea', 'deltaBeta', 'raw', 'weights' are supported (and 'weights' is not tested)."

        mu.miniAbsIso = mu.miniAbsIsoCharged + mu.miniAbsIsoNeutral
        mu.miniRelIso = mu.miniAbsIso/mu.pt()


    def attachIsoAnnulus04(self, mu):  # annulus isolation with outer cone of 0.4 and delta beta PU correction
        mu.miniIsoR = 10.0/min(max(mu.pt(), 50),200)
        mu.absIsoAnCharged = self.IsolationComputer.chargedAbsIso      (mu.physObj, 0.4, mu.miniIsoR, 0.0, self.IsolationComputer.selfVetoNone)
        mu.absIsoAnPho     = self.IsolationComputer.photonAbsIsoRaw    (mu.physObj, 0.4, mu.miniIsoR, 0.0, self.IsolationComputer.selfVetoNone) 
        mu.absIsoAnNHad    = self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, 0.4, mu.miniIsoR, 0.0, self.IsolationComputer.selfVetoNone) 
        mu.absIsoAnPU      = self.IsolationComputer.puAbsIso           (mu.physObj, 0.4, mu.miniIsoR, 0.0, self.IsolationComputer.selfVetoNone)
        mu.absIsoAnNeutral = max(0.0, mu.absIsoAnPho + mu.absIsoAnNHad - 0.5*mu.absIsoAnPU)

        mu.absIsoAn04 = mu.absIsoAnCharged + mu.absIsoAnNeutral
        mu.relIsoAn04 = mu.absIsoAn04/mu.pt()


    def attachIsolationScan(self, mu):

        what = "mu" if (abs(mu.pdgId()) == 13) else ("eleB" if mu.isEB() else "eleE")
        vetoreg = {"mu":0.0001,"eleB":0,"eleE":0.015}[what]

        if what=="mu":
            mu.ScanAbsIsoCharged005 = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.05, vetoreg, 0.0)
            mu.ScanAbsIsoCharged01  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.1, vetoreg, 0.0)
            mu.ScanAbsIsoCharged02  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.2, vetoreg, 0.0)
            mu.ScanAbsIsoCharged03  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.3, vetoreg, 0.0)
            mu.ScanAbsIsoCharged04  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.4, vetoreg, 0.0)
        else:
            mu.ScanAbsIsoCharged005 = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.05, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoCharged01  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.1, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoCharged02  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.2, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoCharged03  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.3, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoCharged04  = self.IsolationComputer.chargedAbsIso(mu.physObj, 0.4, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)

        if what=="mu":
            mu.ScanAbsIsoNeutral005 = self.IsolationComputer.neutralAbsIsoRaw(mu.physObj, 0.05, 0.01, 0.5)
            mu.ScanAbsIsoNeutral01  = self.IsolationComputer.neutralAbsIsoRaw(mu.physObj, 0.1,  0.01, 0.5)
            mu.ScanAbsIsoNeutral02  = self.IsolationComputer.neutralAbsIsoRaw(mu.physObj, 0.2,  0.01, 0.5)
            mu.ScanAbsIsoNeutral03  = self.IsolationComputer.neutralAbsIsoRaw(mu.physObj, 0.3,  0.01, 0.5)
            mu.ScanAbsIsoNeutral04  = self.IsolationComputer.neutralAbsIsoRaw(mu.physObj, 0.4,  0.01, 0.5)
        else:
            vetoreg = {"eleB":0.0,"eleE":0.08}[what]
            mu.ScanAbsIsoNeutral005 = self.IsolationComputer.photonAbsIsoRaw(mu.physObj, 0.05, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)+self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, 0.05, 0.0, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoNeutral01 = self.IsolationComputer.photonAbsIsoRaw(mu.physObj, 0.1, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)+self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, 0.1, 0.0, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoNeutral02 = self.IsolationComputer.photonAbsIsoRaw(mu.physObj, 0.2, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)+self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, 0.2, 0.0, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoNeutral03 = self.IsolationComputer.photonAbsIsoRaw(mu.physObj, 0.3, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)+self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, 0.3, 0.0, 0.0, self.IsolationComputer.selfVetoNone)
            mu.ScanAbsIsoNeutral04 = self.IsolationComputer.photonAbsIsoRaw(mu.physObj, 0.4, vetoreg, 0.0, self.IsolationComputer.selfVetoNone)+self.IsolationComputer.neutralHadAbsIsoRaw(mu.physObj, 0.4, 0.0, 0.0, self.IsolationComputer.selfVetoNone)


    def matchLeptons(self, event):
        def plausible(rec,gen):
            if abs(rec.pdgId()) == 11 and abs(gen.pdgId()) != 11:   return False
            if abs(rec.pdgId()) == 13 and abs(gen.pdgId()) != 13:   return False
            dr = deltaR(rec.eta(),rec.phi(),gen.eta(),gen.phi())
            if dr < 0.3: return True
            if rec.pt() < 10 and abs(rec.pdgId()) == 13 and gen.pdgId() != rec.pdgId(): return False
            if dr < 0.7: return True
            if min(rec.pt(),gen.pt())/max(rec.pt(),gen.pt()) < 0.3: return False
            return True

        leps = event.inclusiveLeptons if self.cfg_ana.match_inclusiveLeptons else event.selectedLeptons
        match = matchObjectCollection3(leps, 
                                       event.genleps + event.gentauleps, 
                                       deltaRMax = 1.2, filter = plausible)
        for lep in leps:
            gen = match[lep]
            lep.mcMatchId  = (gen.sourceId if gen != None else  0)
            lep.mcMatchTau = (gen in event.gentauleps if gen else -99)
            lep.mcLep=gen

    def isFromB(self,particle,bid=5, done={}):
        for i in xrange( particle.numberOfMothers() ): 
            mom  = particle.mother(i)
            momid = abs(mom.pdgId())
            if momid / 1000 == bid or momid / 100 == bid or momid == bid: 
                return True
            elif mom.status() == 2 and self.isFromB(mom, done=done):
                return True
        return False

    def matchAnyLeptons(self, event): 
        event.anyLeptons = [ x for x in event.genParticles if x.status() == 1 and abs(x.pdgId()) in [11,13] ]
        leps = event.inclusiveLeptons if hasattr(event, 'inclusiveLeptons') else event.selectedLeptons
        match = matchObjectCollection3(leps, event.anyLeptons, deltaRMax = 0.3, filter = lambda x,y : abs(x.pdgId()) == abs(y.pdgId()))
        for lep in leps:
            gen = match[lep]
            lep.mcMatchAny_gp = gen
            if gen:
                if   self.isFromB(gen):       lep.mcMatchAny = 5 # B (inclusive of B->D)
                elif self.isFromB(gen,bid=4): lep.mcMatchAny = 4 # Charm
                else: lep.mcMatchAny = 1
            else: 
                lep.mcMatchAny = 0
            # fix case where the matching with the only prompt leptons failed, but we still ended up with a prompt match
            if gen != None and hasattr(lep,'mcMatchId') and lep.mcMatchId == 0:
                if isPromptLepton(gen, False): lep.mcMatchId = 100
            elif not hasattr(lep,'mcMatchId'):
                lep.mcMatchId = 0
            if not hasattr(lep,'mcMatchTau'): lep.mcMatchTau = 0

    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        #call the leptons functions
        self.makeLeptons(event)

        if self.cfg_comp.isMC and self.cfg_ana.do_mc_match:
            self.matchLeptons(event)
            self.matchAnyLeptons(event)
            
        return True

#A default config
setattr(LeptonAnalyzer,"defaultConfig",cfg.Analyzer(
    verbose=False,
    class_object=LeptonAnalyzer,
    # input collections
    muons='slimmedMuons',
    electrons='slimmedElectrons',
    rhoMuon= 'fixedGridRhoFastjetAll',
    rhoElectron = 'fixedGridRhoFastjetAll',
##    photons='slimmedPhotons',
    # energy scale corrections and ghost muon suppression (off by default)
    doMuScleFitCorrections=False, # "rereco"
    doRochesterCorrections=False,
    doElectronScaleCorrections=False, # "embedded" in 5.18 for regression
    doSegmentBasedMuonCleaning=False,
    # inclusive very loose muon selection
    inclusive_muon_id  = "POG_ID_Loose",
    inclusive_muon_pt  = 3,
    inclusive_muon_eta = 2.4,
    inclusive_muon_dxy = 0.5,
    inclusive_muon_dz  = 1.0,
    muon_dxydz_track   = "muonBestTrack",
    # loose muon selection
    loose_muon_id     = "POG_ID_Loose",
    loose_muon_pt     = 5,
    loose_muon_eta    = 2.4,
    loose_muon_dxy    = 0.05,
    loose_muon_dz     = 0.2,
    loose_muon_relIso = 0.4,
    # loose_muon_isoCut = lambda muon :muon.miniRelIso < 0.2 
    # inclusive very loose electron selection
    inclusive_electron_id  = "",
    inclusive_electron_pt  = 5,
    inclusive_electron_eta = 2.5,
    inclusive_electron_dxy = 0.5,
    inclusive_electron_dz  = 1.0,
    inclusive_electron_lostHits = 1.0,
    # loose electron selection
    loose_electron_id     = "", #POG_MVA_ID_NonTrig_full5x5",
    loose_electron_pt     = 7,
    loose_electron_eta    = 2.4,
    loose_electron_dxy    = 0.05,
    loose_electron_dz     = 0.2,
    loose_electron_relIso = 0.4,
    # loose_electron_isoCut = lambda electron : electron.miniRelIso < 0.1
    loose_electron_lostHits = 1.0,
    # muon isolation correction method (can be "rhoArea" or "deltaBeta")
    mu_isoCorr = "rhoArea" ,
    mu_effectiveAreas = "Phys14_25ns_v1", #(can be 'Data2012' or 'Phys14_25ns_v1')
    mu_tightId = "POG_ID_Tight" ,
    # electron isolation correction method (can be "rhoArea" or "deltaBeta")
    ele_isoCorr = "rhoArea" ,
    ele_effectiveAreas = "Spring15_25ns_v1" , #(can be 'Data2012' or 'Phys14_25ns_v1', or 'Spring15_50ns_v1' or 'Spring15_25ns_v1')
    ele_tightId = "Cuts_2012" ,
    # minimum deltaR between a loose electron and a loose muon (on overlaps, discard the electron)
    min_dr_electron_muon = 0.02,
    # Mini-isolation, with pT dependent cone: will fill in the miniRelIso, miniRelIsoCharged, miniRelIsoNeutral variables of the leptons (see https://indico.cern.ch/event/368826/ )
    doMiniIsolation = False, # off by default since it requires access to all PFCandidates 
    packedCandidates = 'packedPFCandidates',
    miniIsolationPUCorr = 'rhoArea', # Allowed options: 'rhoArea' (EAs for 03 cone scaled by R^2), 'deltaBeta', 'raw' (uncorrected), 'weights' (delta beta weights; not validated)
                                     # Choose None to just use the individual object's PU correction
    miniIsolationVetoLeptons = None, # use 'inclusive' to veto inclusive leptons and their footprint in all isolation cones
    # Activity Annulus
    doIsoAnnulus = False, # off by default since it requires access to all PFCandidates 
    # do MC matching 
    do_mc_match = True, # note: it will in any case try it only on MC, not on data
    match_inclusiveLeptons = False, # match to all inclusive leptons
    )
)
