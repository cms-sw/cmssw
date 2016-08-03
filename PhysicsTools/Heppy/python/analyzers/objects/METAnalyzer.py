from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.HeppyCore.utils.deltar import * 
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import Type1METCorrector, setFakeRawMETOnOldMiniAODs
import PhysicsTools.HeppyCore.framework.config as cfg

import copy
import ROOT
from math import hypot

from copy import deepcopy

class METAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(METAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.recalibrateMET   = cfg_ana.recalibrate
        self.applyJetSmearing = cfg_comp.isMC and cfg_ana.applyJetSmearing
        self.old74XMiniAODs         = cfg_ana.old74XMiniAODs
        self.jetAnalyzerPostFix = getattr(cfg_ana, 'jetAnalyzerPostFix', '')
        if self.recalibrateMET in [ "type1", True ]:
            if self.recalibrateMET == "type1":
                self.type1METCorrector = Type1METCorrector(self.old74XMiniAODs)
        elif self.recalibrateMET != False:
            raise RuntimeError("Unsupported value %r for option 'recalibrate': allowed are True, False, 'type1'" % self.recalibrateMET)

    def declareHandles(self):
        super(METAnalyzer, self).declareHandles()
        self.handles['met'] = AutoHandle( self.cfg_ana.metCollection, 'std::vector<pat::MET>' )
        if self.cfg_ana.doMetNoPU: 
            self.handles['nopumet'] = AutoHandle( self.cfg_ana.noPUMetCollection, 'std::vector<pat::MET>' )
        if self.cfg_ana.doTkMet:
            self.handles['cmgCand'] = AutoHandle( self.cfg_ana.candidates, self.cfg_ana.candidatesTypes )
            #self.handles['vertices'] =  AutoHandle( "offlineSlimmedPrimaryVertices", 'std::vector<reco::Vertex>', fallbackLabel="offlinePrimaryVertices" )
            self.mchandles['packedGen'] = AutoHandle( 'packedGenParticles', 'std::vector<pat::PackedGenParticle>' )

    def beginLoop(self, setup):
        super(METAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    def applyDeltaMet(self, met, deltaMet):
        px,py = self.met.px()+deltaMet[0], self.met.py()+deltaMet[1]
        met.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))

    def adduParaPerp(self, met, boson, postfix):

        upara = 0
        uperp = 0
        uX = - met.px() - boson.px()
        uY = - met.py() - boson.py()
        u1 = (uX*boson.px() + uY*boson.py())/boson.pt()
        u2 = (uX*boson.py() - uY*boson.px())/boson.pt()

        setattr(met, "upara"+postfix, u1)
        setattr(met, "uperp"+postfix, u2)

    def makeTkMETs(self, event):
        charged = []
        chargedchs = []
        chargedPVLoose = []
        chargedPVTight = []
        dochs=getattr(self.cfg_ana,"includeTkMetCHS",True)       
        dotight=getattr(self.cfg_ana,"includeTkMetPVTight",True)       
        doloose=getattr(self.cfg_ana,"includeTkMetPVLoose",True)       
        pfcands = self.handles['cmgCand'].product()

        for pfcand in pfcands:

## ===> require the Track Candidate charge and with a  minimum dz 
            if (pfcand.charge()!=0):

                pvflag = pfcand.fromPV()
                pxy = pfcand.px(), pfcand.py()

                if abs(pfcand.dz())<=self.cfg_ana.dzMax:
                    charged.append(pxy)

                if dochs and  pvflag>0:
                    chargedchs.append(pxy)

                if doloose and pvflag>1:
                    chargedPVLoose.append(pxy)

                if dotight and pvflag>2:
                    chargedPVTight.append(pxy)

        def sumXY(pxys):
            px, py = sum(x[0] for x in pxys), sum(x[1] for x in pxys)
            return ROOT.reco.Particle.LorentzVector(-px, -py, 0, hypot(px,py))
        setattr(event, "tkMet"+self.cfg_ana.collectionPostFix, sumXY(charged))
        setattr(event, "tkMetPVchs"+self.cfg_ana.collectionPostFix, sumXY(chargedchs))
        setattr(event, "tkMetPVLoose"+self.cfg_ana.collectionPostFix, sumXY(chargedPVLoose))
        setattr(event, "tkMetPVTight"+self.cfg_ana.collectionPostFix, sumXY(chargedPVTight))
        getattr(event,"tkMet"+self.cfg_ana.collectionPostFix).sumEt = sum([hypot(x[0],x[1]) for x in charged])
        getattr(event,"tkMetPVchs"+self.cfg_ana.collectionPostFix).sumEt = sum([hypot(x[0],x[1]) for x in chargedchs])
        getattr(event,"tkMetPVLoose"+self.cfg_ana.collectionPostFix).sumEt = sum([hypot(x[0],x[1]) for x in chargedPVLoose])
        getattr(event,"tkMetPVTight"+self.cfg_ana.collectionPostFix).sumEt = sum([hypot(x[0],x[1]) for x in chargedPVTight])

        if  hasattr(event,'zll_p4'):
            self.adduParaPerp(getattr(event,"tkMet"+self.cfg_ana.collectionPostFix), event.zll_p4,"_zll")
            self.adduParaPerp(getattr(event,"tkMetPVchs"+self.cfg_ana.collectionPostFix), event.zll_p4,"_zll")
            self.adduParaPerp(getattr(event,"tkMetPVLoose"+self.cfg_ana.collectionPostFix), event.zll_p4,"_zll")
            self.adduParaPerp(getattr(event,"tkMetPVTight"+self.cfg_ana.collectionPostFix), event.zll_p4,"_zll")


    def makeGenTkMet(self, event):
        genCharged = [ (x.px(),x.py()) for x in self.mchandles['packedGen'].product() if x.charge() != 0 and abs(x.eta()) < 2.4 ]
        px, py = sum(x[0] for x in genCharged), sum(x[1] for x in genCharged)
        setattr(event,"tkGenMet"+self.cfg_ana.collectionPostFix, ROOT.reco.Particle.LorentzVector(-px , -py, 0, hypot(px,py)))

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
        self.metNoMu.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
        px,py = self.metNoMuNoPU.px()+mupx, self.metNoMuNoPU.py()+mupy
        self.metNoMuNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
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
        self.metNoEle.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))

        px,py = self.metNoEleNoPU.px()+elepx, self.metNoEleNoPU.py()+elepy
        self.metNoEleNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
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
        self.metNoPhoton.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
        setattr(event, "metNoPhoton"+self.cfg_ana.collectionPostFix, self.metNoPhoton)
        if self.cfg_ana.doMetNoPU: 
          self.metNoPhotonNoPU = copy.deepcopy(self.metNoPU)
          px,py = self.metNoPhotonNoPU.px()+phopx, self.metNoPhotonNoPU.py()+phopy
          self.metNoPhotonNoPU.setP4(ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
          setattr(event, "metNoPhotonNoPU"+self.cfg_ana.collectionPostFix, self.metNoPhotonNoPU)


    def makeMETs(self, event):
        import ROOT
        if self.cfg_ana.copyMETsByValue:
          self.met = ROOT.pat.MET(self.handles['met'].product()[0])
          if self.cfg_ana.doMetNoPU: self.metNoPU = ROOT.pat.MET(self.handles['nopumet'].product()[0])
        else:
          self.met = self.handles['met'].product()[0]
          if self.cfg_ana.doMetNoPU: self.metNoPU = self.handles['nopumet'].product()[0]

        if self.recalibrateMET == "type1":
          type1METCorr = getattr(event, 'type1METCorr'+self.jetAnalyzerPostFix)
          self.type1METCorrector.correct(self.met, type1METCorr)
        elif self.recalibrateMET == True:
          deltaMetJEC = getattr(event, 'deltaMetFromJEC'+self.jetAnalyzerPostFix)
          self.applyDeltaMet(self.met, deltaMetJEC)
        if self.applyJetSmearing:
          deltaMetSmear = getattr(event, 'deltaMetFromJetSmearing'+self.jetAnalyzerPostFix)
          self.applyDeltaMet(self.met, deltaMetSmear)


        if (not self.cfg_ana.copyMETsByValue) and getattr(self.cfg_ana, 'makeShiftedMETs', True):
          shifts = [] 
          for obj in 'JetEn', 'JetRes', 'MuonEn', 'ElectronEn', 'PhotonEn', 'TauEn', 'UnclusteredEn':
            for sh in 'Up', 'Down':
                shifts.append( (obj+sh, getattr(self.met,obj+sh)) )
          shifts.append( ('NoShift', self.met.NoShift) )
          for name,i in shifts:
               key = i
               m = ROOT.pat.MET(self.met)
               if self.old74XMiniAODs:
                    if key > 12:   key = 12
                    elif key <= 3: key = { 'JetEnUp':0, 'JetEnDown':1, 'JetResUp':2, 'JetResDown':3 }[name]
               m.setP4(self.met.shiftedP4(key))
               setattr(event, "met{0}_shifted_{1}".format(self.cfg_ana.collectionPostFix, i),m)
               setattr(event, "met{0}_shifted_{1}".format(self.cfg_ana.collectionPostFix, name),m)

        self.met_sig = self.met.significance()
        self.met_sumet = self.met.sumEt()

        if self.old74XMiniAODs and self.recalibrateMET != "type1":
           oldraw = self.met.shiftedP2_74x(12,0);
           setFakeRawMETOnOldMiniAODs( self.met, oldraw.px, oldraw.py, self.met.shiftedSumEt_74x(12,0) )
           px, py = oldraw.px, oldraw.py
        else:
           px, py = self.met.uncorPx(), self.met.uncorPy()
        self.met_raw = ROOT.reco.Particle.LorentzVector(px,py,0,hypot(px,py))

        if hasattr(event,'zll_p4'):
            self.adduParaPerp(self.met,event.zll_p4,"_zll")
            self.adduParaPerp(self.met_raw, event.zll_p4,"_zll")
            setattr(event,"met_raw"+self.cfg_ana.collectionPostFix, self.met_raw)
            setattr(event,"met_raw.upara_zll"+self.cfg_ana.collectionPostFix, self.met_raw.upara_zll)
            setattr(event,"met_raw.uperp_zll"+self.cfg_ana.collectionPostFix, self.met_raw.uperp_zll)

        if hasattr(event,'gamma_p4'):
            self.adduParaPerp(self.met,event.gamma_p4,"_gamma")
            self.adduParaPerp(self.met_raw, event.gamma_p4,"_gamma")
            setattr(event,"met_raw"+self.cfg_ana.collectionPostFix, self.met_raw)
            setattr(event,"met_raw.upara_gamma"+self.cfg_ana.collectionPostFix, self.met_raw.upara_gamma)
            setattr(event,"met_raw.uperp_gamma"+self.cfg_ana.collectionPostFix, self.met_raw.uperp_gamma)

        if hasattr(event,"met"+self.cfg_ana.collectionPostFix): raise RuntimeError, "Event already contains met with the following postfix: "+self.cfg_ana.collectionPostFix
        setattr(event, "met"+self.cfg_ana.collectionPostFix, self.met)
        if self.cfg_ana.doMetNoPU: setattr(event, "metNoPU"+self.cfg_ana.collectionPostFix, self.metNoPU)
        setattr(event, "met_sig"+self.cfg_ana.collectionPostFix, self.met_sig)
        setattr(event, "met_sumet"+self.cfg_ana.collectionPostFix, self.met_sumet)

        genMET = self.met.genMET()
        if genMET:
          setattr(event, "met_genPt"+self.cfg_ana.collectionPostFix, genMET.pt())
          setattr(event, "met_genPhi"+self.cfg_ana.collectionPostFix, genMET.phi())
        else:
          setattr(event, "met_genPt"+self.cfg_ana.collectionPostFix, float('nan'))
          setattr(event, "met_genPhi"+self.cfg_ana.collectionPostFix, float('nan'))

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

        if getattr(self.cfg_ana,"doTkGenMet",self.cfg_ana.doTkMet) and self.cfg_comp.isMC and hasattr(event, 'genParticles'):
            self.makeGenTkMet(event)

        return True


setattr(METAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = METAnalyzer,
    metCollection     = "slimmedMETs",
    noPUMetCollection = "slimmedMETs",
    copyMETsByValue = False,
    recalibrate = True,
    applyJetSmearing = True,
    jetAnalyzerPostFix = "",
    old74XMiniAODs = False, # need to set to True to get proper Raw MET on plain 74X MC produced with CMSSW <= 7_4_12
    makeShiftedMETs = True,
    doTkMet = False,
    includeTkMetCHS = True,
    includeTkMetPVLoose = True,
    includeTkMetPVTight = True,
    doMetNoPU = False, # Not existing in MiniAOD at the moment
    doMetNoMu = False,  
    doMetNoEle = False,  
    doMetNoPhoton = False,  
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    collectionPostFix = "",
    )
)
