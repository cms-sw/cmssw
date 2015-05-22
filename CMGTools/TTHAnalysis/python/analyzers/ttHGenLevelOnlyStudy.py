import operator 
import itertools
import copy

from ROOT import TLorentzVector

from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.Event import Event
from CMGTools.RootTools.statistics.Counter import Counter, Counters
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.physicsobjects.Lepton import Lepton
from CMGTools.RootTools.physicsobjects.Photon import Photon
from CMGTools.RootTools.physicsobjects.Electron import Electron
from CMGTools.RootTools.physicsobjects.Muon import Muon
from CMGTools.RootTools.physicsobjects.Jet import Jet
from CMGTools.RootTools.physicsobjects.PhysicsObjects import GenParticle

from CMGTools.RootTools.utils.DeltaR import deltaR,deltaPhi
from CMGTools.RootTools.physicsobjects.genutils import *

class LeptonFromGen:
    def __init__(self, physObj):
        self.physObj = physObj
    def __getattr__(self, attr):
        if hasattr(self.physObj, attr):
            return getattr(self.physObj, attr)
        raise RuntimeError, "Missing attribute '%s'" % attr
class JetFromGen:
    def __init__(self, physObj):
        self.physObj = physObj
        self.btag = False
    def btagCSV(self,tag):
        return self.btag
    def __getattr__(self, attr):
        if hasattr(self.physObj, attr):
            return getattr(self.physObj, attr)
        raise RuntimeError, "Missing attribute '%s'" % attr

class ttHGenLevelOnlyStudy( Analyzer ):
    """
    Fakes a reco event starting from GEN-only files      
    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHGenLevelOnlyStudy,self).__init__(cfg_ana,cfg_comp,looperName)
        self.doPDFWeights = hasattr(self.cfg_ana, "PDFWeights") and len(self.cfg_ana.PDFWeights) > 0
        if self.doPDFWeights:
            self.pdfWeightInit = False
    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------
        

    def declareHandles(self):
        super(ttHGenLevelOnlyStudy, self).declareHandles()
        self.mchandles['genParticles'] = AutoHandle( 'genParticles', 'std::vector<reco::GenParticle>' )
        self.mchandles['jets'] = AutoHandle( 'ak5GenJets', 'std::vector<reco::GenJet>' )
        self.mchandles['met'] = AutoHandle( 'genMetTrue', 'std::vector<reco::GenMET>' )
        if self.doPDFWeights:
            self.mchandles['pdfstuff'] = AutoHandle( 'generator', 'GenEventInfoProduct' )

    def beginLoop(self):
        super(ttHGenLevelOnlyStudy,self).beginLoop()

    def doLeptons(self,iEvent,event):
            
        event.selectedLeptons = []
        for l in event.genParticles: 
            if abs(l.pdgId()) not in [11,13] or l.status() != 1: continue
            if abs(l.pdgId()) == 13:
                if l.pt() <= 5 or abs(l.eta()) > 2.4: continue
            if abs(l.pdgId()) == 11:
                if l.pt() <= 7 or abs(l.eta()) > 2.5: continue
            if not isNotFromHadronicShower(l):
                continue
            event.selectedLeptons.append(LeptonFromGen(l))

    def doJets(self,iEvent,event):
        event.cleanJetsAll = []
        event.cleanJetsFwd = []
        event.cleanJets = []
        for j in self.mchandles['jets'].product(): 
            if j.pt() < 25: continue
            for l in event.selectedLeptons:
                if l.pt() > 10 and deltaR(l.eta(),l.phi(),j.eta(),j.phi()) < 0.5:
                    continue
            jo = JetFromGen(j)
            event.cleanJetsAll.append(jo)
            if abs(j.eta()) < 2.4:
                event.cleanJetsAll.append(jo)
            else:
                event.cleanJetsFwd.append(jo)
    def doBTag(self,iEvent,event):
        bs = []
        for gp in event.genParticles:
            if gp.status() != 2: continue
            id = abs(gp.pdgId())
            if id == 5 or ((id % 1000) / 100) == 5 or ((id % 10000)/1000) == 5:
                bs.append(gp)
        for j in event.cleanJetsAll:
            for gp in bs:
                if deltaR(gp.eta(),gp.phi(),j.eta(),j.phi()) < 0.4:
                    gp.btag = True
    def doMET(self,iEvent,event):
        event.met = self.mchandles['met'].product().front()

    def makeZs(self, event, maxLeps):
        event.bestZ1 = [ 0., -1,-1 ]
        event.bestZ1sfss = [ 0., -1,-1 ]
        event.bestZ2 = [ 0., -1,-1, 0. ]
        nlep = len(event.selectedLeptons)
        for i,l1 in enumerate(event.selectedLeptons):
            for j in range(i+1,nlep):
                if j >= maxLeps: break
                l2 = event.selectedLeptons[j]    
                if l1.pdgId() == -l2.pdgId():
                    zmass = (l1.p4() + l2.p4()).M()
                    if event.bestZ1[0] == 0 or abs(zmass - 91.188) < abs(event.bestZ1[0] - 91.188):
                        event.bestZ1 = [ zmass, i, j ]
                if l1.pdgId() == l2.pdgId():
                    zmass = (l1.p4() + l2.p4()).M()
                    if event.bestZ1sfss[0] == 0 or abs(zmass - 91.188) < abs(event.bestZ1sfss[0] - 91.188):
                        event.bestZ1sfss = [ zmass, i, j ]
        if event.bestZ1[0] != 0 and nlep > 3:
            for i,l1 in enumerate(event.selectedLeptons):
                if i == event.bestZ1[1]: continue
                for j in range(i+1,nlep):
                    if j >= maxLeps: break
                    if j == event.bestZ1[2]: continue
                    l2 = event.selectedLeptons[j]    
                    if l1.pdgId() == -l2.pdgId():
                        if l1.pt() + l2.pt() > event.bestZ2[0]:
                            event.bestZ2 = [ l1.pt() + l2.pt(), i, j, (l1.p4() + l2.p4()).M() ]

    def makeMlls(self, event, maxLeps):
        mllsfos = self.mllValues(event,  lambda l1,l2 : l1.pdgId()  == -l2.pdgId(),  maxLeps)
        mllafos = self.mllValues(event,  lambda l1,l2 : l1.charge() == -l2.charge(),  maxLeps)
        mllafss = self.mllValues(event,  lambda l1,l2 : l1.charge() ==  l2.charge(),  maxLeps)
        mllafas = self.mllValues(event,  lambda l1,l2 : True, maxLeps)
        event.minMllSFOS = min(mllsfos)
        event.minMllAFOS = min(mllafos)
        event.minMllAFSS = min(mllafss)
        event.minMllAFAS = min(mllafas)
        event.maxMllSFOS = max(mllsfos)
        event.maxMllAFAS = max(mllafas)
        event.maxMllAFOS = max(mllafos)
        event.maxMllAFSS = max(mllafss)
        drllafos = self.drllValues(event,  lambda l1,l2 : l1.charge() == -l2.charge(),  maxLeps)
        drllafss = self.drllValues(event,  lambda l1,l2 : l1.charge() ==  l2.charge(),  maxLeps)
        event.minDrllAFSS = min(drllafss)
        event.minDrllAFOS = min(drllafos)
        event.maxDrllAFOS = max(drllafos)
        event.maxDrllAFSS = max(drllafss)
        ptllafos = self.ptllValues(event,  lambda l1,l2 : l1.charge() == -l2.charge(),  maxLeps)
        ptllafss = self.ptllValues(event,  lambda l1,l2 : l1.charge() ==  l2.charge(),  maxLeps)
        event.minPtllAFSS = min(ptllafss)
        event.minPtllAFOS = min(ptllafos)
        event.maxPtllAFOS = max(ptllafos)
        event.maxPtllAFSS = max(ptllafss)
        leps = event.selectedLeptons; nlep = len(leps)
        event.m2l = (leps[0].p4() + leps[1].p4()).M() if nlep >= 2 else 0
        event.pt2l = (leps[0].p4() + leps[1].p4()).Pt() if nlep >= 2 else 0
        event.q3l = sum([l.charge() for l in leps[:2]]) if nlep >= 3 else 0
        event.ht3l = sum([l.pt() for l in leps[:2]]) if nlep >= 3 else 0
        event.pt3l = (leps[0].p4() + leps[1].p4() + leps[2].p4()).Pt() if nlep >= 3 else 0
        event.m3l = (leps[0].p4() + leps[1].p4() + leps[2].p4()).M() if nlep >= 3 else 0
        event.q4l = sum([l.charge() for l in leps[:3]])  if nlep >= 4 else 0
        event.ht4l = sum([l.pt() for l in leps[:3]]) if nlep >= 4 else 0
        event.pt4l = (leps[0].p4() + leps[1].p4() + leps[2].p4() + leps[3].p4()).Pt() if nlep >= 4 else 0
        event.m4l = (leps[0].p4() + leps[1].p4() + leps[2].p4() + leps[3].p4()).M() if nlep >= 4 else 0

    def mllValues(self, event, pairSelection, maxLeps):
        return self.llValues(event, lambda l1,l2: (l1.p4() + l2.p4()).M(), pairSelection, maxLeps)

    def drllValues(self, event, pairSelection, maxLeps):
        return self.llValues(event, lambda l1,l2: deltaR(l1.eta(), l1.phi(), l2.eta(), l2.phi()), pairSelection, maxLeps)

    def ptllValues(self, event, pairSelection, maxLeps):
        return self.llValues(event, lambda l1,l2: (l1.p4() + l2.p4()).Pt(), pairSelection, maxLeps)

    def llValues(self, event, function, pairSelection, maxLeps):
        pairs = []
        nlep = len(event.selectedLeptons)
        for i,l1 in enumerate(event.selectedLeptons):
            for j in range(i+1,nlep):
                if j >= maxLeps: break
                l2 = event.selectedLeptons[j]    
                if pairSelection(l1,l2):
                    pairs.append( function(l1, l2) )
        if pairs == []: pairs.append(-1)
        return pairs


    def initPDFWeights(self):
        from ROOT import PdfWeightProducerTool
        self.pdfWeightInit = True
        self.pdfWeightTool = PdfWeightProducerTool()
        for pdf in self.cfg_ana.PDFWeights:
            self.pdfWeightTool.addPdfSet(pdf+".LHgrid")
        self.pdfWeightTool.beginJob()

    def makePDFWeights(self, event):
        if not self.pdfWeightInit: self.initPDFWeights()
        self.pdfWeightTool.processEvent(self.mchandles['pdfstuff'].product())
        event.pdfWeights = {}
        for pdf in self.cfg_ana.PDFWeights:
            ws = self.pdfWeightTool.getWeights(pdf+".LHgrid")
            event.pdfWeights[pdf] = [w for w in ws]
            #print "Produced %d weights for %s: %s" % (len(ws),pdf,event.pdfWeights[pdf])

    def doHT(self, iEvent, event):
        import ROOT

        objects25 = [ j for j in event.cleanJets if j.pt() > 25 ] + event.selectedLeptons
        objects30 = [ j for j in event.cleanJets if j.pt() > 30 ] + event.selectedLeptons
        objects40  = [ j for j in event.cleanJets if j.pt() > 40 ] + event.selectedLeptons
        objects40j = [ j for j in event.cleanJets if j.pt() > 40 ] 

        event.htJet25 = sum([x.pt() for x in objects25])
        event.mhtJet25vec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in objects25])) , -1.*(sum([x.py() for x in objects25])), 0, 0 )     
        event.mhtPhiJet25 = event.mhtJet25vec.phi()
        event.mhtJet25 = event.mhtJet25vec.pt()

        event.htJet30 = sum([x.pt() for x in objects30])
        event.mhtJet30vec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in objects30])) , -1.*(sum([x.py() for x in objects30])), 0, 0 )             
        event.mhtJet30 = event.mhtJet30vec.pt()
        event.mhtPhiJet30 = event.mhtJet30vec.phi()

        event.htJet40 = sum([x.pt() for x in objects40])
        event.mhtJet40vec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in objects40])) , -1.*(sum([x.py() for x in objects40])), 0, 0 )             
        event.mhtJet40 = event.mhtJet40vec.pt()
        event.mhtPhiJet40 = event.mhtJet40vec.phi()

        event.htJet40j = sum([x.pt() for x in objects40j])
        event.mhtJet40jvec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in objects40j])) , -1.*(sum([x.py() for x in objects40j])), 0, 0 )                     
        event.mhtJet40j = event.mhtJet40jvec.pt()
        event.mhtPhiJet40j = event.mhtJet40jvec.phi()

    def process(self, iEvent, event):
        self.readCollections( iEvent )

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        event.genParticles = [ gp for gp in self.mchandles['genParticles'].product() ]

        event.eventWeigth = 1.0
        event.run = iEvent.eventAuxiliary().id().run()
        event.lumi = iEvent.eventAuxiliary().id().luminosityBlock()
        event.eventId = iEvent.eventAuxiliary().id().event()
        
        self.doLeptons(iEvent,event)
        self.makeZs(event, 4)
        self.makeMlls(event, 4)
        self.doJets(iEvent,event)
        self.doBTag(iEvent,event)
        self.doMET(iEvent,event)
        self.doHT(iEvent,event)

        # do PDF weights, if requested
        if self.doPDFWeights:
            self.makePDFWeights(event)
        return True
