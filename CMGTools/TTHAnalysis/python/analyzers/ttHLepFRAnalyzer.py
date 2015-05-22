from math import *

from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle

from CMGTools.RootTools.utils.DeltaR import deltaR, deltaPhi

from CMGTools.RootTools.physicsobjects.JetReCalibrator import Type1METCorrection
from CMGTools.TTHAnalysis.analyzers.ttHLepTreeProducerNew import *
import ROOT

def mtw(x1,x2):
    return sqrt(2*x1.pt()*x2.pt()*(1-cos(x1.phi()-x2.phi())))
def dR(x,y,x2=None,y2=None):
    if x2 != None: return deltaR(x,y,x2,y2)
    return deltaR(x.eta(),x.phi(),y.eta(),y.phi())
def dPhi(x,y):
    return deltaPhi(x.phi() if hasattr(x,'phi') else x, 
                     y.phi() if hasattr(y,'phi') else y)

class ttHLepFRAnalyzer( ttHLepTreeProducerNew ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHLepFRAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 

        self.globalVariables = [ 
            NTupleVariable("nVert",  lambda ev: len(ev.goodVertices), int, help="Number of good vertices"),
            NTupleVariable("mtw",    lambda ev: mtw(ev.selectedLeptons[0], ev.met), help="M_{T}(W) with lepton and default met"),
            NTupleVariable("mtwRaw", lambda ev: mtw(ev.selectedLeptons[0], ev.metRaw), help="M_{T}(W) with lepton and raw met"),
            NTupleVariable("mtwT1",  lambda ev: mtw(ev.selectedLeptons[0], ev.metT1), help="M_{T}(W) with lepton and python type1 met"),
            NTupleVariable("dphi",   lambda ev: dPhi(ev.selectedLeptons[0], ev.tag) if ev.tag != None else 0, help="#Delta#phi between tag and probe"),
            NTupleVariable("dr",     lambda ev:   dR(ev.selectedLeptons[0], ev.tag) if ev.tag != None else 0, help="#DeltaR between tag and probe"),
            NTupleVariable("nJet30", lambda ev: sum([j.pt() > 30 for j in ev.cleanJets]), int, help="Number of jets with pt > 30"),
            NTupleVariable("nJet40", lambda ev: sum([j.pt() > 40 for j in ev.cleanJets]), int, help="Number of jets with pt > 40"),
            NTupleVariable("nBJetMedium30", lambda ev: sum([j.btagWP("CSVv2IVFM") for j in ev.cleanJets if j.pt() > 30]), int, help="Number of jets with pt > 30 passing CSV medium"),
            NTupleVariable("nBJetMedium40", lambda ev: sum([j.btagWP("CSVv2IVFM") for j in ev.cleanJets if j.pt() > 40]), int, help="Number of jets with pt > 40 passing CSV medium"),
        ]
        self.globalObjects = {
            "probe"  : NTupleObject("Probe",   leptonTypeSusyFR, help="Probe lepton"),
            "met"    : NTupleObject("met",     metType, help="PF E_{T}^{miss}, after default type 1 corrections"),
            "metRaw" : NTupleObject("metRaw",  metType, help="PF E_{T}^{miss}, without type 1 corrections"),
            #"metOld" : NTupleObject("metOld", metType, help="PF E_{T}^{miss}, with old type 1 corrections"),
            "metT1"  : NTupleObject("metT1",   metType, help="PF E_{T}^{miss}, after python type 1 corrections"),
        }
        self.collections = {
            "tagJets" : NTupleCollection("Jet", jetTypeTTH, 3, sortDescendingBy = lambda jet : jet.pt(), help="Tag jet (#DeltaR > 1.0 from the probe)"),
        }

        # MET computer
        if self.cfg_comp.isMC:
            self.type1MET = Type1METCorrection("START53_V20","AK5PF",  False)
        else:
            self.type1MET = Type1METCorrection("GR_P_V42_AN4","AK5PF", True) 

        ## Now book the variables
        self.initDone = True
        self.declareVariables()

    def declareHandles(self):
        super(ttHLepFRAnalyzer, self).declareHandles()
        self.handles['met']    = AutoHandle( 'cmgPFMET', 'std::vector<cmg::BaseMET>' )
        self.handles['metRaw'] = AutoHandle( 'cmgPFMETRaw', 'std::vector<cmg::BaseMET>' )
        self.handles['muons']  = AutoHandle( 'cmgMuonSel', 'std::vector<cmg::Muon>' )            
        self.handles['rho']    = AutoHandle( ('kt6PFJets','rho',''), 'double' )

    def beginLoop(self):
        super(ttHLepFRAnalyzer,self).beginLoop()
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')
        count.register('one lepton')

    def process(self, iEvent, event):
        self.readCollections( iEvent )

        # MET calculation
        event.met = self.handles['met'].product()[0]
        event.metRaw = self.handles['metRaw'].product()[0]
        event.metOld = event.met.__class__(event.met)
        event.metT1  = event.met.__class__(event.metRaw)
        px0,  py0 = event.metRaw.px(), event.metRaw.py()
        dpx0, dpy0 = self.type1MET.getMETCorrection( event.allJetsUsedForMET, float(self.handles['rho'].product()[0]),  self.handles['muons'].product())
        event.metT1.setP4(ROOT.reco.Particle.LorentzVector(px0+dpx0,py0+dpy0, 0, hypot(px0+dpx0,py0+dpy0)))
        if hasattr(event, 'deltaMetFromJEC'):
            px,py = event.met.px()+event.deltaMetFromJEC[0], event.met.py()+event.deltaMetFromJEC[1]
            event.met.setP4(  ROOT.reco.Particle.LorentzVector(px,py, 0, hypot(px,py)))
            #print "run ",event.run," lumi ", event.lumi," event ", event.eventId, ": MET correction: correct %+7.3f %+7.3f    by hand %+7.3f %+7.3f    diff %+7.3f %+7.3f (phi %+5.3f) " % (event.met.px()-event.metRaw.px(), event.met.py()-event.metRaw.py(), px0-event.metRaw.px(), py0-event.metRaw.py(), event.met.px()-px0,event.met.py()-py0, atan2(event.met.py()-py0, event.met.px()-px0))
            #print "run ",event.run," lumi ", event.lumi," event ", event.eventId, ": old value: ",event.met.pt()," new value: ",hypot(px,py),"  raw met: ",event.metRaw.pt()," type 1 by hand: ",hypot(px0,py0)
            
        self.counters.counter('pairs').inc('all events')

        if len([l for l in event.selectedLeptons if l.pt() > 10])==1: 
            self.counters.counter('pairs').inc('one lepton')
            for lep in event.selectedLeptons:
                lep.mvaValue = -99.0
                event.probe = lep
                event.tagJets = [ j for j in event.cleanJets if dR(j,lep) > 1.0 ]
                event.tag = event.tagJets[0] if len(event.tagJets) else None
                self.fillTree(iEvent, event)
                break
         
        return True
