import ROOT,itertools,math      
from array import array          
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()
# 

#0.005
#idCut=[
#20, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 48, 44, 44, 44, 60, 20, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 48, 20, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 60, 20, 24, 20, 44, 44, 44, 44, 44, 44, 44, 44, 20, 44, 44, 44, 60, 20, 24, 24, 24, 24, 44, 24, 24, 24, 20, 44, 40, 44, 44, 48, 44, 20, 44, 44, 48, 48, 48, 48, 60, 48, 48, 60, 60, 48, 60, 60, 60, 20, 44, 60, 60, 60, 60, 60, 60, 60, 48, 60, 60, 44, 60, 60, 60, 20, 40, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 16, 20, 60, 60, 60, 60, 60, 60, 60, 60, 48, 60, 44, 60, 60, 60, 16, 20, 44, 60, 60, 60, 60, 60, 48, 60, 60, 60, 60, 60, 60, 60, 20, 20, 40, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 48, 60, 20, 16, 20, 44, 48, 44, 44, 44, 60, 60, 60, 60, 48, 48, 48, 60, 20, 20, 20, 60, 44, 60, 60, 60, 60, 44, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]

#0.006

idCut=[
36, 56, 64, 64, 32, 32, 60, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #0
28, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 28, 28, 28, 28,       #16
28, 52, 56, 52, 52, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #32
28, 52, 52, 52, 56, 56, 56, 56, 56, 56, 56, 28, 56, 56, 56, 56,       #48
28, 36, 36, 36, 36, 36, 36, 36, 36, 36, 56, 56, 36, 56, 36, 36,       #64
28, 56, 60, 60, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #80
28, 92, 120, 120, 92, 92, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,     #96
28, 120, 152, 128, 100, 124, 80, 80, 64, 64, 64, 64, 64, 64, 64,64,   #112 
28, 88, 92, 64, 92, 60, 60, 60, 92, 60, 60, 60, 60, 60, 60, 60,       #128   
36, 88, 88, 64, 64, 60, 60, 60, 60, 60, 60, 60, 92, 92, 92,92,        #144
28, 64, 92, 88, 88, 88, 88, 88, 88, 88, 88, 88, 120, 120, 120,120,    #160
28, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,92,        #176
28, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,92,        #192
64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,64,        #208
64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,64,        #224
64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]       #240

idCut=[
40, 56, 64, 88, 64, 32, 60, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #0
28, 56, 56, 54, 36, 36, 36, 36, 36, 36, 36, 36, 32, 28, 28, 28,       #16
28, 52, 56, 52, 52, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #32
28, 56, 52, 52, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #48
28, 52, 36, 36, 36, 36, 36, 36, 36, 36, 56, 56, 36, 56, 36, 36,       #64
28, 60, 60, 60, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,       #80
28, 120, 120, 120, 92, 92, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,     #96
28, 120, 152, 128, 100, 124, 80, 80, 64, 100, 64, 64, 64, 64, 64,64,   #112 
28, 88, 92, 64, 92, 60, 60, 60, 92, 60, 60, 60, 60, 60, 60, 60,       #128   
36, 88, 88, 64, 64, 60, 60, 60, 60, 60, 60, 60, 92, 92, 92,92,        #144
28, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,88,    #160
28, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,92,        #176
28, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,92,        #192
64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,64,        #208
64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,64,        #224
64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]       #240


def ID(eta,pt,quality):
     p=pt
     if p>4095:
          p=4095
     p=p/256
     e=int(abs(eta))
     e=e/256
     addr=(e<<4) | p
     if (quality-min(64,pt/32))>=(idCut[addr]):
          return True
     else:
          return False

#verbose=False
#tag='singleMuonOfficial'
#isData=False
#tag='signal'

def strAppend(s):
     return "root://cmsxrootd.fnal.gov/"+s

from dy200 import * 
from jpsi import * 
from minbias import * 
tag='/uscmst1b_scratch/lpc1/3DayLifetime/bachtis/DY200'
#events=Events(['/uscmst1b_scratch/lpc1/3DayLifetime/bachtis/MinBias.root'
#          ])
events=Events(map(strAppend,dy200))

#events=Events(minbias)
#events=Events(['reprocess.root'])

def fetchTracks(event):
     trackHandle = Handle('vector<TTTrack<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>,Phase2TrackerDigi,edm::refhelper::FindForDetSetVector<Phase2TrackerDigi> > > >')
     event.getByLabel("l1tTTTracksFromTrackletEmulation:Level1TTTracks",trackHandle)
     return trackHandle.product()
     

def fetchTPS(event,tag,etamax=3.0):
     phiSeg2    = Handle  ('vector<l1t::TrackerMuon>')
     event.getByLabel(tag,phiSeg2)
     return filter(lambda x: abs(x.eta())<etamax,phiSeg2.product())


def fetchStubs(event,tag):
     phiSeg2    = Handle  ('vector<l1t::MuonStub>')
     event.getByLabel(tag,phiSeg2)
     return phiSeg2.product()


def fetchGEN(event,etaMax=3.0):
     genH  = Handle  ('vector<reco::GenParticle>')
     event.getByLabel('genParticles',genH)
     genMuons=filter(lambda x: abs(x.pdgId())==13 and x.status()==1 and abs(x.eta())<etaMax,genH.product())
     return genMuons


def deltaPhi( p1, p2):
     '''Computes delta phi, handling periodic limit conditions.'''
     res = p1 - p2
     while res > math.pi:
         res -= 2*math.pi
     while res < -math.pi:
         res += 2*math.pi
     return res

def deltaR( *args ):
     return math.sqrt( deltaR2(*args) )

def deltaR2( e1, p1, e2, p2):
     de = e1 - e2
     dp = deltaPhi(p1, p2)
     return de*de + dp*dp




quality      = ROOT.TH3D("quality","",16,0,4096,16,0,4096,128,0,512)
qualityAll   = ROOT.TH3D("qualityAll","",16,0,4096,16,0,4096,128,0,512)
histoData={'effpt':{}, 'effeta':{},'geneta':{},'genphi':{},'effphi':{}}
histoData['genpt'] = ROOT.TH1D("genpt","genpt",50,0,50)

thresholds=[1,3,5,15,20]
for t in thresholds:
     histoData['effpt'][t] = ROOT.TH1D("effpt_"+str(t),"effpt_"+str(t),50,0,50)
     histoData['effeta'][t] = ROOT.TH1D("effeta_"+str(t),"effeta_"+str(t),48,-2.4,2.4)
     histoData['effphi'][t] = ROOT.TH1D("effphi_"+str(t),"effphi_"+str(t),32,-math.pi,math.pi)
     histoData['geneta'][t] = ROOT.TH1D("geneta_"+str(t),"effeta_"+str(t),48,-2.4,2.4)
     histoData['genphi'][t] = ROOT.TH1D("genphi_"+str(t),"genphi_"+str(t),32,-math.pi,math.pi)




histoData['rate']= ROOT.TH1D("rate","rate",50,0,100)
histoData['rate20eta']= ROOT.TH1D("rate20eta","rate",8,0,8)


BUNCHFACTOR=40000*2760.0/3564.0

QUALITYCUT=0
verbose=0
counter=-1
for event in events:
    counter=counter+1
    if counter==100000:
         break;
    gen=fetchGEN(event,2.4)
    stubs = fetchStubs(event,'l1tGMTStubs')
    tps = filter(lambda x: ID(x.hwEta(),x.hwPt(),x.hwQual()),fetchTPS(event,'l1tGMTMuons'))
#    tps = fetchTPS(event,'gmtMuons')
    tracks=fetchTracks(event)

    if verbose:
         for t in sorted(tracks,key=lambda x: x.momentum().transverse(),reverse=True):
              print("Track ",t.eta(),t.phi(),t.momentum().transverse(),t.getStubRefs().size())
         for t in sorted(tps,key=lambda x: x.pt(),reverse=True):
              print("Tracker Muon pt={pt} eta={eta} phi={phi} qual={qual}".format(pt=t.pt(),eta=t.eta(),phi=t.phi(),qual=t.hwQual))

         #         import pdb;pdb.set_trace()
         for s in stubs:
              print(" All Stub depth={depth} eta={eta1},{eta1I},{eta2},{eta2I} phi={phi1},{phi1I},{phi2},{phi2I} qualities={etaQ},{phiQ}".format(depth=s.depthRegion(),eta1=s.offline_eta1(),eta1I=s.eta1(),eta2=s.offline_eta2(),eta2I=s.eta2(),phi1=s.offline_coord1(),phi1I=s.coord1(),phi2=s.offline_coord2(),phi2I=s.coord2(),etaQ=s.etaQuality(),phiQ=s.quality()))


    if len(tps)>0:
        tpsInEta =filter(lambda x: abs(x.eta())<2.4,tps)
        if len(tpsInEta)>0:
            best=max(tpsInEta,key=lambda x: x.pt())
            qualityAll.Fill(abs(best.hwEta()),best.hwPt(),best.hwQual()-min(63,best.hwPt()/32))
            pt=best.pt()
            if pt>99.9:
                pt=99.9
            histoData['rate'].Fill(pt)
            if pt>20:
                 histoData['rate20eta'].Fill(abs(best.hwEta())/512)
#                 import pdb;pdb.set_trace()


    if counter %1000==0:
         print(counter)
    if verbose:
         print ("EVENT:",counter)
    
    #efficiencies -loop on gen
    for g in gen:
        histoData['genpt'].Fill(g.pt())
        for thres in thresholds:
            if g.pt()>(thres+2.0):
                histoData['geneta'][thres].Fill(g.eta())
                histoData['genphi'][thres].Fill(g.phi())

        if verbose:
            print("Gen Muon pt={pt} eta={eta} phi={phi} charge={charge}".format(pt=g.pt(),eta=g.eta(),phi=g.phi(),charge=g.charge()))

        foundMatch=False        
        tpsMatched = sorted(filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3, tps),key=lambda x:x.pt(),reverse=True)
#        if len(tpsMatched)==0 and g.pt()>10:
#             print("Not Matched ",g.pt(),g.eta(),len(tps))
#             import pdb;pdb.set_trace()
        if len(tpsMatched)>0:
            foundMatch=True
          
            if verbose:
                 for t in tpsMatched:
                      print(" -> matched track ->Tracker Muon pt={pt} eta={eta} phi={phi} qual={qual}".format(pt=t.pt(),eta=t.eta(),phi=t.phi(),qual=t.hwQual()))
            best=tpsMatched[0]
            quality.Fill(abs(best.hwEta()),best.hwPt(),best.hwQual()-min(63,best.hwPt()/32))
        for thres in thresholds:
            overThreshold = filter(lambda x: x.pt()>thres,tpsMatched)
            if len(overThreshold)>0:
                histoData['effpt'][thres].Fill(g.pt())
                if g.pt()>(thres+2.0):
                    histoData['effeta'][thres].Fill(g.eta())
                    histoData['effphi'][thres].Fill(g.phi())







        stubsMatched = filter(lambda x: deltaR(g.eta(),g.phi(),x.offline_eta1(),x.offline_coord1())<0.3, stubs)
        if verbose:
             for s in stubsMatched:
                  print(" -> matched stub -> Muon Stub  eta={eta1},{eta2} phi={phi1},{phi2} qualities={etaQ},{phiQ}".format(eta1=s.offline_eta1(),eta2=s.offline_eta2(),phi1=s.offline_coord1(),phi2=s.offline_coord2(),etaQ=s.etaQuality(),phiQ=s.quality()))
             
             

f=ROOT.TFile("tpsResults_output.root","RECREATE")
f.cd()
quality.Write()
qualityAll.Write()

c = histoData['rate'].GetCumulative(False)
c.Scale(float(BUNCHFACTOR)/float(counter))
c.Write("rate")     

for t in thresholds:
    g = ROOT.TGraphAsymmErrors(histoData['effpt'][t],histoData['genpt'])
    g.Write("eff_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effeta'][t],histoData['geneta'][t])
    g.Write("effeta_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effphi'][t],histoData['genphi'][t])
    g.Write("effphi_"+str(t))

f.Close()

