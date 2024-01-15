import ROOT,itertools,math      
from array import array          
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()




def strAppend(s):
     return "root://cmsxrootd.fnal.gov/"+s

def fetchKMTF(event,tag,etamax=3.0):
     phiSeg2    = Handle  ('vector<l1t::SAMuon>')
     event.getByLabel(tag,phiSeg2)
     return list(filter(lambda x: abs(x.eta())<etamax,phiSeg2.product()))

def fetchStubs(event,tag):
     phiSeg2    = Handle  ('vector<l1t::MuonStub>')
     event.getByLabel(tag,phiSeg2)
     return phiSeg2.product()


def fetchGEN(event,etaMax=3.0):
     genH  = Handle  ('vector<reco::GenParticle>')
     event.getByLabel('genParticles',genH)
     genMuons=list(filter(lambda x: abs(x.pdgId())==13 and x.status()==1 and abs(x.eta())<etaMax,genH.product()))
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


histoData={'effpt':{}, 'effeta':{},'geneta':{},'genphi':{},'effphi':{},'effptB':{},'effptO':{},'effptE':{}}
histoData['genpt'] = ROOT.TH1D("genpt","genpt",20,0,100)
histoData['genptB'] = ROOT.TH1D("genptB","genptB",20,0,100)
histoData['genptO'] = ROOT.TH1D("genptO","genptO",20,0,100)
histoData['genptE'] = ROOT.TH1D("genptE","genptE",20,0,100)
rate = ROOT.TH1D("rate","rate",20,0,100)
rateB = ROOT.TH1D("rateB","rateB",20,0,100)
rateO = ROOT.TH1D("rateO","rateO",20,0,100)
rateE = ROOT.TH1D("rateE","rateE",20,0,100)

thresholds=[0,1,3,5,15,20]
for t in thresholds:
     histoData['effpt'][t] = ROOT.TH1D("effpt_"+str(t),"effpt_"+str(t),20,0,100)
     histoData['effptB'][t] = ROOT.TH1D("effptB_"+str(t),"effpt_"+str(t),20,0,100)
     histoData['effptO'][t] = ROOT.TH1D("effptO_"+str(t),"effpt_"+str(t),20,0,100)
     histoData['effptE'][t] = ROOT.TH1D("effptE_"+str(t),"effpt_"+str(t),20,0,100)
     histoData['effeta'][t] = ROOT.TH1D("effeta_"+str(t),"effeta_"+str(t),48,-2.4,2.4)
     histoData['effphi'][t] = ROOT.TH1D("effphi_"+str(t),"effphi_"+str(t),32,-math.pi,math.pi)
     histoData['geneta'][t] = ROOT.TH1D("geneta_"+str(t),"effeta_"+str(t),48,-2.4,2.4)
     histoData['genphi'][t] = ROOT.TH1D("genphi_"+str(t),"genphi_"+str(t),32,-math.pi,math.pi)




etaLUT = ROOT.TH2D("etaLUT","etaLUT",256,0,256,128,0.0,1.0)

from samples200 import *
toProcess = jpsi200

files=[]

for p in toProcess:
     files.append(strAppend(p))

#events=Events(['file:/uscmst1b_scratch/lpc1/3DayLifetime/bachtis/test.root'])
events=Events(files)



displaced=0



BUNCHFACTOR=40000*2760.0/3564.0

counter=0;
for event in events:

    gen=fetchGEN(event,2.4)
    sa=fetchKMTF(event,'l1tSAMuonsGmt:prompt',2.5)
    if displaced==1:
         gen=fetchGEN(event,0.)
         sa=fetchKMTF(event,'l1tKMTFMuonsGmt:displaced',2.5)

    highq=[]
    
    #reject single stub muons
    for s in sa:
         if s.hwQual()>0:
              highq.append(s)
    sa=highq     

    maxPT = None 
    maxPTB = None 
    maxPTO = None 
    maxPTE = None 

    for s in sa:
     if maxPT==None or s.pt()>maxPT:
          maxPT=s.pt()
     if (maxPTB==None or s.pt()>maxPTB) and abs(s.eta())<0.83:
          maxPTB=s.pt()
     if (maxPTO==None or s.pt()>maxPTO) and abs(s.eta())>0.83 and abs(s.eta())<1.2:
          maxPTO=s.pt()
     if (maxPTE==None or s.pt()>maxPTE) and abs(s.eta())>1.2:
          maxPTE=s.pt()
    if maxPT!=None:
         rate.Fill(maxPT)
    if maxPTB!=None:
         rateB.Fill(maxPTB)
    if maxPTO!=None:
         rateO.Fill(maxPTO)
    if maxPTE!=None:
         rateE.Fill(maxPTE)


    
#    print('---------------------NEW EVENT---------------------')
    if counter %10000 ==0:
         print(counter)

#    if counter==150000:
#         break;
#    print(counter)
#    print('Generated muons')
    for m in gen:

#         print("gen pt=",m.pt())
         histoData['genpt'].Fill(m.pt())
         if abs(m.eta())<0.83:
              histoData['genptB'].Fill(m.pt())
         elif abs(m.eta())>0.83 and abs(m.eta())<1.2:
              histoData['genptO'].Fill(m.pt())
         else:     
              histoData['genptE'].Fill(m.pt())
         for t in thresholds:
              if m.pt()>(float(t)+10.0):
                   histoData['geneta'][t].Fill(m.eta())
                   histoData['genphi'][t].Fill(m.phi())
         #fill the etaLUT      
         for r in sa:
               if abs(deltaPhi(r.phi(),m.phi()))<0.3 and r.pt()>float(t) and r.hwQual()>0:
                    code=0;
                    for s in r.stubs():
                         code = code| ( (int(abs(s.etaRegion())+1))<<(2*(s.depthRegion()-1)))
                    etaLUT.Fill(code,abs(m.eta()))

         for t in thresholds:
#              print("Threshold {}".format(t))
#              for s in sa:
#                   print("muon pt={}".format(s.pt()))
              matched=False
              for r in sa:
#                   print("SA over muon pt={}".format(r.pt()))
                   
                   if deltaR(r.eta(),r.phi(),m.eta(),m.phi())<0.5 and r.pt()>float(t) and r.hwQual()>0:
                        matched=True
                        break
#              print("matched={}".format(matched)) 
              if matched: 
                   histoData['effpt'][t].Fill(m.pt())
                   if abs(m.eta())<0.83:
                        histoData['effptB'][t].Fill(m.pt())
                   elif abs(m.eta())>0.83 and abs(m.eta())<1.2:
                        histoData['effptO'][t].Fill(m.pt())
                   else:     
                        histoData['effptE'][t].Fill(m.pt())
                   if m.pt()>(t+10):
                        histoData['effeta'][t].Fill(m.eta())
                        histoData['effphi'][t].Fill(m.phi())

    counter=counter+1

f=ROOT.TFile("saStudy_results.root","RECREATE")
f.cd()
#c = histoData['rate'].GetCumulative(False)
#c.Scale(float(BUNCHFACTOR)/float(counter))
#c.Write("rate")     
histoData['genpt'].Write("genpt")
histoData['genptB'].Write("genptB")
histoData['genptO'].Write("genptO")
histoData['genptE'].Write("genptE")
etaLUT.Write()
rate.Write()
rateB.Write()
rateE.Write()
rateO.Write()

c = rate.GetCumulative(False)
c.Scale(float(BUNCHFACTOR)/float(counter))
c.Write("rate")     
c = rateB.GetCumulative(False)
c.Scale(float(BUNCHFACTOR)/float(counter))
c.Write("rateBarrel")     
c = rateO.GetCumulative(False)
c.Scale(float(BUNCHFACTOR)/float(counter))
c.Write("rateOverlap")     
c = rateE.GetCumulative(False)
c.Scale(float(BUNCHFACTOR)/float(counter))
c.Write("rateEndcap")     


for t in thresholds:
    histoData['effpt'][t].Write("numpt_"+str(t))
    histoData['effeta'][t].Write("numeta_"+str(t))
    histoData['effphi'][t].Write("numphi_"+str(t))
    histoData['geneta'][t].Write("geneta_"+str(t))
    histoData['genphi'][t].Write("genphi_"+str(t))
 
    g = ROOT.TGraphAsymmErrors(histoData['effpt'][t],histoData['genpt'])
    g.Write("eff_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effptB'][t],histoData['genptB'])
    g.Write("effB_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effptO'][t],histoData['genptO'])
    g.Write("effO_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effptE'][t],histoData['genptE'])
    g.Write("effE_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effeta'][t],histoData['geneta'][t])
    g.Write("effeta_"+str(t))
    g = ROOT.TGraphAsymmErrors(histoData['effphi'][t],histoData['genphi'][t])
    g.Write("effphi_"+str(t))

f.Close()
