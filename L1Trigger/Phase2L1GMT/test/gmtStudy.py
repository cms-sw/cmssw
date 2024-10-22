import ROOT,itertools,math      
import argparse
from array import array          
from DataFormats.FWLite import Events, Handle
import subprocess
ROOT.FWLiteEnabler.enable()


class muon_trigger_analyzer(object):
     def __init__(self,prefix,thresholds=[0,3,5,10,15,20,30]):
          self.bunchfactor = 40000*2760.0/3564.0
          self.prefix=prefix
          self.thresholds = thresholds
          self.histoData={'effpt':{}, 'effeta':{},'geneta':{},'genphi':{},'genlxy':{},'efflxy':{},'effphi':{},'effptB':{},'effptO':{},'effptE':{},'rateeta':{}}
          self.histoData['genpt']         = ROOT.TH1D(prefix+"_genpt","genpt",50,0,100)
          self.histoData['genptB']        = ROOT.TH1D(prefix+"_genptB","genptB",50,0,100)
          self.histoData['genptO']        = ROOT.TH1D(prefix+"_genptO","genptO",50,0,100)
          self.histoData['genptE']        = ROOT.TH1D(prefix+"_genptE","genptE",50,0,100)
          self.histoData['rate']          = ROOT.TH1D(prefix+"_rate","rate",20,0,100)
          self.histoData['resolution']    = ROOT.TH2D(prefix+"_resolution","resolution",20,0,100,100,-2,2)
          self.histoData['rateBarrel']    = ROOT.TH1D(prefix+"_rateBarrel","rate",20,0,100)
          self.histoData['rateOverlap']   = ROOT.TH1D(prefix+"_rateOverlap","rate",20,0,100)
          self.histoData['rateEndcap']    = ROOT.TH1D(prefix+"_rateEndcap","rate",20,0,100)
          for t in thresholds:
               self.histoData['effpt'][t] = ROOT.TH1D(prefix+"_effpt_"+str(t),"effpt_"+str(t),50,0,100)
               self.histoData['effptB'][t] = ROOT.TH1D(prefix+"_effptB_"+str(t),"effpt_"+str(t),50,0,100)
               self.histoData['effptO'][t] = ROOT.TH1D(prefix+"_effptO_"+str(t),"effpt_"+str(t),50,0,100)
               self.histoData['effptE'][t] = ROOT.TH1D(prefix+"_effptE_"+str(t),"effpt_"+str(t),50,0,100)
               self.histoData['effeta'][t] = ROOT.TH1D(prefix+"_effeta_"+str(t),"effeta_"+str(t),48,-2.4,2.4)
               self.histoData['effphi'][t] = ROOT.TH1D(prefix+"_effphi_"+str(t),"effphi_"+str(t),32,-math.pi,math.pi)
               self.histoData['efflxy'][t] = ROOT.TH1D(prefix+"_efflxy_"+str(t),"efflxy_"+str(t),50,0,200)
               self.histoData['geneta'][t] = ROOT.TH1D(prefix+"_geneta_"+str(t),"effeta_"+str(t),48,-2.4,2.4)
               self.histoData['genphi'][t] = ROOT.TH1D(prefix+"_genphi_"+str(t),"genphi_"+str(t),32,-math.pi,math.pi)
               self.histoData['genlxy'][t] = ROOT.TH1D(prefix+"_genlxy_"+str(t),"genlxy_"+str(t),50,0,200)
               self.histoData['rateeta'][t] = ROOT.TH1D(prefix+"_rateeta_"+str(t),"rateeta_"+str(t),24,-2.4,2.4)

     def deltaPhi(self, p1, p2):
          '''Computes delta phi, handling periodic limit conditions.'''
          res = p1 - p2
          while res > math.pi:
               res -= 2*math.pi
          while res < -math.pi:
               res += 2*math.pi
          return res

     def deltaR(self, *args ):
          return math.sqrt( self.deltaR2(*args) )

     def deltaR2(self, e1, p1, e2, p2):
          de = e1 - e2
          dp = self.deltaPhi(p1, p2)
          return de*de + dp*dp
     def getLxy(self,muon):
          return abs(math.sqrt(muon.vx()*muon.vx()+muon.vy()*muon.vy()))

     def getDxy(self,muon):
          tanphi = math.tan(m.phi())
          x=(tanphi*tanphi*muon.vx()-muon.vy()*tanphi)/(1+tanphi*tanphi)
          y=muon.vy()+tanphi*(x-muon.vx())
          return abs(math.sqrt(x*x+y*y))

     def getEta1(self,muon):
          lxy = self.getLxy(muon)
          vz = muon.vz()
          theta1 = math.atan((700.-lxy)/(650.-vz))
          
          if (theta1 < 0):
               theta1 = math.pi+theta1
          eta1 = -math.log(math.tan(theta1/2.0))
          return eta1

     def getEta2(self,muon):
          lxy = self.getLxy(muon)
          vz = muon.vz()
          theta2 = math.pi-math.atan((700.-lxy)/(650.+vz))
          if theta2 > math.pi:
               theta2 = theta2-math.pi
          eta2 = -math.log(math.tan(theta2/2.0))
          return eta2                                     

     def getAcceptance(self,muon):
          eta1 = self.getEta1(muon)
          eta2 = self.getEta2(muon)
          if muon.eta() < eta1 and muon.eta() > eta2:
               return True #Muon is within barrel acceptance
          else:
               return False #Muon is outside of barrel 

     def getSt2Eta(self,muon):
          lxy = self.getLxy(muon)
          vz = muon.vz()
          theta_mu = 2*math.atan(math.exp(-muon.eta()))
          st1_z = (512.-lxy)/math.tan(theta_mu)+vz
          st1_r = 512.
          theta_st1 = math.atan2(st1_r,st1_z)
          eta_st1 = -math.log(math.tan(theta_st1/2.))
          return eta_st1

     def getSt2Phi(self,muon):  
          # calculate intersection of line and circle
          x1 = muon.vx()
          y1 = muon.vy()
          x2 = muon.vx() + muon.px()/(muon.px()**2+muon.py()**2)
          y2 = muon.vy() + muon.py()/(muon.px()**2+muon.py()**2)
          r = 512.
          dx = x2-x1
          dy = y2-y1
          dr = math.sqrt(dx**2+dy**2)
          D = x1*y2-x2*y1
          delta = (r**2)*(dr**2)-D**2
          if delta < 0:
               return math.atan2(y1, x1)
          # Two possible intersections = two possible phi values
          xP = (D*dy+math.copysign(1,dy)*dx*math.sqrt(delta))/dr**2
          xM = (D*dy-math.copysign(1,dy)*dx*math.sqrt(delta))/dr**2
          yP = (-D*dx+abs(dy)*math.sqrt(delta))/dr**2
          yM = (-D*dx-abs(dy)*math.sqrt(delta))/dr**2
          
          p1 = (xP, yP)
          p2 = (xM, yM)
          phi1 = math.atan2(yP,xP)
          phi2 = math.atan2(yM,xM)
          
          phi = min([phi1, phi2], key = lambda x: abs(self.deltaPhi(x, muon.phi()))) #probably a better way to select which intersection
          return phi   


     def process(self,gen,l1,dr=0.3,verbose=0):
          #first efficiency
          for g in gen:
               if verbose:
                    print("Gen Muon pt={pt} eta={eta} phi={phi} vxy={dxy}".format(pt=g.pt(),eta=g.eta(),phi=g.phi(),dxy=math.sqrt(g.vx()*g.vx()+g.vy()*g.vy())))
               self.histoData['genpt'].Fill(g.pt())
               if abs(g.eta())<0.83:
                    self.histoData['genptB'].Fill(g.pt())
               elif abs(g.eta())>0.83 and abs(g.eta())<1.2:      
                    self.histoData['genptO'].Fill(g.pt())
               else:
                    self.histoData['genptE'].Fill(g.pt())
               #Now let's process every threshold
               for t in self.thresholds:
                    if g.pt()>(t+10):
                         self.histoData['geneta'][t].Fill(g.eta())
                         self.histoData['genphi'][t].Fill(g.phi())
                         if self.getAcceptance(g):
                              self.histoData['genlxy'][t].Fill(self.getLxy(g))
                              
                    matched=[]
                    matchedDisplaced=[]
                    for mu in l1:                         
                         if mu.pt()<float(t):
                              continue
                         if(self.deltaR(g.eta(),g.phi(),mu.eta(),mu.phi()))<dr:
                            matched.append(mu)
                         if(self.deltaR(self.getSt2Eta(g),self.getSt2Phi(g),mu.eta(),mu.phi()))<dr:
                            matchedDisplaced.append(mu)

#                    if len(matched)==0 and g.pt()>20 and self.prefix=='tk' and t==15:
#                         import pdb;pdb.set_trace()
                    if len(matched)>0:
                         self.histoData['effpt'][t].Fill(g.pt())
                         if abs(g.eta())<0.83:
                              self.histoData['effptB'][t].Fill(g.pt())
                         elif abs(g.eta())>0.83 and abs(g.eta())<1.2:      
                              self.histoData['effptO'][t].Fill(g.pt())
                         else:
                              self.histoData['effptE'][t].Fill(g.pt())
                         if g.pt()>(t+10):
                              self.histoData['effeta'][t].Fill(g.eta())
                              self.histoData['effphi'][t].Fill(g.phi())
                         deltaPt=10000
                         best=None
                         for match in matched:
                              delta=abs(match.pt()-g.pt())
                              if delta<deltaPt:
                                   deltaPt=delta
                                   best=match
                         if deltaPt<10000:
                              self.histoData['resolution'].Fill(g.pt(),(best.pt()-g.pt())/g.pt())

                    if len(matchedDisplaced)>0:
                         if g.pt()>(t+10) and self.getAcceptance(g):
                              self.histoData['efflxy'][t].Fill(self.getLxy(g))


          #now rate
          maxElement=None
          maxPt=0
          for l in l1:
               if verbose:
                    print("{prefix} Muon pt={pt} eta={eta} phi={phi} stubs={stubs}".format(prefix=self.prefix,pt=l.pt(),eta=l.eta(),phi=l.phi(),stubs=len(l.stubs())))
                    for s in l.stubs():
                         print("-----> Associated Stub etaR={eta} phiR={phi} depthR={depth} coord1={coord1} coord2={coord2} q={q} ".format(eta=s.etaRegion(),phi=s.phiRegion(),depth=s.depthRegion(),coord1=s.offline_coord1(),coord2=s.offline_coord2(),q=s.quality()))
               if l.pt()>maxPt:
                    maxPt=l.pt()
                    maxElement=l
          if maxElement!=None:
               self.histoData['rate'].Fill(maxPt)
               if abs(maxElement.eta())<0.83:
                    self.histoData['rateBarrel'].Fill(maxPt)
               elif abs(maxElement.eta())>0.83 and abs(maxElement.eta())<1.2:
                    self.histoData['rateOverlap'].Fill(maxPt)
               else:
                    self.histoData['rateEndcap'].Fill(maxPt)
               for t in self.thresholds:
                    if maxPt>t:
                         self.histoData['rateeta'][t].Fill(maxElement.eta())                    

     def write(self,f):
          f.cd()
          self.histoData['genpt'].Write()
          self.histoData['genptB'].Write()
          self.histoData['genptO'].Write()
          self.histoData['genptE'].Write()
          self.histoData['resolution'].Write()
          c =self.histoData['rate'].GetCumulative(False)
          c.Scale(float(self.bunchfactor)/float(counter))
          c.Write(self.prefix+"_rate")     
          c =self.histoData['rateBarrel'].GetCumulative(False)
          c.Scale(float(self.bunchfactor)/float(counter))
          c.Write(self.prefix+"_rateBarrel")     
          c =self.histoData['rateOverlap'].GetCumulative(False)
          c.Scale(float(self.bunchfactor)/float(counter))
          c.Write(self.prefix+"_rateOverlap")     
          c =self.histoData['rateEndcap'].GetCumulative(False)
          c.Scale(float(self.bunchfactor)/float(counter))
          c.Write(self.prefix+"_rateEndcap")     
          for t in self.thresholds:
               c =self.histoData['rateeta'][t]
               c.Scale(float(self.bunchfactor)/float(counter))
               c.Write(self.prefix+"_rateeta_"+str(t))     
               g = ROOT.TGraphAsymmErrors(self.histoData['effpt'][t],self.histoData['genpt'])
               g.Write(self.prefix+"_eff_"+str(t))
               g = ROOT.TGraphAsymmErrors(self.histoData['effptB'][t],self.histoData['genptB'])
               g.Write(self.prefix+"_effB_"+str(t))
               g = ROOT.TGraphAsymmErrors(self.histoData['effptO'][t],self.histoData['genptO'])
               g.Write(self.prefix+"_effO_"+str(t))
               g = ROOT.TGraphAsymmErrors(self.histoData['effptE'][t],self.histoData['genptE'])
               g.Write(self.prefix+"_effE_"+str(t))
               g = ROOT.TGraphAsymmErrors(self.histoData['effeta'][t],self.histoData['geneta'][t])
               g.Write(self.prefix+"_effeta_"+str(t))
               g = ROOT.TGraphAsymmErrors(self.histoData['effphi'][t],self.histoData['genphi'][t])
               g.Write(self.prefix+"_effphi_"+str(t))
               g = ROOT.TGraphAsymmErrors(self.histoData['efflxy'][t],self.histoData['genlxy'][t])
               g.Write(self.prefix+"_efflxy_"+str(t))
               
          


#HELPERS          
def strAppend(s):
     return "root://cmsxrootd.fnal.gov/"+s

def fetchSTA(event,tag,etamax=3.0):
     phiSeg2    = Handle  ('vector<l1t::SAMuon>')
     event.getByLabel(tag,phiSeg2)
     return list(filter(lambda x: abs(x.eta())<etamax,phiSeg2.product()))
def fetchTPS(event,tag,etamax=3.0):
     phiSeg2    = Handle  ('vector<l1t::TrackerMuon>')
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

def EOSls(path):
    print('eos root://cmseos.fnal.gov/ find -name "*root" %s' % path )
    p = subprocess.Popen('eos root://cmseos.fnal.gov/ find -name "*root" %s' % path, 
                         stdout = subprocess.PIPE, stderr = subprocess.PIPE,
                         shell=True)
    stdout, stderr = p.communicate()
    out = ["root://cmseos.fnal.gov/"+i.decode('UTF-8') for i in stdout.split()]
    return out
#DATASET
# from samples200 import *
# toProcess = dy200
# files=[]
# for p in toProcess:
     # files.append(strAppend(p))
# events=Events(files)
# tag='DY_v1'
# tag='disp200'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--tag', default="Tau", help="fdf")
parser.add_argument('--prod', default="gmtMuons", help="fdf")
args = parser.parse_args()
tag = args.tag
samples = {
    "MB_org" : "/eos/uscms//store/group/lpctrig/benwu/GMT_Ntupler/Spring22_GMToriginal_v2/MinBias_TuneCP5_14TeV-pythia8/PHASEII_MinBias/",
    "DY_org" : "/eos/uscms//store/group/lpctrig/benwu/GMT_Ntupler/Spring22_GMToriginal_v2/DYToLL_M-50_TuneCP5_14TeV-pythia8/PHASEII_DYToLL/",
    "MB_v1" : "/eos/uscms//store/group/lpctrig/benwu/GMT_Ntupler/Spring22_GMT_v3/MinBias_TuneCP5_14TeV-pythia8/PHASEII_MinBias/",
    "DY_v1" : "/eos/uscms//store/group/lpctrig/benwu/GMT_Ntupler/Spring22_GMT_v3/DYToLL_M-50_TuneCP5_14TeV-pythia8/PHASEII_DYToLL/",
}
flist = EOSls(samples[tag])
events= Events(flist)
# events=Events(['file:/uscmst1b_scratch/lpc1/3DayLifetime/bachtis/gmt.root'])
#events=Events(['file:gmt.root'])
# events=Events(['file:reprocess.root'])


#ANALYSIS SETUP
verbose=0
saAnalyzer = muon_trigger_analyzer('sta_prompt')
kmtfAnalyzer_p = muon_trigger_analyzer('kmtf_prompt')
kmtfAnalyzer_d = muon_trigger_analyzer('kmtf_disp')
tkAnalyzer_1st = muon_trigger_analyzer('tk_1st')
tkAnalyzer_2st = muon_trigger_analyzer('tk_2st')
tkAnalyzer_2m = muon_trigger_analyzer('tk_2m')
tkAnalyzer_3m = muon_trigger_analyzer('tk_3m')
tkAnalyzer_4m = muon_trigger_analyzer('tk_4m')

#EVENT LOOP

counter=0;
for event in events:
     if verbose:
          print('EVENT {}'.format(counter))
     gen=fetchGEN(event,2.4)
     genKMTF=fetchGEN(event,0.83)    
     sa=fetchSTA(event,'gmtSAMuons:prompt',2.5)
     kmtf_p=fetchSTA(event,'gmtKMTFMuons:prompt',2.5)
     kmtf_d=fetchSTA(event,'gmtKMTFMuons:displaced',2.5)
     tps=fetchTPS(event,'gmtTkMuons',2.5)
     # sa=fetchSTA(event,'l1tSAMuonsGmt:prompt',2.5)
     # kmtf_p=fetchSTA(event,'l1tKMTFMuonsGmt:prompt',2.5)
     # kmtf_d=fetchSTA(event,'l1tKMTFMuonsGmt:displaced',2.5)
     # tps=fetchTPS(event,'l1tTkMuonsGmt',2.5)
     tps_2st=list(filter(lambda x: x.stubs().size()>1,tps))
     tps_2m=list(filter(lambda x: x.numberOfMatches()>1,tps))
     tps_3m=list(filter(lambda x: x.numberOfMatches()>2,tps))
     tps_4m=list(filter(lambda x: x.numberOfMatches()>3,tps))
     #analyze
     saAnalyzer.process(gen,sa,0.6,verbose)
     kmtfAnalyzer_p.process(genKMTF,kmtf_p,0.6,verbose)
     kmtfAnalyzer_d.process(genKMTF,kmtf_d,0.6,verbose)
     tkAnalyzer_1st.process(gen,tps,0.2,verbose)
     tkAnalyzer_2st.process(gen,tps_2st,0.2,0)
     tkAnalyzer_2m.process(gen,tps_2m,0.2,0)
     tkAnalyzer_3m.process(gen,tps_3m,0.2,0)
     tkAnalyzer_4m.process(gen,tps_4m,0.2,0)



#     if verbose==1 and counter==69:
#          import pdb;pdb.set_trace()
     #counter 
#     if counter==100000:
#          break

     if (counter %10000 ==0):
          print(counter)
     counter=counter+1

f=ROOT.TFile("gmtAnalysis_{tag}.root".format(tag=tag),"RECREATE")
f.cd()
saAnalyzer.write(f)
kmtfAnalyzer_p.write(f)
kmtfAnalyzer_d.write(f)
tkAnalyzer_1st.write(f)
tkAnalyzer_2st.write(f)
tkAnalyzer_2m.write(f)
tkAnalyzer_3m.write(f)
tkAnalyzer_4m.write(f)
f.Close()
