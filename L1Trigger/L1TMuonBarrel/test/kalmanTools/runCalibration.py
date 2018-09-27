import ROOT
import itertools
import math
from DataFormats.FWLite import Events, Handle
from array import array

import numpy
def median(lst):
    return numpy.median(numpy.array(lst))




def fetchSegmentsPhi(event,ontime=True,twinMux=True):
    phiSeg    = Handle  ('L1MuDTChambPhContainer')
    if twinMux:
        event.getByLabel('simTwinMuxDigis',phiSeg)
    else:
        event.getByLabel('simDtTriggerPrimitiveDigis',phiSeg)
    if ontime:
        filtered=filter(lambda x: x.bxNum()==0, phiSeg.product().getContainer())
        return filtered
    else:
        return phiSeg.product().getContainer()

def fetchSegmentsEta(event,ontime=True):
    thetaSeg  = Handle  ('L1MuDTChambThContainer')
    event.getByLabel('dtTriggerPrimitiveDigis',thetaSeg)
    if ontime:
        filtered=filter(lambda x: x.bxNum()==0, thetaSeg.product().getContainer())
        return filtered
    else:
        return thetaSeg.product().getContainer()    

def fetchGEANT(event):
    geantH  = Handle  ('vector<PSimHit>')
    event.getByLabel('g4SimHits:MuonDTHits',geantH)
    geant=filter(lambda x: x.pabs()>0.5 and abs(x.particleType())==13,geantH.product())
    return geant

def fetchGEN(event,etaMax=1.2):
    genH  = Handle  ('vector<reco::GenParticle>')
    event.getByLabel('genParticles',genH)
    genMuons=filter(lambda x: abs(x.pdgId())==13 and x.status()==1 and abs(x.eta())<etaMax,genH.product())
    return genMuons

def segINT(seg,f1=1,f2=1):
    return seg.phi()*f1,seg.phiB()*f2


def qPTInt(qPT,bits=14):
    lsb = lsBIT(bits)
    floatbinary = int(math.floor(abs(qPT)/lsb))
    return int((qPT/abs(qPT))*floatbinary)

def lsBIT(bits=14):
    maximum=1.25
    lsb = 1.25/pow(2,bits-1)
    return lsb


def getTrueCurvature(muon,geant,segments):
    thisMuonGEANT = filter(lambda x: (muon.charge()>0 and x.particleType()==13) or ((muon.charge()<0) and x.particleType()==-13),geant)
    energyInfo={1:[], 2:[],3:[],4:[]}
    qInfo={1:0.0, 2:0.0,3:0.0,4:0.0}
    qInfoINT={1:0, 2:0,3:0,4:0}
    for p in thisMuonGEANT:
        detid=ROOT.DTChamberId(p.detUnitId())
        station = detid.station()
        for s in segments:
            if s.stNum()==detid.station() and s.whNum()==detid.wheel() and s.scNum()==detid.sector()-1:
                energyInfo[station].append(p.pabs()*muon.pt()/muon.energy())
                break;

            
    for s in [1,2,3,4]:
        if len(energyInfo[s])==0:
            continue
        p = median(energyInfo[s])
        qInfo[s]=muon.charge()/p 
        qInfoINT[s] = qPTInt(qInfo[s])
    return qInfo,qInfoINT    

def matchTrack(muon,segments,geant):
    thisMuonGEANT = filter(lambda x: (muon.charge()>0 and x.particleType()==13) or ((muon.charge()<0) and x.particleType()==-13),geant)
    chambers=[]
    for p in thisMuonGEANT:        
        detid=ROOT.DTChamberId(p.detUnitId())
        chambers.append(p.detUnitId())
        
    chambers=list(set(chambers))


    assocSeg=[]   
    for s in segments:
        for c in chambers:
            detid=ROOT.DTChamberId(c)
            if s.whNum()==detid.wheel() and s.stNum()==detid.station() and s.scNum()==detid.sector()-1:
                if not (s in assocSeg):
                    assocSeg.append(s)

    return assocSeg




events = Events([
    'file:singleMuonOfficial.root',
]
)




stations=[1,2,3,4]
PHISCALE=pow(2,11)
PHIBSCALE=pow(2,9)
PHIFACTOR = 1
PHIBFACTOR =8
RELFACTOR = 1


DROR = {4:0.147*RELFACTOR,3:0.173*RELFACTOR,2:0.154*RELFACTOR}
DRORB = {4:(1+0.147),3:(1+0.173),2:(1+0.154)}
alpha = {4:-0.0523,3:-0.0793,2:-0.0619}
beta = {4:0.069,3:0.079,2:0.055}


DRORCHI = {4: (726.-433.)/433. ,
           3: (619.-433.)/433. ,
           2: (512.-433.)/433.}


binsk = 200
maxk=8192


histos={}

offset={1:0.156,2:0.138,3:0.775,4:0.0}
offsetINV={1:0.207,2:0.,3:0.,4:0.0}

histos['phiProp']={}
histos['phiPropChi']={}
histos['phiBProp']={}
histos['curvFromPhiB']={}
histos['curvFromDPhi']={}
histos['phiBFromCurv']={}
histos['phiPropChiV']={}
histos['deltaPhiVsPhiB']={}


for i,j in itertools.permutations([1,2,3,4],2):
    if not (i in histos['deltaPhiVsPhiB'].keys()):
        histos['deltaPhiVsPhiB'][i]={}
    histos['deltaPhiVsPhiB'][i][j]=ROOT.TH2D("deltaPhiVsPhiB_"+str(i)+"_"+str(j),"",256,-511,512,2048,-2047,2048)

    if not (i in histos['curvFromDPhi'].keys()):
        histos['curvFromDPhi'][i]={}
    histos['curvFromDPhi'][i][j]=ROOT.TH2D("curvFromDPhi_"+str(i)+"_"+str(j),"",512,-2047,2048,1024,-8192,8192)
    
    
for s in [1,2,3,4]:
    histos['curvFromPhiB'][s]=ROOT.TH2D("curvFromPhiB_"+str(s),"",1024,-512,511,4*400,-8*400,8*400)
    histos['phiBFromCurv'][s]=ROOT.TH2D("phiBFromCurv_"+str(s),"",256,-512,511,256,-511,512)
    histos['phiProp'][s]=ROOT.TH2D("phiProp_"+str(s),"",binsk,-maxk,maxk,50,-200,200)
    histos['phiPropChiV'][s]=ROOT.TH2D("phiPropChiV_"+str(s),"",binsk,-maxk,maxk,50,-200,200)
    histos['phiPropChi'][s]=ROOT.TH2D("phiPropChi_"+str(s),"",binsk,-3000,3000,50,-200,200)
    histos['phiBProp'][s]=ROOT.TH2D("phiBProp_"+str(s),"",binsk,-maxk,maxk,100,-2000,2000)


    
N=0
for event in events:
    N=N+1
    if N==100000:
        break;
    genMuons=fetchGEN(event)
    segments=fetchSegmentsPhi(event)
    segmentsTheta=fetchSegmentsEta(event)
    geant=fetchGEANT(event)
    segmentsTheta=sorted(segmentsTheta,key=lambda x: x.stNum())

    
    for g in genMuons:
        trueK,trueKINT = getTrueCurvature(g,geant,segments)
        cotTheta = int(g.eta()/0.010875)
        segTheta=matchTrack(g,segmentsTheta,geant)
        seg=matchTrack(g,segments,geant)


        for s in seg:
            phi,phiB=segINT(s,PHIFACTOR,PHIBFACTOR)
            histos['curvFromPhiB'][s.stNum()].Fill(s.phiB(),trueKINT[s.stNum()])
#            histos['phiBFromCurv'][s.stNum()].Fill(trueKINT[s.stNum()]>>4,phiB)
            histos['phiBFromCurv'][s.stNum()].Fill(qPTInt(g.charge()/g.pt())>>4,s.phiB())
                
        for s1,s2 in itertools.permutations(seg,2):
            phi1,phiB1=segINT(s1,PHIFACTOR,PHIBFACTOR)
            phi2,phiB2 = segINT(s2,PHIFACTOR,PHIBFACTOR)

            if (s2.scNum()==s1.scNum()+1) or (s1.scNum()==11 and s2.scNum()==0) :
                phi2=phi2+2144
            if (s2.scNum()==s1.scNum()-1) or (s1.scNum()==0 and s2.scNum()==11) :
                phi2=phi2-2144
                
            
            
            if s1.code()>4 and (s1.stNum()!=s2.stNum()):
                histos['deltaPhiVsPhiB'][s1.stNum()][s2.stNum()].Fill(s1.phiB(),phi2-phi1)
                histos['curvFromDPhi'][s1.stNum()][s2.stNum()].Fill(phi2-phi1,qPTInt(g.charge()/g.pt()))

            if s1.stNum()+1==s2.stNum():                
                if s2.scNum()==s1.scNum()+1 or (s2.scNum()==0 and s1.scNum()==11):
                    phi2=phi2+2144
                if s1.scNum()==s2.scNum()+1 or (s2.scNum()==11 and s1.scNum()==0):
                    phi2=phi2-2144
                st=s2.stNum()    
                qPT=trueKINT[st]
                propPhi = phi2-phiB2*DROR[st]+alpha[st]*qPT    
                propPhiB =DRORB[st]*phiB2+beta[st]*qPT 
                
                histos['phiProp'][s2.stNum()].Fill(trueKINT[s2.stNum()],(phi1-phi2)+DROR[s2.stNum()]*phiB2)
                histos['phiBProp'][s2.stNum()].Fill(trueKINT[s2.stNum()],phiB1-DRORB[s2.stNum()]*phiB2)


                # for chi 2 lookmonly from station 1 -> 2,3,4
            if s1.stNum()==1 and s2.stNum()!=1: 
                histos['phiPropChi'][s2.stNum()].Fill(trueKINT[s1.stNum()],(phi2-phi1)+DRORCHI[s2.stNum()]*phiB1)
                histos['phiPropChiV'][s2.stNum()].Fill(qPTInt(g.charge()/g.pt()),(phi2-phi1))
                
                
f=ROOT.TFile("calibrationConstants.root","RECREATE")
for s in [4,3,2,1]:
    histos['phiProp'][s].Write()
    histos['phiPropChi'][s].Write()
    histos['phiPropChiV'][s].Write()

    histos['phiBProp'][s].Write()
    histos['curvFromPhiB'][s].Write()
    histos['phiBFromCurv'][s].Write()


for i,j in itertools.permutations([1,2,3,4],2):
    histos['deltaPhiVsPhiB'][i][j].Write()
    histos['curvFromDPhi'][i][j].Write()

    
f.Close()
