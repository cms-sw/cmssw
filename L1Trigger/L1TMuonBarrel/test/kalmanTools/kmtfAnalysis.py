<<<<<<< HEAD
from __future__ import print_function
=======
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
import ROOT,itertools,math      #
from array import array         # 
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()
# 



<<<<<<< HEAD
verbose=False
=======
#verbose=False
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
tag='singleMuonOfficial'
isData=False

#tag='zerobias'
#tag='zskim'
#isData=True






##A class to keep BMTF data
class BMTFMuon:
    def __init__(self,mu,pt,eta,phi):
        self.muon=mu
        self.p4 = ROOT.reco.Candidate.PolarLorentzVector(pt,eta,phi,0.105)
    
    def quality(self):
        return self.muon.hwQual()

<<<<<<< HEAD
    def hasFineEta(self):
        return self.muon.hwHF()


    def ptUnconstrained(self):
        return self.muon.hwPt2()


=======
    def phiINT(self):
        return self.muon.hwPhi()

    def processor(self):
        return self.muon.processor()

    def hasFineEta(self):
        return self.muon.hwHF()

    def ptUnconstrained(self):
        return self.muon.hwPt2()

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
    def dxy(self):
        return self.muon.hwDXY()

    def charge(self):
        if self.muon.hwSign()>0:
            return -1
        else:
            return +1 

<<<<<<< HEAD

=======
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
    def __getattr__(self, name):
        return getattr(self.p4,name)


###Common methods############




def fetchTP(event,etaMax=0.83):
    tH  = Handle  ('trigger::TriggerEvent')
    mH  = Handle  ('std::vector<reco::Muon>')


    event.getByLabel('hltTriggerSummaryAOD','','HLT',tH)
    event.getByLabel('muons',mH)

<<<<<<< HEAD
    muons=filter( lambda x: x.passed(ROOT.reco.Muon.CutBasedIdMediumPrompt) and x.numberOfMatches()>1 and x.pt()>10.0 and abs(x.eta())<2.4 and x.isolationR03().sumPt/x.pt()<0.1,mH.product())
=======
    muons=filter( lambda x: x.passed(ROOT.reco.Muon.CutBasedIdMediumPrompt) and x.numberOfMatches()>1 and x.pt()>10.0 and abs(x.eta())<2.4 and x.isolationR03().sumPt/x.pt()<0.15,mH.product())
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
    if len(muons)<2:
        return []
    trigger=tH.product()
#    for f in range(0,trigger.sizeFilters()):
<<<<<<< HEAD
#        print(f,trigger.filterLabel(f))
=======
#        print f,trigger.filterLabel(f)
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
#    import pdb;pdb.set_trace()    
    obj = trigger.getObjects()
    index = trigger.filterIndex(ROOT.edm.InputTag("hltL3fL1sMu22Or25L1f0L2f10QL3Filtered27Q::HLT"))
    if index==trigger.sizeFilters():
        return []
    keys = trigger.filterKeys(index)
    hlt=[]
    for key in keys:
        hlt.append(obj[key])
    if len(hlt)==0:
        return []

<<<<<<< HEAD
    triggered=hlt[0]
    tag=None
    probe=None

    tags = filter(lambda x: deltaR(x.eta(),x.phi(),triggered.eta(),triggered.phi())<0.3 and x.pt()>25.0,muons)
    if len(tags)==0:
        return []
    tag = min(muons,key=lambda x: deltaR(x.eta(),x.phi(),triggered.eta(),triggered.phi()))
    muons.remove(tag)
        
    for mu in muons:
        if deltaR(mu.eta(),mu.phi(),tag.eta(),tag.phi())>1.0:
            if deltaR(mu.eta(),mu.phi(),triggered.eta(),triggered.phi())>1.0:
                if abs(mu.eta())<etaMax:
                    probe = mu
                    break

    if probe!=None:
        return [probe]
    else:
        return []
=======
    tags =[]
    for x in muons:
        for triggered in hlt:
            if deltaR(x.eta(),x.phi(),triggered.eta(),triggered.phi())<0.3 and x.pt()>25:
                tags.append(x)
                break
    if len(tags)==0:
        return []


    probes=[]
    for mu in muons:
        isProbe=False
        for tag in tags:
            if deltaR(mu.eta(),mu.phi(),tag.eta(),tag.phi())>1.0:
                if abs(mu.eta())<etaMax:
                    isProbe=True
                    break
        if isProbe:
            probes.append(mu)
        

    return probes
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3



def fetchStubs(event,ontime=True):
    phiSeg2    = Handle  ('std::vector<L1MuKBMTCombinedStub>')
    event.getByLabel('simKBmtfStubs',phiSeg2)
    if ontime:
        filtered=filter(lambda x: x.bxNum()==0, phiSeg2.product())
        return filtered
    else:
        return phiSeg2.product()

def fetchStubsOLD(event,ontime=False,isData=True):
    phiSeg    = Handle  ('L1MuDTChambPhContainer')
    if not isData:
        event.getByLabel('simTwinMuxDigis',phiSeg)
    else:
        event.getByLabel('BMTFStage2Digis',phiSeg)
    if ontime:
        filtered=filter(lambda x: x.bxNum()==0, phiSeg.product().getContainer())
        return filtered
    else:
        return phiSeg.product().getContainer()

def fetchStubsOLDTheta(event,isData=True):
    phiSeg    = Handle  ('L1MuDTChambThContainer')
    if not isData:
        event.getByLabel('dtTriggerPrimitiveDigis',phiSeg)
    else:
        event.getByLabel('BMTFStage2Digis',phiSeg)
    return phiSeg.product().getContainer()


def fetchGEN(event,etaMax=1.2):
    genH  = Handle  ('vector<reco::GenParticle>')
    event.getByLabel('genParticles',genH)
    genMuons=filter(lambda x: abs(x.pdgId())==13 and x.status()==1 and abs(x.eta())<etaMax,genH.product())
    return genMuons


def fetchRECO(event,etaMax=0.8):
    genH  = Handle  ('vector<reco::Muon>')
    event.getByLabel('muons',genH)
    genMuons=filter(lambda x: x.passed(ROOT.reco.Muon.CutBasedIdMediumPrompt) and x.pt()>10.0 and abs(x.eta())<2.4 and x.isolationR03().sumPt/x.pt()<0.15 and abs(x.eta())<etaMax,genH.product())
    return genMuons


def globalBMTFPhi(muon,calib=None):
    temp=muon.processor()*48+muon.hwPhi()
    temp=temp*2*math.pi/576.0-math.pi*15.0/180.0;
    if temp>math.pi:
        temp=temp-2*math.pi;

    if calib=='KMTF':
        K=1.0/muon.hwPt()
        if muon.hwSign()>0:
            K=-1.0/muon.hwPt()
#        return temp+3.752*K/(1.0-2.405*K*K/abs(K))
        return temp+5.740*K

    if calib=='BMTF':
        K=1.0/muon.hwPt()
        if muon.hwSign()>0:
            K=-1.0/muon.hwPt()
        return temp+5.740*K

    return temp;

<<<<<<< HEAD
=======

def rawPhi(muon):
    temp=muon.processor()*48+muon.hwPhi()
    temp=temp*2*math.pi/576.0-math.pi*15.0/180.0;
    if temp>math.pi:
        temp=temp-2*math.pi;
    return temp;


>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
def fetchBMTF(event,isData,etaMax=1.2):
    bmtfH  = Handle  ('BXVector<l1t::RegionalMuonCand>')
    if isData:
        event.getByLabel('BMTFStage2Digis:BMTF',bmtfH)
    else:
        event.getByLabel('simBmtfDigis:BMTF',bmtfH)

    bmtf=bmtfH.product()
    bmtfMuons=[]
    for bx in [0]:
        for j in range(0,bmtf.size(bx)):
            mu = bmtf.at(bx,j)
            pt = mu.hwPt()*0.5
<<<<<<< HEAD
            #calibration
            K=1.0/pt
#            K = 1.146*K-0.271*K*K+6.199e-4

            pt=1.0/K
            ####          
            phi=globalBMTFPhi(mu,'BMTF')
            eta = mu.hwEta()*0.010875           
            if abs(eta)<=etaMax:
                bmtfMuons.append(BMTFMuon(mu,pt,eta,phi))
    return sorted(bmtfMuons,key=lambda x: x.pt(),reverse=True)


def fetchKMTFNew(event,etaMax=1.2):
=======
            K=1.0/pt
            K=1.181*K/(1+0.4867*K)
            pt=1.0/K
            ####          
            phi=globalBMTFPhi(mu,'BMTF')
            rawP = rawPhi(mu)
            eta = mu.hwEta()*0.010875           
            if abs(eta)<=etaMax:
                b = BMTFMuon(mu,pt,eta,phi)
                b.rawPhi=rawP
                bmtfMuons.append(b)
    return sorted(bmtfMuons,key=lambda x: x.pt(),reverse=True)


def fetchKMTFNew(event,etaMax=1.2,saturate=True):
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
    kbmtfH  = Handle  ('BXVector<l1t::RegionalMuonCand>')
    event.getByLabel('simKBmtfDigis:BMTF',kbmtfH)
    kbmtf=kbmtfH.product()
    kbmtfMuons=[]
    for bx in [0]:
        for j in range(0,kbmtf.size(bx)):
            mu = kbmtf.at(bx,j)
            pt = mu.hwPt()*0.5
<<<<<<< HEAD

#            if pt!=0:
#                K=1.0/pt
#                K=0.932*K+0.182*K*K-4.826e-4
#                pt=1.0/K
            if pt>128.0:
                pt=128.0
            phi=globalBMTFPhi(mu,'KMTF')
            eta = mu.hwEta()*0.010875           
            if abs(eta)<=etaMax:
                kbmtfMuons.append(BMTFMuon(mu,pt,eta,phi))
=======
            K=1.0/pt
            K=1.18*K
            pt=1.0/K
            if pt>140.0 and saturate:
                pt=140.0
            phi=globalBMTFPhi(mu,'KMTF')
            rawP = rawPhi(mu)
            eta = mu.hwEta()*0.010875           
            if abs(eta)<=etaMax:
                b = BMTFMuon(mu,pt,eta,phi)
                b.rawPhi=rawP
                kbmtfMuons.append(b)
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
    return sorted(kbmtfMuons,key=lambda x: x.pt(),reverse=True)


def qPTInt(qPT,bits=14):
    lsb = lsBIT(bits)
    floatbinary = int(math.floor(abs(qPT)/lsb))
    return int((qPT/abs(qPT))*floatbinary)

def lsBIT(bits=14):
    maximum=1.25
    lsb = 1.25/pow(2,bits-1)
    return lsb

#import pdb;pdb.set_trace()

def fetchKMTF(event,etaMax=0.83,chi2=800000,dxyCut=100000):
<<<<<<< HEAD
    kmtfH  = Handle('vector<L1MuKBMTrack>')
    event.getByLabel('simKBmtfDigis',kmtfH)
    kmtf=filter(lambda x: abs(x.eta())<etaMax and x.approxChi2()<chi2 and abs(x.dxy())<dxyCut,kmtfH.product())
#    for k in kmtf:
#        K=1.0/k.pt()
#        if K<0.14:
#            K=0.967*K+0.756*K*K
#        else:
#            K=1.4*K-2.45*K*K
#        pt=1.0/K
#        k.setPtEtaPhi(pt,k.eta(),k.phi(),1)
        
#    kmtf=filter(lambda x: abs(x.eta())<etaMax and x.approxChi2()<chi2,kmtfH.product())
    return sorted(kmtf,key=lambda x: x.pt(),reverse=True)

def curvResidual(a,b):
    return (a.charge()/a.pt()-b.charge()/b.pt())*b.pt()/b.charge()
=======
    kmtfH  = Handle('BXVector<L1MuKBMTrack>')
    event.getByLabel('simKBmtfDigis',kmtfH)
    kmtf=kmtfH.product()
    out=[]
    for bx in [0]:
        for j in range(0,kmtf.size(bx)):
            mu =  kmtf.at(bx,j)
            if abs(mu.eta())<etaMax and mu.approxChi2()<chi2 and abs(mu.dxy())<dxyCut:
                out.append(mu)
    return sorted(out,key=lambda x: x.pt(),reverse=True)

def curvResidual(a,b,factor=1.0):
    return (a.charge()/a.pt()-factor*b.charge()/b.pt())*b.pt()/(factor*b.charge())
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3

def ptResidual(a,b):
    return (a.pt()-b.pt())/b.pt()

def curvResidualSTA(a,b):
    return (a.charge()/a.ptUnconstrained()-b.charge()/b.pt())*b.pt()/b.charge()




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


<<<<<<< HEAD
def log(counter,mystubs,gen,kmtf,bmtf):   
    print("--------EVENT"+str(counter)+"------------")
    print("-----------------------------")
    print("-----------------------------")
    print('Stubs:')
    for stub in mystubs:
        print('wheel={w} sector={sc} station={st} high/low={ts} phi={phi} phiB={phiB} qual={qual} BX={BX} eta1={eta1} eta2={eta2}'.format(w=stub.whNum(),sc=stub.scNum(),st=stub.stNum(),ts=stub.tag(),phi=stub.phi(),phiB=8*stub.phiB(),qual=stub.quality(),BX=stub.bxNum(),eta1=stub.eta1(),eta2=stub.eta2()))
    print('Gen muons:')
    for g in gen:
        print("Generated muon charge={q} pt={pt} eta={eta} phi={phi}".format(q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi()))
    print('BMTF:')
    for g in bmtf :
        print("BMTF charge={q} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF}".format(q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),qual=g.quality(),dxy=g.dxy(),pt2=g.ptUnconstrained(),HF=g.hasFineEta()))
    print('KMTF:')
    for g in kmtf :
        print("KMTF charge={q} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF}".format(q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),qual=g.quality(),dxy=g.dxy(),pt2=g.ptUnconstrained(),HF=g.hasFineEta()))
=======
def log(counter,mystubs,gen,kmtfFull,kmtf,bmtf):   
    print "--------EVENT"+str(counter)+"------------"
    print "-----------------------------"
    print "-----------------------------"
    print 'Stubs:'
    for stub in mystubs:
        print 'wheel={w} sector={sc} station={st} high/low={ts} phi={phi} phiB={phiB} qual={qual} BX={BX} eta1={eta1} eta2={eta2}'.format(w=stub.whNum(),sc=stub.scNum(),st=stub.stNum(),ts=stub.tag(),phi=stub.phi(),phiB=8*stub.phiB(),qual=stub.quality(),BX=stub.bxNum(),eta1=stub.eta1(),eta2=stub.eta2())
    print 'Gen muons:'
    for g in gen:
        print "Generated muon charge={q} pt={pt} eta={eta} phi={phi}".format(q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi())
    print 'BMTF:'
    for g in bmtf :
        print "BMTF sector={sector} charge={q} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF} rawPhi={hwPHI}".format(sector=g.processor(),q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),qual=g.quality(),dxy=g.dxy(),pt2=g.ptUnconstrained(),HF=g.hasFineEta(),hwPHI=g.phiINT())
    print 'KMTF:'
    for g in kmtf :
        print "KMTF sector={sector} charge={q} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF} rawPhi={hwPHI}".format(sector=g.processor(),q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),qual=g.quality(),dxy=g.dxy(),pt2=g.ptUnconstrained(),HF=g.hasFineEta(),hwPHI=g.phiINT())
    print 'KMTF Full:'
    for g in kmtfFull :
        print "KMTF charge={q} pt={pt} eta={eta} phi={phi} pattern={pattern}".format(q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),pattern=g.hitPattern() )
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3




<<<<<<< HEAD
    print("-----------------------------")
    print("-----------------------------")
    print("c + enter to continue")
    import pdb;pdb.set_trace()

#########Histograms#############

resKMTF = ROOT.TH2D("resKMTF","resKF",25,0,100,60,-2,2)
resSTAKMTF = ROOT.TH2D("resSTAKMTF","resKF",100,0,100,100,-8,8)
resBMTF = ROOT.TH2D("resBMTF","resKF",25,0,100,60,-2,2)
=======
    print "-----------------------------"
    print "-----------------------------"
    print "c + enter to continue"
    import pdb;pdb.set_trace()

#########Histograms#############
resKMTF = ROOT.TH2D("resKMTF","resKF",50,3,103,60,-2,2)

resKMTFTrack={}
for i in [0,3,5,6,7,9,10,11,12,13,14,15]:
    resKMTFTrack[i]=ROOT.TH2D("resKMTFTrack_"+str(i),"resKF",70,3,143,60,-2,2)


chiLow = ROOT.TH1D("chiLow","resKF",512,0,8192)

resKMTFEta = ROOT.TH2D("resKMTFEta","resKF",8,0,0.8,60,-2,2)

resSTAKMTF = ROOT.TH2D("resSTAKMTF","resKF",100,0,100,100,-8,8)
resBMTF = ROOT.TH2D("resBMTF","resKF",50,3,103,60,-2,2)
resBMTFEta = ROOT.TH2D("resBMTFEta","resKF",8,0,0.8,60,-2,2)

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
resPTKMTF = ROOT.TH2D("resPTKMTF","resKF",100,0,100,60,-2,2)
resPTBMTF = ROOT.TH2D("resPTBMTF","resKF",100,0,100,60,-2,2)
resEtaKMTF = ROOT.TH2D("resEtaKMTF","resKF",5,-1.2,1.2,50,-20.5*0.010875,20.5*0.010875)
resEtaBMTF = ROOT.TH2D("resEtaBMTF","resKF",5,-1.2,1.2,50,-20.5*0.010875,20.5*0.010875)


<<<<<<< HEAD
resPhiKMTF = ROOT.TH2D("resPhiKMTF","resKF",50,0,100,250,-0.5,0.5)
resPhiBMTF = ROOT.TH2D("resPhiBMTF","resKF",50,0,100,250,-0.5,0.5)
=======
resPhiKMTF = ROOT.TH2D("resPhiKMTF","resKF",50,3,103,250,-0.5,0.5)
phiCalibKMTF = ROOT.TH2D("phiCalibKMTF","resKF",100,-1.0/3.2,1.0/3.2,101,-0.5,0.5)
phiCalibBMTF = ROOT.TH2D("phiCalibBMTF","resKF",100,-1.0/3.2,1.0/3.2,101,-0.5,0.5)

resPhiBMTF = ROOT.TH2D("resPhiBMTF","resKF",50,3,103,250,-0.5,0.5)
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
resRBMTF = ROOT.TH1D("resRBMTF","resKF",250,0,8)
resRKMTF = ROOT.TH1D("resRKMTF","resKF",250,0,8)

fineEtaKMTF = ROOT.TH1D("fineEtaKMTF","fineEtaKMTF",2,0,2)
fineEtaBMTF = ROOT.TH1D("fineEtaBMTF","fineEtaBMTF",2,0,2)



genPt=ROOT.TH1F("genPt","genPt",50,0,100)

PTThresholds=[0,5,7,10,15,25,30]
genPtKMTF={}
genPtBMTF={}

for p in PTThresholds:
    genPtKMTF[p]=ROOT.TH1F("genPtKMTF_"+str(p),"genPt",50,0,100)
    genPtBMTF[p]=ROOT.TH1F("genPtBMTF_"+str(p),"genPt",50,0,100)



<<<<<<< HEAD
kfCalib = ROOT.TH2D("kfCalib","resKF",100,1.0/100.0,1.0/3.0,100,0,10)
bmtfCalib = ROOT.TH2D("bmtfCalib","resKF",100,1.0/100.0,1.0/3.0,100,0,10)
=======



kfCalibPlus = ROOT.TH2D("kfCalibPlus","resKF",560,3.2,143.2,100,0,10)
kfCalibMinus = ROOT.TH2D("kfCalibMinus","resKF",560,3.2,143.2,100,0,10)
kfCalib = ROOT.TH2D("kfCalib","resKF",560,3.2,143.2,100,0,10)
bmtfCalib = ROOT.TH2D("bmtfCalib","resKF",280,0.,140,100,0,10)



>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3


#etaArr = [-0.8,-0.5,-0.3,-0.15,0.0,0.15,0.3,0.5,0.8]

etaArr=[]
for i in range(0,51):
    etaArr.append(-0.833+ 2*0.833*i/50.0)


genEta=ROOT.TH1F("genEta","genEta",len(etaArr)-1,array('f',etaArr))
genEtaKMTF=ROOT.TH1F("genEtaKMTF","genPt",len(etaArr)-1,array('f',etaArr))
genEtaBMTF=ROOT.TH1F("genEtaBMTF","genEta",len(etaArr)-1,array('f',etaArr))

genPhi=ROOT.TH1F("genPhi","genEta",50,-math.pi,math.pi)
genPhiKMTF=ROOT.TH1F("genPhiKMTF","genPt",50,-math.pi,math.pi)
genPhiBMTF=ROOT.TH1F("genPhiBMTF","genEta",50,-math.pi,math.pi)


qualityKMTF = ROOT.TH1F("qualityKMTF","quality",16,0,16)
qualityBMTF = ROOT.TH1F("qualityBMTF","quality",16,0,16)

dxyKMTF = ROOT.TH1F("dxyKMTF","chiBest",32,0,32)

etaBMTF = ROOT.TH1F("etaBMTF","rateBMTF",24,-1.2,1.2)
etaKMTF = ROOT.TH1F("etaKMTF","rateKMTF",24,-1.2,1.2)

rateBMTF = ROOT.TH1F("rateBMTF","rateBMTF",20,2.5,102.5)
rateKMTF = ROOT.TH1F("rateKMTF","rateKMTF",20,2.5,102.5)


rateBMTFp7 = ROOT.TH1F("rateBMTFp7","rateBMTF",20,2.5,102.5)
rateKMTFp7 = ROOT.TH1F("rateKMTFp7","rateKMTF",20,2.5,102.5)

##############################


events=Events([tag+'.root'])
counter=-1
for event in events:
    counter=counter+1
    #fetch stubs
    stubs=[]
    kmtf=[]
    bmtf=[]
    stubsOLD=[]


    stubs = fetchStubs(event)
#    stubsOLD = fetchStubsOLD(event,True,isData)
#    stubsOLDTheta = fetchStubsOLDTheta(event,isData)
#    stubs = []
    stubsOLD = []
#    stubsOLDTheta = []


#    if counter==1000:
#        break;

<<<<<<< HEAD
#    print('OLD stubs')
#    for s in stubsOLD:
#        print(s.bxNum(),s.whNum(),s.scNum(),s.stNum(),s.phi(),s.phiB(),s.code())
=======
#    print 'OLD stubs'
#    for s in stubsOLD:
#        print s.bxNum(),s.whNum(),s.scNum(),s.stNum(),s.phi(),s.phiB(),s.code()
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3


#    reco=fetchRECO(event,2.4)
    reco=[]

    #fetch gen
    if isData:
<<<<<<< HEAD
        gen=fetchTP(event,0.83)
=======
#        gen=fetchTP(event,0.83)
        gen=[]
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
    else:
        gen  = fetchGEN(event,0.83)
#        bmtf = []
#        bmtf = fetchBMTF(event,isData,1.5)

    #fetch kalman (prompt)
    kmtf = fetchKMTFNew(event,1.5)
<<<<<<< HEAD
#   bmtf = fetchBMTF(event,isData,1.5)
    bmtf=[]

#    for g in gen:
#        print('GEN', g.pt(),g.eta(),g.phi())

#    for k in kmtf:
#        print('L1', k.pt(),k.eta(),k.phi())
=======
    bmtf = fetchBMTF(event,isData,1.5)
#    bmtf=[]

    #fetch detailed kalman
    kmtfFull = fetchKMTF(event,1.5)


#    for g in gen:
#        print 'GEN', g.pt(),g.eta(),g.phi() 

#    for k in kmtf:
#        print 'L1', k.pt(),k.eta(),k.phi() 
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3




    for track in kmtf:
        dxyKMTF.Fill(abs(track.dxy()))
        qualityKMTF.Fill(track.quality())
        etaKMTF.Fill(track.eta())
        fineEtaKMTF.Fill(track.hasFineEta())

    if len(kmtf)>0:   
        PT=kmtf[0].pt()        
        if kmtf[0].pt()>102.4:
            PT=102.4
            
        if abs(kmtf[0].eta())<0.7:
            rateKMTFp7.Fill(PT)

        rateKMTF.Fill(PT)

    for track in bmtf:
        qualityBMTF.Fill(track.quality())
        etaKMTF.Fill(track.eta())
        fineEtaBMTF.Fill(track.hasFineEta())

    if (len(bmtf)>0):    
        PT=bmtf[0].pt()        
        if bmtf[0].pt()>102.4:
            PT=102.4           
        if abs(bmtf[0].eta())<0.7:
            rateBMTFp7.Fill(PT)
        rateBMTF.Fill(PT)


<<<<<<< HEAD
=======
#        if len(kmtf)>0 and len(bmtf) and kmtf[0].pt()>50 and bmtf[0].pt()<20:
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3



    ##loop on gen and fill resolutuion and efficiencies
    for g in gen:
        if abs(g.eta())>0.83:
            continue
<<<<<<< HEAD
=======
        gK=g.charge()/g.pt()
        genPhiAt2 = g.phi()-2.675*gK;

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
        genPt.Fill(g.pt())
        ##the eta efficiency we want at the plateau to see strucuture
        if g.pt()>27.0:
            genEta.Fill(g.eta())
            genPhi.Fill(g.phi())

        #match *(loosely because we still use coarse eta)
<<<<<<< HEAD
#        matchedBMTF = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,bmtf) 
#        matchedKMTF = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,kmtf) 

        matchedBMTF = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<1.3,bmtf) 
        matchedKMTF = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<1.3,kmtf) 

#        if len(matchedKMTF)==0:
#            log(counter,stubs,gen,kmtf,bmtf)


#        if len(matchedKMTF)==0:
#            log(counter,stubs,gen,kmtf,bmtf)
=======
        matchedBMTF = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,bmtf) 
        matchedKMTF = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,kmtf) 
        matchedKMTFFull = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,kmtfFull) 

#        matchedBMTF = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<2.5,bmtf) 
#        matchedKMTF = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<2.5,kmtf) 
#        matchedKMTFFull = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<2.5,kmtfFull) 


>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3


        bestBMTF=None        
        if len(matchedBMTF)>0:
            bestBMTF = min(matchedBMTF,key = lambda x:  abs(curvResidual(x,g)))  
            resBMTF.Fill(g.pt(),curvResidual(bestBMTF,g))
<<<<<<< HEAD

            resPTBMTF.Fill(g.pt(),ptResidual(bestBMTF,g))
            resEtaBMTF.Fill(g.eta(),bestBMTF.eta()-g.eta())
            resPhiBMTF.Fill(g.pt(),bestBMTF.phi()-g.phi())
            resRBMTF.Fill(deltaR(g.eta(),g.phi(),bestBMTF.eta(),bestBMTF.phi()))
            bmtfCalib.Fill(1.0/bestBMTF.pt(),bestBMTF.pt()/g.pt())
=======
            resBMTFEta.Fill(abs(g.eta()),curvResidual(bestBMTF,g))

            resPTBMTF.Fill(g.pt(),ptResidual(bestBMTF,g))
            resEtaBMTF.Fill(g.eta(),bestBMTF.eta()-g.eta())
            resPhiBMTF.Fill(g.pt(),bestBMTF.rawPhi-genPhiAt2)
            if g.pt()<140:
                phiCalibBMTF.Fill(g.charge()/g.pt(),bestBMTF.rawPhi-g.phi())
            resRBMTF.Fill(deltaR(g.eta(),g.phi(),bestBMTF.eta(),bestBMTF.phi()))
            bmtfCalib.Fill(bestBMTF.pt(),bestBMTF.pt()/g.pt())

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3

            if g.pt()>27 and bestBMTF.pt()>15:
                genEtaBMTF.Fill(g.eta())
                genPhiBMTF.Fill(g.phi())

            # for the turn on , first cut on pt and then match    
            for threshold in PTThresholds:
                filteredBMTF = filter(lambda x: x.pt()>=float(threshold),matchedBMTF)
                if len(filteredBMTF)>0:
                    genPtBMTF[threshold].Fill(g.pt())

            

        bestKMTF=None        
        if len(matchedKMTF)>0:
            bestKMTF = min(matchedKMTF,key = lambda x:  abs(curvResidual(x,g)))  

<<<<<<< HEAD
#            if g.pt()>50 and bestKMTF.pt()<10:
#                log(counter,stubs,gen,kmtf,bmtf)
            resKMTF.Fill(g.pt(),curvResidual(bestKMTF,g))
=======

            resKMTF.Fill(g.pt(),curvResidual(bestKMTF,g))
            resKMTFEta.Fill(abs(g.eta()),curvResidual(bestKMTF,g))
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
            resSTAKMTF.Fill(g.pt(),curvResidualSTA(bestKMTF,g))
            resPTKMTF.Fill(g.pt(),ptResidual(bestKMTF,g))


            
            resEtaKMTF.Fill(g.eta(),bestKMTF.eta()-g.eta())
<<<<<<< HEAD
            resPhiKMTF.Fill(g.pt(),bestKMTF.phi()-g.phi())
            resRKMTF.Fill(deltaR(g.eta(),g.phi(),bestKMTF.eta(),bestKMTF.phi()))
            kfCalib.Fill(1.0/bestKMTF.pt(),bestKMTF.pt()/g.pt())
=======
            resPhiKMTF.Fill(g.pt(),bestKMTF.rawPhi-genPhiAt2)
            if g.pt()<140:
                phiCalibKMTF.Fill(g.charge()/g.pt(),bestKMTF.rawPhi-g.phi())

            resRKMTF.Fill(deltaR(g.eta(),g.phi(),bestKMTF.eta(),bestKMTF.phi()))
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
            K = bestKMTF.charge()/bestKMTF.pt()
            if K==0:
                K=1;
                
#        if len(matchedKMTF)>0 and len(matchedBMTF)>0:
#            if bestBMTF.hasFineEta() and (not bestKMTF.hasFineEta()):

#                for s in stubsOLDTheta:
#                    d=[]
#                    for i in range(0,7):
#                        d.append(s.position(i))
<<<<<<< HEAD
#                    print(s.bxNum(),s.scNum(), s.whNum(), s.stNum(),d)
#                log(counter,stubs,gen,kmtf,bmtf)
=======
#                    print s.bxNum(),s.scNum(), s.whNum(), s.stNum(),d    

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3

         

#            if abs(curvResidual(bestKMTF,g))>2.:
            if g.pt()>27 and bestKMTF.pt()>15:
                genEtaKMTF.Fill(g.eta())
                genPhiKMTF.Fill(g.phi())

                
            for threshold in PTThresholds:
                filteredKMTF = filter(lambda x: x.pt()>=float(threshold),matchedKMTF)
                if len(filteredKMTF)>0:
                    genPtKMTF[threshold].Fill(g.pt())
                        
<<<<<<< HEAD
        if bestKMTF==None and bestBMTF!=None and g.pt()<4:
            log(counter,stubs,gen,kmtf,bmtf)
#        if bestKMTF!=None and bestBMTF!=None and abs(bestKMTF.eta()-g.eta())>abs(bestBMTF.eta()-g.eta()):
#            print('Input Theta stubs')
=======
#        if bestKMTF==None  and bestBMTF!=None and  g.pt()<5:
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)

#        if bestKMTF!=None  and bestBMTF!=None and abs(genPhiAt2-bestKMTF.rawPhi)>4.0*abs(genPhiAt2-bestBMTF.rawPhi) and g.pt()>30:
#            print 'Residual Kalman=',abs(genPhiAt2-bestKMTF.rawPhi),'raw=',bestKMTF.rawPhi,'int=',bestKMTF.phiINT()
#            print 'Residual BMTF=',abs(genPhiAt2-bestBMTF.rawPhi),'raw=',bestBMTF.rawPhi,'int=',bestBMTF.phiINT()
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)

#        if bestKMTF!=None  and g.pt()<5 and  curvResidual(bestKMTF,g)>0.2:
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)


#        if bestKMTF!=None and bestBMTF!=None and abs(bestKMTF.eta()-g.eta())>abs(bestBMTF.eta()-g.eta()):
#            print 'Input Theta stubs'
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
#            for s in stubsOLDTheta:
#                if s.bxNum()!=0:
#                    continue
#                a=[]
#                for i in range(0,7):
#                    a.append(s.position(i))
<<<<<<< HEAD
#                print(s.whNum(),s.scNum(),s.stNum(),':',a)
#            print('Combined stubs')
#
#            for s in stubs:
#                print(s.whNum(),s.scNum(),s.stNum(),s.eta1(),s.eta2(),s.qeta1(),s.qeta2())
#            import pdb;pdb.set_trace()

=======
#                print s.whNum(),s.scNum(),s.stNum(),':',a
#            print 'Combined stubs'
#
#            for s in stubs:
#                print s.whNum(),s.scNum(),s.stNum(),s.eta1(),s.eta2(),s.qeta1(),s.qeta2()
#            import pdb;pdb.set_trace()

        if len(matchedKMTFFull)>0:
            bestKMTFFull = min(matchedKMTFFull,key = lambda x:  abs(curvResidual(x,g)))  
            if g.pt()<5:
                chiLow.Fill(bestKMTFFull.curvatureAtMuon());

                    
            resKMTFTrack[bestKMTFFull.hitPattern()].Fill(g.pt(),curvResidual(bestKMTFFull,g))
            resKMTFTrack[0].Fill(g.pt(),curvResidual(bestKMTFFull,g))
            if bestKMTFFull.charge()>0:
                kfCalibPlus.Fill(bestKMTFFull.pt(),bestKMTFFull.pt()/g.pt())
            else:
                kfCalibMinus.Fill(bestKMTFFull.pt(),bestKMTFFull.pt()/g.pt())
            kfCalib.Fill(bestKMTFFull.pt(),bestKMTFFull.pt()/g.pt())





         



                        

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3

        
        
f=ROOT.TFile("results_"+tag+".root","RECREATE")




<<<<<<< HEAD

resKMTF.Write()     
resSTAKMTF.Write()     
resPTKMTF.Write()     
resBMTF.Write()
=======
chiLow.Write()

resKMTF.Write()     
for n,t in resKMTFTrack.iteritems():
    t.Write()
    
resKMTFEta.Write()     
resSTAKMTF.Write()     
resPTKMTF.Write()     
resBMTF.Write()
resBMTFEta.Write()
>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
resPTBMTF.Write()
resEtaKMTF.Write()     
resEtaBMTF.Write()     
resPhiKMTF.Write()     
resRKMTF.Write()     
resPhiBMTF.Write()     
<<<<<<< HEAD
=======

phiCalibBMTF.Write()
phiCalibKMTF.Write()


>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
resRBMTF.Write()     
#bmtfCalib.Write()
#kfCalib.Write()
genPt.Write()

for p in PTThresholds:
    genPtKMTF[p].Write()
    genPtBMTF[p].Write()
    kmtfEff = ROOT.TGraphAsymmErrors(genPtKMTF[p],genPt)
    kmtfEff.Write("efficiencyVsPtKMTF"+str(p))
    bmtfEff = ROOT.TGraphAsymmErrors(genPtBMTF[p],genPt)
    bmtfEff.Write("efficiencyVsPtBMTF"+str(p))




kmtfEffEta = ROOT.TGraphAsymmErrors(genEtaKMTF,genEta)
kmtfEffEta.Write("efficiencyVsEtaKMTF")

bmtfEffEta = ROOT.TGraphAsymmErrors(genEtaBMTF,genEta)
bmtfEffEta.Write("efficiencyVsEtaBMTF")


kmtfEffPhi = ROOT.TGraphAsymmErrors(genPhiKMTF,genPhi)
kmtfEffPhi.Write("efficiencyVsPhiKMTF")

bmtfEffPhi = ROOT.TGraphAsymmErrors(genPhiBMTF,genPhi)
bmtfEffPhi.Write("efficiencyVsPhiBMTF")

etaKMTF.Write()
etaBMTF.Write()


dxyKMTF.Write()
qualityKMTF.Write()
qualityBMTF.Write()



rateBMTF.Write()      
c = rateBMTF.GetCumulative(False)
c.SetLineWidth(3)
c.SetLineColor(ROOT.kBlack)
c.Write("normRateBMTF")     

<<<<<<< HEAD
kfCalib.Write()
=======
kfCalibPlus.Write()
kfCalibMinus.Write()
kfCalib.Write()

>>>>>>> ad4437d91bc... branch for 10_2_X including only the Kalman Emulator v2.3
bmtfCalib.Write()

rateKMTF.Write()      
c = rateKMTF.GetCumulative(False)
c.Write("normRateKMTF")     

rateBMTFp7.Write()      
c = rateBMTFp7.GetCumulative(False)
c.Write("normRateBMTFEtaP7")     

rateKMTFp7.Write()      
c = rateKMTFp7.GetCumulative(False)
c.Write("normRateKMTFEtaP7")     



fineEtaKMTF.Write()
fineEtaBMTF.Write()

f.Close()






