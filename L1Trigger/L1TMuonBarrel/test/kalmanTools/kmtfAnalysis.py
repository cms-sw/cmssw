from __future__ import print_function
import ROOT,itertools,math      #
from array import array         #
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()
#



#verbose=False
#tag='singleMuonOfficial'
#isData=False

tag='zerobias'
#tag='zskim'
isData=True





##A class to keep BMTF data
class BMTFMuon:
    def __init__(self,mu,pt,eta,phi):
        self.muon=mu
        self.p4 = ROOT.reco.Candidate.PolarLorentzVector(pt,eta,phi,0.105)

    def quality(self):
        return self.muon.hwQual()

    def phiINT(self):
        return self.muon.hwPhi()

    def processor(self):
        return self.muon.processor()

    def hasFineEta(self):
        return self.muon.hwHF()

    def ptUnconstrained(self):
        return self.muon.hwPtUnconstrained()

    def dxy(self):
        return self.muon.hwDXY()

    def charge(self):
        if self.muon.hwSign()>0:
            return -1
        else:
            return +1

    def __getattr__(self, name):
        return getattr(self.p4,name)


###Common methods############

def getQual(track):
    q=0
    for stub in track.stubs():
        q+=stub.quality()
    return q;


def fetchTP(event,etaMax=0.83):
    tH  = Handle  ('trigger::TriggerEvent')
    mH  = Handle  ('std::vector<reco::Muon>')


    event.getByLabel('hltTriggerSummaryAOD','','HLT',tH)
    event.getByLabel('muons',mH)

    muons=filter( lambda x: x.passed(ROOT.reco.Muon.CutBasedIdMediumPrompt) and x.numberOfMatches()>1 and x.pt()>10.0 and abs(x.eta())<2.4 and x.isolationR03().sumPt/x.pt()<0.15,mH.product())
    if len(muons)<2:
        return []
    trigger=tH.product()
#    for f in range(0,trigger.sizeFilters()):
#        print(f,trigger.filterLabel(f))
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


def rawPhi(muon):
    temp=muon.processor()*48+muon.hwPhi()
    temp=temp*2*math.pi/576.0-math.pi*15.0/180.0;
    if temp>math.pi:
        temp=temp-2*math.pi;
    return temp;


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
    kbmtfH  = Handle  ('BXVector<l1t::RegionalMuonCand>')
    event.getByLabel('simKBmtfDigis:BMTF',kbmtfH)
    kbmtf=kbmtfH.product()
    kbmtfMuons=[]
    for bx in [0]:
        for j in range(0,kbmtf.size(bx)):
            mu = kbmtf.at(bx,j)
            pt =mu.hwPt()*0.5
            K=1.0/pt
            K=1.14*K
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

def fetchKMTF(event,etaMax=0.83,patterns=[],comps=[],chis=[],pts=[]):
    kmtfH  = Handle('BXVector<L1MuKBMTrack>')
    event.getByLabel('simKBmtfDigis',kmtfH)
    kmtf=kmtfH.product()
    out=[]
    for bx in [0]:
        for j in range(0,kmtf.size(bx)):
            mu =  kmtf.at(bx,j)
            if abs(mu.eta())<etaMax:
                veto=False
                for pattern,comp,chi,pt in zip(patterns,comps,chis,pts):
                    if mu.hitPattern()==p and mu.pt()>pt and (mu.trackCompatibility()>comp or mu.approxChi2()>chi):
                        veto=True
                        break;
                if not veto:
                    out.append(mu)
    return sorted(out,key=lambda x: x.pt(),reverse=True)

def curvResidual(a,b,factor=1.0):
    return (a.charge()/a.pt()-factor*b.charge()/b.pt())*b.pt()/(factor*b.charge())

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


def log(counter,mystubs,gen,kmtfFull,kmtf,bmtf):
    print("--------EVENT"+str(counter)+"------------")
    print("-----------------------------")
    print("-----------------------------")
    print('Stubs:')
    for stub in mystubs:
        print('wheel={w} sector={sc} station={st} high/low={ts} phi={phi} phiB={phiB} qual={qual} BX={BX} eta1={eta1} eta2={eta2}'.format(w=stub.whNum(),sc=stub.scNum(),st=stub.stNum(),ts=stub.tag(),phi=stub.phi(),phiB=stub.phiB(),qual=stub.quality(),BX=stub.bxNum(),eta1=stub.eta1(),eta2=stub.eta2()))
    print('Gen muons:')
    for g in gen:
        print("Generated muon charge={q} pt={pt} eta={eta} phi={phi}".format(q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi()))
    print('BMTF:')
    for g in bmtf :
        print("BMTF sector={sector} charge={q} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF} rawPhi={hwPHI}".format(sector=g.processor(),q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),qual=g.quality(),dxy=g.dxy(),pt2=g.ptUnconstrained(),HF=g.hasFineEta(),hwPHI=g.phiINT()))
    print('KMTF:')
    for g in kmtf :
        print("KMTF sector={sector} charge={q} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF} rawPhi={hwPHI}".format(sector=g.processor(),q=g.charge(),pt=g.pt(),eta=g.eta(),phi=g.phi(),qual=g.quality(),dxy=g.dxy(),pt2=g.ptUnconstrained(),HF=g.hasFineEta(),hwPHI=g.phiINT()))
    print('KMTF Full:')
    for g in kmtfFull :
        print("KMTF charge={q} pt={pt} ptSTA={PTSTA} eta={eta} phi={phi} pattern={pattern} chi={chi1} comp={comp}".format(q=g.charge(),pt=g.pt(),PTSTA=g.ptUnconstrained(),eta=g.eta(),phi=g.phi(),pattern=g.hitPattern(),chi1=g.approxChi2(),comp=g.trackCompatibility() ))




    print("-----------------------------")
    print("-----------------------------")
    print("c + enter to continue")
    import pdb;pdb.set_trace()

#########Histograms#############
resKMTF = ROOT.TH2D("resKMTF","resKF",50,3,103,60,-2,2)


resKMTFTrack={}
for i in [0,3,5,6,7,9,10,11,12,13,14,15]:
    resKMTFTrack[i]=ROOT.TH2D("resKMTFTrack_"+str(i),"resKF",70,3,143,60,-2,2)



chiMatched={}
trackComp={}
trackCompAll={}
chiAll={}

for i in [0,3,5,6,7,9,10,11,12,13,14,15]:
    chiMatched[i] = ROOT.TH2D("chiMatched_"+str(i),"resKF",32,0,1024,64,0,256)
    trackComp[i] = ROOT.TH2D("trackComp_"+str(i),"resKF",32,0,1024,100,0,100)
    trackCompAll[i] = ROOT.TH2D("trackCompAll_"+str(i),"resKF",20,0,1024,100,0,100)
    chiAll[i] = ROOT.TH2D("chiAll_"+str(i),"resKF",32,0,1024,64,0,256)



quality = ROOT.TH1D("quality","resKF",24,0,24)

resKMTFEta = ROOT.TH2D("resKMTFEta","resKF",8,0,0.8,60,-2,2)

resSTAKMTF = ROOT.TH2D("resSTAKMTF","resKF",100,0,100,100,-8,8)
resBMTF = ROOT.TH2D("resBMTF","resKF",50,3,103,60,-2,2)
resBMTFEta = ROOT.TH2D("resBMTFEta","resKF",8,0,0.8,60,-2,2)

resPTKMTF = ROOT.TH2D("resPTKMTF","resKF",100,0,100,60,-2,2)
resPTBMTF = ROOT.TH2D("resPTBMTF","resKF",100,0,100,60,-2,2)
resEtaKMTF = ROOT.TH2D("resEtaKMTF","resKF",5,-1.2,1.2,50,-20.5*0.010875,20.5*0.010875)
resEtaBMTF = ROOT.TH2D("resEtaBMTF","resKF",5,-1.2,1.2,50,-20.5*0.010875,20.5*0.010875)


resPhiKMTF = ROOT.TH2D("resPhiKMTF","resKF",50,3,103,250,-0.5,0.5)
phiCalibKMTF = ROOT.TH2D("phiCalibKMTF","resKF",100,-1.0/3.2,1.0/3.2,101,-0.5,0.5)
phiCalibBMTF = ROOT.TH2D("phiCalibBMTF","resKF",100,-1.0/3.2,1.0/3.2,101,-0.5,0.5)

resPhiBMTF = ROOT.TH2D("resPhiBMTF","resKF",50,3,103,250,-0.5,0.5)
resRBMTF = ROOT.TH1D("resRBMTF","resKF",250,0,8)
resRKMTF = ROOT.TH1D("resRKMTF","resKF",250,0,8)

fineEtaKMTF = ROOT.TH1D("fineEtaKMTF","fineEtaKMTF",2,0,2)
fineEtaBMTF = ROOT.TH1D("fineEtaBMTF","fineEtaBMTF",2,0,2)



genPt=ROOT.TH1F("genPt","genPt",50,0,100)

PTThresholds=[0,5,7,10,15,25,30]
genPtKMTF={}
genPtBMTF={}
genEtaKMTF={}
genEtaBMTF={}

etaArr=[]
for i in range(0,21):
    etaArr.append(-0.833+ 2*0.833*i/20.0)


for p in PTThresholds:
    genPtKMTF[p]=ROOT.TH1F("genPtKMTF_"+str(p),"genPt",50,0,100)
    genPtBMTF[p]=ROOT.TH1F("genPtBMTF_"+str(p),"genPt",50,0,100)
    genEtaKMTF[p]=ROOT.TH1F("genEtaKMTF"+str(p),"genPt",len(etaArr)-1,array('f',etaArr))
    genEtaBMTF[p]=ROOT.TH1F("genEtaBMTF"+str(p),"genEta",len(etaArr)-1,array('f',etaArr))



kfCalibPlus={}
kfCalibMinus={}
ratePerTrack={}
for track in [3,5,6,7,9,10,11,12,13,14,15]:
    kfCalibPlus[track] = ROOT.TH2D("kfCalibPlus_"+str(track),"resKF",560,3.2,143.2,100,0,10)
    kfCalibMinus[track] = ROOT.TH2D("kfCalibMinus_"+str(track),"resKF",560,3.2,143.2,100,0,10)
    ratePerTrack[track] = ROOT.TH1F("ratePerTrack_"+str(track),"rateKMTF",20,2.5,102.5)

kfCalib = ROOT.TH2D("kfCalib","resKF",560,3.2,143.2,100,0,10)
bmtfCalib = ROOT.TH2D("bmtfCalib","resKF",280,0.,140,100,0,10)





#etaArr = [-0.8,-0.5,-0.3,-0.15,0.0,0.15,0.3,0.5,0.8]



genEta=ROOT.TH1F("genEta","genEta",len(etaArr)-1,array('f',etaArr))

genPhi=ROOT.TH1F("genPhi","genEta",50,-math.pi,math.pi)
genPhiKMTF=ROOT.TH1F("genPhiKMTF","genPt",50,-math.pi,math.pi)
genPhiBMTF=ROOT.TH1F("genPhiBMTF","genEta",50,-math.pi,math.pi)


qualityKMTF = ROOT.TH1F("qualityKMTF","quality",16,0,16)
qualityBMTF = ROOT.TH1F("qualityBMTF","quality",16,0,16)

dxyKMTF = ROOT.TH1F("dxyKMTF","chiBest",4,0,4)

etaBMTF = ROOT.TH1F("etaBMTF","rateBMTF",24,-1.2,1.2)
etaKMTF = ROOT.TH1F("etaKMTF","rateKMTF",24,-1.2,1.2)

rateBMTF = ROOT.TH1F("rateBMTF","rateBMTF",50,0,50)
rateKMTF = ROOT.TH1F("rateKMTF","rateKMTF",50,0,50)



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

#    print('OLD stubs')
#    for s in stubsOLD:
#        print(s.bxNum(),s.whNum(),s.scNum(),s.stNum(),s.phi(),s.phiB(),s.code())


#    reco=fetchRECO(event,2.4)
    reco=[]

    #fetch gen
    if isData:
        if tag=="zerobias" or tag=="output":
            gen=[]
        else:
            gen=fetchTP(event,0.83)

    else:
        gen  = fetchGEN(event,0.83)
#        bmtf = []
#        bmtf = fetchBMTF(event,isData,1.5)

    #fetch kalman (prompt)
    kmtf = fetchKMTFNew(event,1.5)
    bmtf = fetchBMTF(event,isData,1.5)
#    bmtf=[]

    #fetch detailed kalman
    kmtfFull = fetchKMTF(event,1.5)


#    for g in gen:
#        print('GEN', g.pt(),g.eta(),g.phi() )

#    for k in kmtf:
#        print('L1', k.pt(),k.eta(),k.phi() )

#    log(counter,stubs,gen,kmtfFull,kmtf,bmtf)


    for track in kmtfFull:
        chiAll[track.hitPattern()].Fill(abs(track.curvatureAtVertex()),track.approxChi2())
        trackCompAll[track.hitPattern()].Fill(abs(track.curvatureAtVertex()),min(track.trackCompatibility(),99.5));

        quality.Fill(getQual(track))

    for track in kmtf:
        dxyKMTF.Fill(abs(track.dxy()))
        qualityKMTF.Fill(track.quality())
        etaKMTF.Fill(track.eta())
        fineEtaKMTF.Fill(track.hasFineEta())

    if len(kmtf)>0:
        PT=kmtf[0].pt()
        if kmtf[0].pt()>49.99:
            PT=49.99

        if abs(kmtf[0].eta())<0.7:
            rateKMTFp7.Fill(PT)

        rateKMTF.Fill(PT)

    if len(kmtfFull)>0:

        PT=kmtfFull[0].pt()
        if kmtfFull[0].pt()>49.99:
            PT=49.99
        ratePerTrack[kmtfFull[0].hitPattern()].Fill(PT)


    for track in bmtf:
        qualityBMTF.Fill(track.quality())
        etaBMTF.Fill(track.eta())
        fineEtaBMTF.Fill(track.hasFineEta())

    if (len(bmtf)>0):
        PT=bmtf[0].pt()
        if bmtf[0].pt()>49.99:
            PT=49.99
        if abs(bmtf[0].eta())<0.7:
            rateBMTFp7.Fill(PT)
        rateBMTF.Fill(PT)

#    if ( len(kmtfFull)>0)  and (kmtfFull[0].pt()>50):
#        log(counter,stubs,gen,kmtfFull,kmtf,bmtf)


#    if (len(kmtf)>0 and kmtf[0].pt()>20) and  (len(bmtf)==0 or (len(bmtf)>0 and bmtf[0].pt()<10)):
#        log(counter,stubs,gen,kmtfFull,kmtf,bmtf)


#    log(counter,stubs,gen,kmtfFull,kmtf,bmtf)

    ##loop on gen and fill resolutuion and efficiencies
    for g in gen:
        if abs(g.eta())>0.83:
            continue
        gK=g.charge()/g.pt()
        genPhiAt2 = g.phi()-2.675*gK;

        genPt.Fill(g.pt())
        ##the eta efficiency we want at the plateau to see strucuture
        if g.pt()>40.0:
            genEta.Fill(g.eta())
            genPhi.Fill(g.phi())

        #match *(loosely because we still use coarse eta)
        matchedBMTF = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,bmtf)
        matchedKMTF = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,kmtf)
        matchedKMTFFull = filter(lambda x: deltaR(g.eta(),g.phi(),x.eta(),x.phi())<0.3,kmtfFull)

#        matchedBMTF = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<2.5,bmtf)
#        matchedKMTF = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<2.5,kmtf)
#        matchedKMTFFull = filter(lambda x: abs(deltaPhi(g.phi(),x.phi()))<2.5,kmtfFull)




        bestBMTF=None
        if len(matchedBMTF)>0:
            bestBMTF = max(matchedBMTF,key = lambda x:  x.quality()*1000+x.pt())
            resBMTF.Fill(g.pt(),curvResidual(bestBMTF,g))
            resBMTFEta.Fill(abs(g.eta()),curvResidual(bestBMTF,g))

            resPTBMTF.Fill(g.pt(),ptResidual(bestBMTF,g))
            resEtaBMTF.Fill(g.eta(),bestBMTF.eta()-g.eta())
            resPhiBMTF.Fill(g.pt(),bestBMTF.rawPhi-genPhiAt2)
            if g.pt()<140:
                phiCalibBMTF.Fill(g.charge()/g.pt(),bestBMTF.rawPhi-g.phi())
            resRBMTF.Fill(deltaR(g.eta(),g.phi(),bestBMTF.eta(),bestBMTF.phi()))
            bmtfCalib.Fill(bestBMTF.pt(),bestBMTF.pt()/g.pt())

            # for the turn on , first cut on pt and then match
            for threshold in PTThresholds:
                filteredBMTF = filter(lambda x: x.pt()>=float(threshold),matchedBMTF)
                if len(filteredBMTF)>0:
                    genPtBMTF[threshold].Fill(g.pt())
                    if g.pt()>40:
                        genEtaBMTF[threshold].Fill(g.eta())


        bestKMTF=None
        if len(matchedKMTF)>0:
            bestKMTF = max(matchedKMTF,key = lambda x:  1000*x.quality()+x.pt())


            resKMTF.Fill(g.pt(),curvResidual(bestKMTF,g))
            resKMTFEta.Fill(abs(g.eta()),curvResidual(bestKMTF,g))
            resSTAKMTF.Fill(g.pt(),curvResidualSTA(bestKMTF,g))
            resPTKMTF.Fill(g.pt(),ptResidual(bestKMTF,g))



            resEtaKMTF.Fill(g.eta(),bestKMTF.eta()-g.eta())
            resPhiKMTF.Fill(g.pt(),bestKMTF.rawPhi-genPhiAt2)
            if g.pt()<140:
                phiCalibKMTF.Fill(g.charge()/g.pt(),bestKMTF.rawPhi-g.phi())

            resRKMTF.Fill(deltaR(g.eta(),g.phi(),bestKMTF.eta(),bestKMTF.phi()))
            K = bestKMTF.charge()/bestKMTF.pt()
            if K==0:
                K=1;

#        if len(matchedKMTF)>0 and len(matchedBMTF)>0:
#            if bestBMTF.hasFineEta() and (not bestKMTF.hasFineEta()):

#                for s in stubsOLDTheta:
#                    d=[]
#                    for i in range(0,7):
#                        d.append(s.position(i))
#                    print(s.bxNum(),s.scNum(), s.whNum(), s.stNum(),d    )




#            if abs(curvResidual(bestKMTF,g))>2.:


            for threshold in PTThresholds:
                filteredKMTF = filter(lambda x: x.pt()>=float(threshold),matchedKMTF)
                if len(filteredKMTF)>0:
                    genPtKMTF[threshold].Fill(g.pt())
                if len(filteredKMTF)>0 and g.pt()>40:
                    genEtaKMTF[threshold].Fill(g.eta())

#        if (bestKMTF==None or (bestKMTF!=None and bestKMTF.pt()<15))  and bestBMTF!=None and  g.pt()>30 and abs(g.eta())>0.15 and abs(g.eta())<0.32:
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)

#        if bestKMTF!=None  and bestBMTF!=None and g.pt()>30 and abs(g.eta())>0.15 and abs(g.eta())<0.3 and bestKMTF.pt()<25 and bestBMTF.pt()>25:
#            print('Residual Kalman=',abs(genPhiAt2-bestKMTF.rawPhi),'raw=',bestKMTF.rawPhi,'int=',bestKMTF.phiINT())
#            print('Residual BMTF=',abs(genPhiAt2-bestBMTF.rawPhi),'raw=',bestBMTF.rawPhi,'int=',bestBMTF.phiINT())
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)

#        if bestKMTF!=None  and g.pt()<5 and  curvResidual(bestKMTF,g)>0.2:
#            log(counter,stubs,gen,kmtfFull,kmtf,bmtf)


#        if bestKMTF!=None and bestBMTF!=None and abs(bestKMTF.eta()-g.eta())>abs(bestBMTF.eta()-g.eta()):
#            print('Input Theta stubs')
#            for s in stubsOLDTheta:
#                if s.bxNum()!=0:
#                    continue
#                a=[]
#                for i in range(0,7):
#                    a.append(s.position(i))
#                print(s.whNum(),s.scNum(),s.stNum(),':',a)
#            print('Combined stubs')
#
#            for s in stubs:
#                print(s.whNum(),s.scNum(),s.stNum(),s.eta1(),s.eta2(),s.qeta1(),s.qeta2())
#            import pdb;pdb.set_trace()

        if len(matchedKMTFFull)>0:
            bestKMTFFull = max(matchedKMTFFull,key = lambda x: x.rank()*1000+x.pt())
            chiMatched[bestKMTFFull.hitPattern()].Fill(abs(bestKMTFFull.curvatureAtVertex()),bestKMTFFull.approxChi2());
            trackComp[bestKMTFFull.hitPattern()].Fill(abs(bestKMTFFull.curvatureAtVertex()),min(bestKMTFFull.trackCompatibility(),99.5));


            resKMTFTrack[bestKMTFFull.hitPattern()].Fill(g.pt(),curvResidual(bestKMTFFull,g))
            resKMTFTrack[0].Fill(g.pt(),curvResidual(bestKMTFFull,g))
            if bestKMTFFull.charge()>0:
                kfCalibPlus[bestKMTFFull.hitPattern()].Fill(bestKMTFFull.pt(),bestKMTFFull.pt()/g.pt())
            else:
                kfCalibMinus[bestKMTFFull.hitPattern()].Fill(bestKMTFFull.pt(),bestKMTFFull.pt()/g.pt())
            kfCalib.Fill(bestKMTFFull.pt(),bestKMTFFull.pt()/g.pt())














f=ROOT.TFile("results_"+tag+".root","RECREATE")





quality.Write()


resKMTF.Write()
for n,t in resKMTFTrack.iteritems():
    t.Write()

resKMTFEta.Write()
resSTAKMTF.Write()
resPTKMTF.Write()
resBMTF.Write()
resBMTFEta.Write()
resPTBMTF.Write()
resEtaKMTF.Write()
resEtaBMTF.Write()
resPhiKMTF.Write()
resRKMTF.Write()
resPhiBMTF.Write()

phiCalibBMTF.Write()
phiCalibKMTF.Write()


resRBMTF.Write()
#bmtfCalib.Write()
#kfCalib.Write()
genPt.Write()

for p in PTThresholds:
    genPtKMTF[p].Write()
    genPtBMTF[p].Write()
    genEtaKMTF[p].Write()
    genEtaBMTF[p].Write()

    kmtfEff = ROOT.TGraphAsymmErrors(genPtKMTF[p],genPt)
    kmtfEff.Write("efficiencyVsPtKMTF"+str(p))
    bmtfEff = ROOT.TGraphAsymmErrors(genPtBMTF[p],genPt)
    bmtfEff.Write("efficiencyVsPtBMTF"+str(p))

    kmtfEff = ROOT.TGraphAsymmErrors(genEtaKMTF[p],genEta)
    kmtfEff.Write("efficiencyVsEtaKMTF"+str(p))
    bmtfEff = ROOT.TGraphAsymmErrors(genEtaBMTF[p],genEta)
    bmtfEff.Write("efficiencyVsEtaBMTF"+str(p))



    genPtKMTF[p].Add(genPtBMTF[p],-1)
    genPtKMTF[p].Divide(genPt)
    genPtKMTF[p].Write("efficiencyDiffVsPt"+str(p))

    genEtaKMTF[p].Add(genEtaBMTF[p],-1)
    genEtaKMTF[p].Divide(genEta)
    genEtaKMTF[p].Write("efficiencyDiffVsEta"+str(p))





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

for track in kfCalibPlus.keys():
    kfCalibPlus[track].Write()
    kfCalibMinus[track].Write()
    ratePerTrack[track].Write()
    chiMatched[track].Write()
    chiAll[track].Write()
    trackComp[track].Write()
    trackCompAll[track].Write()

kfCalib.Write()

bmtfCalib.Write()

rateKMTF.Write()
c = rateKMTF.GetCumulative(False)
c.Write("normRateKMTF")


d = rateBMTF.GetCumulative(False)
d.Divide(c)
d.Write("rateRatioBMTFoverKMTF")


rateBMTFp7.Write()
c = rateBMTFp7.GetCumulative(False)
c.Write("normRateBMTFEtaP7")

rateKMTFp7.Write()
c = rateKMTFp7.GetCumulative(False)
c.Write("normRateKMTFEtaP7")



fineEtaKMTF.Write()
fineEtaBMTF.Write()

f.Close()
