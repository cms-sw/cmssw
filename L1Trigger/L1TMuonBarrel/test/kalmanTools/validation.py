from __future__ import print_function
import ROOT,itertools,math      #
from array import array         #
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()
#




tag='output'


##A class to keep BMTF data




###Common methods############

def fetchStubsOLD(event,ontime=False,isData=True):
    phiSeg    = Handle  ('L1MuDTChambPhContainer')
    if not isData:
        event.getByLabel('simTwinMuxDigis',phiSeg)
    else:
        event.getByLabel('bmtfDigis',phiSeg)
    if ontime:
        filtered=filter(lambda x: x.bxNum()==0, phiSeg.product().getContainer())
        return filtered
    else:
        return phiSeg.product().getContainer()


def fetchStubs(event,ontime=True):
    phiSeg2    = Handle  ('std::vector<L1MuKBMTCombinedStub>')
    event.getByLabel('simKBmtfStubs',phiSeg2)
    if ontime:
        filtered=filter(lambda x: x.bxNum()==0, phiSeg2.product())
        return filtered
    else:
        return phiSeg2.product()
def globalBMTFPhi(muon):
    temp=muon.processor()*48+muon.hwPhi()
    temp=temp*2*math.pi/576.0-math.pi*15.0/180.0;
    if temp>math.pi:
        temp=temp-2*math.pi;

    K=1.0/muon.hwPt()
    if muon.hwSign()>0:
        K=-1.0/muon.hwPt()
    return temp+5.740*K



def fetchKMTF(event,etaMax,collection):
    kbmtfH  = Handle  ('BXVector<l1t::RegionalMuonCand>')
    event.getByLabel(collection,kbmtfH)
    kbmtf=kbmtfH.product()
    kbmtfMuons={}
    for bx in [-3,-2,-1,0,1,2,3]:
        kbmtfMuons[bx]=[]
    for bx in range(kbmtf.getFirstBX(),kbmtf.getLastBX()+1):
        for j in range(0,kbmtf.size(bx)):
            mu = kbmtf.at(bx,j)
            kbmtfMuons[bx].append(mu)
#        kbmtfMuons[bx]=sorted(kbmtfMuons[bx],key=lambda x: x.hwPt(),reverse=True)
    return kbmtfMuons

def curvResidual(a,b):
    return (a.charge()/a.pt()-b.charge()/b.pt())*b.pt()/b.charge()

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


def log(event,counter,mystubs,kmtf,bmtf):
    print("--------EVENT"+str(counter)+"------------")
    print('RUN={run} LUMI={lumi} EVENT={event}'.format(run=event.eventAuxiliary().id().run(),lumi=event.eventAuxiliary().id().luminosityBlock(),event=event.eventAuxiliary().id().event()))
    print("-----------------------------")
    print("-----------------------------")
    print('Stubs:')
    for stub in mystubs:
        print('wheel={w} sector={sc} station={st} high/low={ts} phi={phi} phiB={phiB} qual={qual} BX={BX}'.format(w=stub.whNum(),sc=stub.scNum(),st=stub.stNum(),ts=stub.Ts2Tag(),phi=stub.phi(),phiB=stub.phiB(),qual=stub.code(),BX=stub.bxNum()))
    print('EMU:')
    for g in bmtf :
        print("EMU sector={sector} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF}".format(sector=g.processor(), pt=g.hwPt(),eta=g.hwEta(),phi=g.hwPhi(),qual=g.hwQual(),dxy=g.hwDXY(),pt2=g.hwPtUnconstrained(),HF=g.hwHF()))
    print('DATA:')
    for g in kmtf :
        print("DATA sector={sector} pt={pt} eta={eta} phi={phi} qual={qual} dxy={dxy} pt2={pt2} hasFineEta={HF}".format(sector=g.processor(),pt=g.hwPt(),eta=g.hwEta(),phi=g.hwPhi(),qual=g.hwQual(),dxy=g.hwDXY(),pt2=g.hwPtUnconstrained(),HF=g.hwHF()))
    print("-----------------------------")
    print("-----------------------------")
    print("c + enter to continue")
    import pdb;pdb.set_trace()

###############################

#########Histograms#############
histos={}
histos['fw']={}
histos['fw']['pt1']=ROOT.TH1D("fw_pt1","HW p_{T}",512,0,511)
histos['fw']['eta1']=ROOT.TH1D("fw_eta1","HW #eta",256,-127,128)
histos['fw']['phi1']=ROOT.TH1D("fw_phi1","HW #phi",256,-127,128)
histos['fw']['HF1']=ROOT.TH1D("fw_HF1","HW HF",256,-127,128)
histos['fw']['qual1']=ROOT.TH1D("fw_qual1","HW qual",16,0,16)
histos['fw']['dxy1']=ROOT.TH1D("fw_dxy1","HW DXY",4,0,4)
histos['fw']['ptSTA1']=ROOT.TH1D("fw_ptSTA1","HW STA PT",256,0,255)

histos['fw']['pt2']=ROOT.TH1D("fw_pt2","HW p_{T}",512,0,511)
histos['fw']['eta2']=ROOT.TH1D("fw_eta2","HW #eta",256,-127,128)
histos['fw']['phi2']=ROOT.TH1D("fw_phi2","HW #phi",256,-127,128)
histos['fw']['HF2']=ROOT.TH1D("fw_HF2","HW HF",256,-127,128)
histos['fw']['qual2']=ROOT.TH1D("fw_qual2","HW qual",16,0,16)
histos['fw']['dxy2']=ROOT.TH1D("fw_dxy2","HW DXY",4,0,4)
histos['fw']['ptSTA2']=ROOT.TH1D("fw_ptSTA2","HW STA PT",256,0,255)

histos['fw']['pt3']=ROOT.TH1D("fw_pt3","HW p_{T}",512,0,511)
histos['fw']['eta3']=ROOT.TH1D("fw_eta3","HW #eta",256,-127,128)
histos['fw']['phi3']=ROOT.TH1D("fw_phi3","HW #phi",256,-127,128)
histos['fw']['HF3']=ROOT.TH1D("fw_HF3","HW HF",256,-127,128)
histos['fw']['qual3']=ROOT.TH1D("fw_qual3","HW qual",16,0,16)
histos['fw']['dxy3']=ROOT.TH1D("fw_dxy3","HW DXY",4,0,4)
histos['fw']['ptSTA3']=ROOT.TH1D("fw_ptSTA3","HW STA PT",256,0,255)



histos['emu']={}

histos['emu']['pt1']=ROOT.TH1D("emu_pt1","HW p_{T}",512,0,511)
histos['emu']['eta1']=ROOT.TH1D("emu_eta1","HW #eta",256,-127,128)
histos['emu']['phi1']=ROOT.TH1D("emu_phi1","HW #phi",256,-127,128)
histos['emu']['HF1']=ROOT.TH1D("emu_HF1","HW HF",256,-127,128)
histos['emu']['qual1']=ROOT.TH1D("emu_qual1","HW qual",16,0,16)
histos['emu']['dxy1']=ROOT.TH1D("emu_dxy1","HW DXY",4,0,4)
histos['emu']['ptSTA1']=ROOT.TH1D("emu_ptSTA1","HW STA PT",256,0,255)

histos['emu']['pt2']=ROOT.TH1D("emu_pt2","HW p_{T}",512,0,511)
histos['emu']['eta2']=ROOT.TH1D("emu_eta2","HW #eta",256,-127,128)
histos['emu']['phi2']=ROOT.TH1D("emu_phi2","HW #phi",256,-127,128)
histos['emu']['HF2']=ROOT.TH1D("emu_HF2","HW HF",256,-127,128)
histos['emu']['qual2']=ROOT.TH1D("emu_qual2","HW qual",16,0,16)
histos['emu']['dxy2']=ROOT.TH1D("emu_dxy2","HW DXY",4,0,4)
histos['emu']['ptSTA2']=ROOT.TH1D("emu_ptSTA2","HW STA PT",256,0,255)

histos['emu']['pt3']=ROOT.TH1D("emu_pt3","HW p_{T}",512,0,511)
histos['emu']['eta3']=ROOT.TH1D("emu_eta3","HW #eta",256,-127,128)
histos['emu']['phi3']=ROOT.TH1D("emu_phi3","HW #phi",256,-127,128)
histos['emu']['HF3']=ROOT.TH1D("emu_HF3","HW HF",256,-127,128)
histos['emu']['qual3']=ROOT.TH1D("emu_qual3","HW qual",16,0,16)
histos['emu']['dxy3']=ROOT.TH1D("emu_dxy3","HW DXY",4,0,4)
histos['emu']['ptSTA3']=ROOT.TH1D("emu_ptSTA3","HW STA PT",256,0,255)


for key,histo in histos['fw'].iteritems():
    histo.Sumw2()


def fill(info,mu):
    if len(mu)>0:
        info['pt1'].Fill(mu[0].hwPt())
        info['eta1'].Fill(mu[0].hwEta())
        info['phi1'].Fill(mu[0].hwPhi())
        info['HF1'].Fill(mu[0].hwHF())
        info['qual1'].Fill(mu[0].hwQual())
        info['dxy1'].Fill(mu[0].hwDXY())
        info['ptSTA1'].Fill(mu[0].hwPtUnconstrained())
    else:
        info['pt1'].Fill(0)
        info['eta1'].Fill(0)
        info['phi1'].Fill(0)
        info['HF1'].Fill(0)
        info['qual1'].Fill(0)
        info['dxy1'].Fill(0)
        info['ptSTA1'].Fill(0)

    if len(mu)>1:
        info['pt2'].Fill(mu[1].hwPt())
        info['eta2'].Fill(mu[1].hwEta())
        info['phi2'].Fill(mu[1].hwPhi())
        info['HF2'].Fill(mu[1].hwHF())
        info['qual2'].Fill(mu[1].hwQual())
        info['dxy2'].Fill(mu[1].hwDXY())
        info['ptSTA2'].Fill(mu[1].hwPtUnconstrained())
    else:
        info['pt2'].Fill(0)
        info['eta2'].Fill(0)
        info['phi2'].Fill(0)
        info['HF2'].Fill(0)
        info['qual2'].Fill(0)
        info['dxy2'].Fill(0)
        info['ptSTA2'].Fill(0)

    if len(mu)>2:
        info['pt3'].Fill(mu[2].hwPt())
        info['eta3'].Fill(mu[2].hwEta())
        info['phi3'].Fill(mu[2].hwPhi())
        info['HF3'].Fill(mu[2].hwHF())
        info['qual3'].Fill(mu[2].hwQual())
        info['dxy3'].Fill(mu[2].hwDXY())
        info['ptSTA3'].Fill(mu[2].hwPtUnconstrained())
    else:
        info['pt3'].Fill(0)
        info['eta3'].Fill(0)
        info['phi3'].Fill(0)
        info['HF3'].Fill(0)
        info['qual3'].Fill(0)
        info['dxy3'].Fill(0)
        info['ptSTA3'].Fill(0)







##############################

BUNCHES=[0]


events=Events([tag+'.root'])
counter=-1
for event in events:
    counter=counter+1
    #fetch stubs
    stubs=fetchStubsOLD(event,True)
    unpacker=fetchKMTF(event,100.0,'bmtfDigis:kBMTF')
    emulator=fetchKMTF(event,100.0,'simKBmtfDigis:BMTF')


    for processor in range(0,12):
        for bx in BUNCHES:
            emu=filter(lambda x: x.processor()==processor,emulator[bx])
            data=filter(lambda x: x.processor()==processor,unpacker[bx])
            if (len(emu)+len(data))>0:

                fill(histos['emu'],emu)
                fill(histos['fw'],data)
#                if len(emu)!=0 and len(data)==0:
#                    log(event,counter,stubs,data,emu)
#                    import pdb;pdb.set_trace()

f=ROOT.TFile("validationResults.root","RECREATE")
for key,histo in histos['fw'].iteritems():
    histo.SetMarkerStyle(7)
    histo.Write()
for key,histo in histos['emu'].iteritems():
    histo.SetLineColor(ROOT.kRed)
    histo.Write()


#make fancy plots
histonames=['pt1','eta1','phi1','HF1','qual1','dxy1','ptSTA1']

for h in histonames:
    c=ROOT.TCanvas(h)
    c.cd()
    histos['emu'][h].Draw("HIST")
    histos['emu'][h].GetXaxis().SetTitle(histos['emu'][h].GetTitle())
    histos['emu'][h].GetYaxis().SetTitle("events")
    histos['fw'][h].Draw("SAME")
    c.SetLogy()
    l=ROOT.TLegend(0.6,0.6,0.9,0.8)
    l.AddEntry(histos['emu'][h],"emulator","l")
    l.AddEntry(histos['fw'][h],"data","p")
    l.Draw()
    c.Write("plot_"+h)






f.Close()
