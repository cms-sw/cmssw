import ROOT,itertools,math      
from array import array 
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()

def getBit(q,i):
    return ((1<<i) & q)>>i

def fetchKMTF(event,etaMax=0.83,chi2=800000,dxyCut=100000):
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

####Save the Kalman Gains for LUTs
kalmanGain={}
kalmanGain2H={}
kalmanGain2L={}
kalmanGain2={}
for track in [3,5,6,7,9,10,11,12,13,14,15]:
    for station in [3,2,1]:
        if getBit(track,station-1)==0:
            continue
        partialMask = track & (15<<station)
        if partialMask==0:
            continue;

        if track in [3,5,6,9,10,12]:
            if not (partialMask in kalmanGain2.keys()):
                kalmanGain2[partialMask] = {}
            if not (partialMask in kalmanGain2H.keys()):
                kalmanGain2H[partialMask] = {}
            if not (partialMask in kalmanGain2L.keys()):
                kalmanGain2L[partialMask] = {}
            kalmanGain2[partialMask][station]={}
            for q1 in ['H', 'L']:
                kalmanGain2[partialMask][station][q1]={}
                for q2 in ['H', 'L']:
                    kalmanGain2[partialMask][station][q1][q2]={}

                    kalmanGain2[partialMask][station][q1][q2][0]=ROOT.TH2D("gain2_{track}_{station}_0_{q1}{q2}".format(track=partialMask,station=station,q1=q1,q2=q2),"h",64,0,512,256*2,-100*2,100*2)
                    kalmanGain2[partialMask][station][q1][q2][1]=ROOT.TH2D("gain2_{track}_{station}_1_{q1}{q2}".format(track=partialMask,station=station,q1=q1,q2=q2),"h",64,0,512,256*4,-8*4,8*4)
                    kalmanGain2[partialMask][station][q1][q2][4]=ROOT.TH2D("gain2_{track}_{station}_4_{q1}{q2}".format(track=partialMask,station=station,q1=q1,q2=q2),"h",64,0,512,256*2,-15*2,0)
                    kalmanGain2[partialMask][station][q1][q2][5]=ROOT.TH2D("gain2_{track}_{station}_5_{q1}{q2}".format(track=partialMask,station=station,q1=q1,q2=q2),"h",64,0,512,256*2,0,1*2)

        else:
            if not (partialMask in kalmanGain.keys()):
                kalmanGain[partialMask] = {}
            kalmanGain[partialMask][station] = {}
            kalmanGain[partialMask][station][0]=ROOT.TH2D("gain_{track}_{station}_0".format(track=partialMask,station=station),"h",64,0,1024,256,-100,100)
            kalmanGain[partialMask][station][4]=ROOT.TH2D("gain_{track}_{station}_4".format(track=partialMask,station=station),"h",64,0,1024,256,-15,0)


    for station in [0]:
        if not (track in kalmanGain.keys()):
            kalmanGain[track] = {}
        kalmanGain[track][0]={}
        kalmanGain[track][0][0]=ROOT.TH2D("gain_{track}_0_0".format(track=track),"h",64,0,1024,128,-5,5)
        kalmanGain[track][0][1]=ROOT.TH2D("gain_{track}_0_1".format(track=track),"h",64,0,1024,128,-5,5)


##############################




for p in [3,5,6,7,9,10,11,12,13,14,15]:
    events=Events(['singleMu0_{}.root'.format(p)]) # Run KBMTF with only 1 pattern at a time to increase statistics
    counter=-1
    for event in events:
        counter=counter+1
        #fetch stubs
        kmtf=[]
        kmtf = fetchKMTF(event,1.5,1000000,1000000)
        ##Fill histograms and rates
        for track in kmtf:
            mask = track.hitPattern()
            qual = {}
            q1 = 'L'
            for stub in track.stubs():
                qual[stub.stNum()] = stub.quality()
            if qual[max(qual)] >= 4:
                q1='H'
            for station in [3,2,1]:
                if not getBit(mask,station-1):
                    continue
                gain = track.kalmanGain(station)
                partialMask = mask & (15<<station)
                if partialMask==0:
                    continue
                q2 = 'L'
                if qual[station] >= 4:
                    q2 = 'H'
                if mask in [3,5,6,9,10,12]:
                    for element in [0,1,4,5]:
                        kalmanGain2[partialMask][station][q1][q2][element].Fill(gain[0]/8,gain[element+1])
                else:        
                    for element in [0,4]:
                        kalmanGain[partialMask][station][element].Fill(gain[0]/4,gain[element+1])

            for station in [0]:
                gain = track.kalmanGain(station)
                kalmanGain[mask][station][0].Fill(gain[0]/2,gain[1])
                kalmanGain[mask][station][1].Fill(gain[0]/2,gain[2])

        
f=ROOT.TFile("gains.root","RECREATE")


for k in kalmanGain.keys():
    for s in kalmanGain[k].keys():
        for e in kalmanGain[k][s].keys():
            kalmanGain[k][s][e].Write()

for k in kalmanGain2.keys():
    for s in kalmanGain2[k].keys():
        for q1 in kalmanGain2[k][s].keys():
            for q2 in kalmanGain2[k][s][q1].keys():
                for e in kalmanGain2[k][s][q1][q2].keys():
                    kalmanGain2[k][s][q1][q2][e].Write()

f.Close()








