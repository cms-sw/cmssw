import ROOT,itertools,math      
from array import array 
from DataFormats.FWLite import Events, Handle
ROOT.FWLiteEnabler.enable()

def getBit(q,i):
    return ((1<<i) & q)>>i


def fetchKMTF(event,etaMax=0.83,chi2=800000,dxyCut=100000):
    kmtfH  = Handle('vector<L1MuKBMTrack>')
    event.getByLabel('simKBmtfDigis',kmtfH)
    kmtf=[x for x in kmtfH.product() if abs(x.eta())<etaMax and x.approxChi2()<chi2 and abs(x.dxy())<dxyCut]
    return sorted(kmtf,key=lambda x: x.pt(),reverse=True)

####Save the Kalman Gains for LUTs
kalmanGain={}
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
            kalmanGain2[partialMask][station] = {}

            kalmanGain2[partialMask][station][0]=ROOT.TH2D("gain2_{track}_{station}_0".format(track=partialMask,station=station),"h",64,0,512,128,-100,100)
            kalmanGain2[partialMask][station][1]=ROOT.TH2D("gain2_{track}_{station}_1".format(track=partialMask,station=station),"h",64,0,512,128,-8,8)
            kalmanGain2[partialMask][station][4]=ROOT.TH2D("gain2_{track}_{station}_4".format(track=partialMask,station=station),"h",64,0,512,128,-15,0)
            kalmanGain2[partialMask][station][5]=ROOT.TH2D("gain2_{track}_{station}_5".format(track=partialMask,station=station),"h",64,0,512,128,0,1)
        else:
            if not (partialMask in kalmanGain.keys()):
                kalmanGain[partialMask] = {}
            kalmanGain[partialMask][station] = {}
            kalmanGain[partialMask][station][0]=ROOT.TH2D("gain_{track}_{station}_0".format(track=partialMask,station=station),"h",64,0,1024,128,-100,100)
            kalmanGain[partialMask][station][4]=ROOT.TH2D("gain_{track}_{station}_4".format(track=partialMask,station=station),"h",64,0,1024,128,-15,0)


    for station in [0]:
        if not (track in kalmanGain.keys()):
            kalmanGain[track] = {}
        kalmanGain[track][0]={}
        kalmanGain[track][0][0]=ROOT.TH2D("gain_{track}_0_0".format(track=track),"h",64,0,1024,128,-5,5)
        kalmanGain[track][0][1]=ROOT.TH2D("gain_{track}_0_1".format(track=track),"h",64,0,1024,128,-5,5)


##############################





events=Events(['lutEvents.root'])
counter=-1
for event in events:
    counter=counter+1
    #fetch stubs
    kmtf=[]
    kmtf = fetchKMTF(event,1.5,1000000,1000000)
    ##Fill histograms and rates
    for track in kmtf:
        mask = track.hitPattern()
        for station in [3,2,1]:
            if not getBit(mask,station-1):
                continue
            gain = track.kalmanGain(station)
            partialMask = mask & (15<<station)
            if partialMask==0:
                continue
            if mask in [3,5,6,9,10,12]:
                for element in [0,1,4,5]:

                    kalmanGain2[partialMask][station][element].Fill(gain[0]/8,gain[element+1])
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
        for e in kalmanGain2[k][s].keys():
            kalmanGain2[k][s][e].Write()



f.Close()








