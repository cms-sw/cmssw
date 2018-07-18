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
for track in [3,5,6,7,9,10,11,12,13,14,15]:
    kalmanGain[track]={}
    for station in [4,3,2,1,0]:
        kalmanGain[track][station]={}
        if station>0:
            kalmanGain[track][station][0]=ROOT.TH2D("gain_{track}_{station}_0".format(track=track,station=station),"h",64,0,1024,128,-100,100)
            kalmanGain[track][station][1]=ROOT.TH2D("gain_{track}_{station}_1".format(track=track,station=station),"h",64,0,1024,128,-8,8)
            kalmanGain[track][station][4]=ROOT.TH2D("gain_{track}_{station}_4".format(track=track,station=station),"h",64,0,1024,128,-15,0)
            kalmanGain[track][station][5]=ROOT.TH2D("gain_{track}_{station}_5".format(track=track,station=station),"h",64,0,1024,128,0,1)
        else:    
            kalmanGain[track][station][0]=ROOT.TH2D("gain_{track}_{station}_0".format(track=track,station=station),"h",128,0,1024,128,-5,5)
            kalmanGain[track][station][1]=ROOT.TH2D("gain_{track}_{station}_1".format(track=track,station=station),"h",128,0,1024,128,-5,5)

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
        mask = track.hitPattern();
        for station in [3,2,1,0]:
            if station!=0 and getBit(mask,station-1)==0:
                continue
            

            gain = track.kalmanGain(station)

            #check if we are at the second hit. If yes use the phiB of the first stub  
            countOuter=0;
            for outerStation in range(station+1,5):
                if getBit(mask,outerStation-1)==1:
                    countOuter=countOuter+1;


            if countOuter==0:
                continue;
            if countOuter==1:
                for element in [0,1,4,5]:
                    kalmanGain[mask][station][element].Fill(gain[0]/4,gain[element+1])
            else:
                if station==0:
                    for element in [0,1]:
                        kalmanGain[mask][station][element].Fill(gain[0]/2,gain[element+1])
                else:
                    for element in [0,1,4,5]:
                        kalmanGain[mask][station][element].Fill(gain[0]/4,gain[element+1])

        
f=ROOT.TFile("gains.root","RECREATE")

for track in [3,5,6,7,9,10,11,12,13,14,15]:
    for station in [4,3,2,1,0]:
        if station>0:
            kalmanGain[track][station][0].Write()
            kalmanGain[track][station][1].Write()
            kalmanGain[track][station][4].Write()
            kalmanGain[track][station][5].Write()

        else:    
            kalmanGain[track][station][0].Write()
            kalmanGain[track][station][1].Write()


f.Close()








