import ROOT

f=ROOT.TFile("gainLUTs.root")
f.cd()

fout=ROOT.TFile("packedGainLUTs.root","RECREATE")
def pack(f,fout,station,code):
    fout.cd()        
    newH=ROOT.TH1D("gain_{code}_{station}".format(station=station,code=code),"h",4*1024,0,4*1024)
    for N,i in enumerate([0,1,4,5]):
        h=f.Get("G_{code}_{station}_{i}".format(station=station,code=code,i=i))
        for j in range(1,h.GetNbinsX()+1):
            newH.SetBinContent(N*1024+j,h.GetBinContent(j))
    newH.Write()

def packV(f,fout,code):
    fout.cd()        
    newH=ROOT.TH1D("gain_{code}_0".format(code=code),"h",2*1024,0,2*1024)
    for N,i in enumerate([0,1]):
        h=f.Get("G_{code}_0_{i}".format(code=code,i=i))
        for j in range(1,h.GetNbinsX()+1):
            newH.SetBinContent(N*1024+j,h.GetBinContent(j))
    newH.Write()

pack(f,fout,3,15)
pack(f,fout,2,15)
pack(f,fout,1,15)
pack(f,fout,3,14)
pack(f,fout,2,14)
pack(f,fout,3,13)
pack(f,fout,1,13)
pack(f,fout,3,12)
pack(f,fout,2,11)
pack(f,fout,1,11)
pack(f,fout,2,10)
pack(f,fout,1,9)
pack(f,fout,2,7)
pack(f,fout,1,7)
pack(f,fout,2,6)
pack(f,fout,1,5)
pack(f,fout,1,3)




packV(f,fout,3)
packV(f,fout,5)
packV(f,fout,6)
packV(f,fout,7)
packV(f,fout,9)
packV(f,fout,10)
packV(f,fout,11)
packV(f,fout,12)
packV(f,fout,13)
packV(f,fout,14)
packV(f,fout,15)

fout.Close()
f.Close()
