import ROOT

f=ROOT.TFile("gainLUTs.root")
f.cd()

fout=ROOT.TFile("packedGainLUTs.root","RECREATE")
def pack(f,fout,station,code):
    fout.cd()        
    newH=ROOT.TH1D("gain_{code}_{station}".format(station=station,code=code),"h",2*1024,0,2*1024)
    for N,i in enumerate([0,4]):
        h=f.Get("G_{code}_{station}_{i}".format(station=station,code=code,i=i))
        for j in range(1,h.GetNbinsX()+1):
            newH.SetBinContent(N*1024+j,h.GetBinContent(j))
    newH.Write()


def pack2(f,fout,station,code):
    fout.cd()        
    for q1 in ['H', 'L']:
        for q2 in ['H', 'L']:
            newH=ROOT.TH1D("gain2_{code}_{station}_{q1}{q2}".format(station=station,code=code,q1=q1,q2=q2),"h",4*512,0,4*512)
            for N,i in enumerate([0,1,4,5]):
                h=f.Get("G2_{code}_{station}_{i}_{q1}{q2}".format(station=station,code=code,i=i,q1=q1,q2=q2))
                for j in range(1,h.GetNbinsX()+1):
                    newH.SetBinContent(N*512+j,h.GetBinContent(j))
            newH.Write()


def packV(f,fout,code):
    fout.cd()        
    newH=ROOT.TH1D("gain_{code}_0".format(code=code),"h",2*1024,0,2*1024)
    for N,i in enumerate([0,1]):
        h=f.Get("G_{code}_0_{i}".format(code=code,i=i))
        for j in range(1,h.GetNbinsX()+1):
            newH.SetBinContent(N*1024+j,h.GetBinContent(j))
    newH.Write()

pack(f,fout,3,8)
pack(f,fout,2,8)
pack(f,fout,2,12)
pack(f,fout,2,4)
pack(f,fout,2,4)
pack(f,fout,1,14)
pack(f,fout,1,12)
pack(f,fout,1,10)
pack(f,fout,1,6)


pack2(f,fout,3,8)
pack2(f,fout,2,8)
pack2(f,fout,2,4)
pack2(f,fout,1,8)
pack2(f,fout,1,4)
pack2(f,fout,1,2)

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
