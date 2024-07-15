import ROOT
from math import pi
f=ROOT.TFile("../data/packedGainLUTs.root")
lut=f.Get("coarseETALUT")
l=[]
for i in range(0,256):
    b = lut.GetXaxis().FindBin(i)
    c=lut.GetBinContent(b)
    l.append(str(int((1<<12)*c/pi)))

print("const ap_uint<BITSSTAMUONETA> coarseEtaLUT[256] ={"+','.join(l)+'};')
    
f.Close()
