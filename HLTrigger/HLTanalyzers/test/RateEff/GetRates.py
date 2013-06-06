import math,ROOT
from ROOT import gROOT, TFile, TChain, TTree, TH1F, TF1

gROOT.Reset()


#file = TFile("MinBiasOutput/hltmenu_8TeV_7.0e33_20130321_MinBias.root")
file = TFile("hltmenu_8TeV_7.0e33_20130513_QCD.root")

histInd = file.Get("individual")
histCum = file.Get("cumulative")

histNevt = file.Get("NEVTS")
nevt = histNevt.GetBinContent(1)


nfillb = 1331.
mfillb = 3564.
xtime = 50e-9
#xsec = 7.2700002e10*1e-36
xsec = 1033680.0*1e-36
ilumi = 7e33
collrate = (nfillb/mfillb)/xtime

def Rate(counts):
    rate = collrate * (1 - math.exp(-1* (xsec*ilumi*counts/nevt)/collrate))
    return rate

def RateErr(counts):
    rateerr = xsec * ilumi * ((math.sqrt(counts + ((counts)**2)/nevt))/nevt)
    return rateerr

nbins = histInd.GetNbinsX()

for b in xrange(1,nbins+1):
    Label = histInd.GetXaxis().GetBinLabel(b)
    CountInd = histInd.GetBinContent(b)
    CountCum = histCum.GetBinContent(b)

    RateInd = Rate(CountInd)
    RateIndErr = RateErr(CountInd)
    RateCum = Rate(CountCum)
    RateCumErr = RateErr(CountCum)

    print Label, "  ", RateInd, " +- ", RateIndErr, "  ", RateCum, " +- ", RateCumErr
