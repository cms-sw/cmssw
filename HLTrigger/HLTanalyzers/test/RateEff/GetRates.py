import math,ROOT
from ROOT import gROOT, TFile, TChain, TTree, TH1F, TF1

gROOT.Reset()


#file = TFile("MinBiasOutput/hltmenu_8TeV_7.0e33_20130321_MinBias.root")
file = TFile("hltmenu_8TeV_5.0e32_20130801Counts.root")

histInd = file.Get("individual")
histCum = file.Get("cumulative")

histNevt = file.Get("NEVTS")
nevt = histNevt.GetBinContent(1)


nfillb = 312.
mfillb = 3564.
xtime = 75e-9
#xsec = 7.2700002e10*1e-36
xsec = (7.13E10)*1e-36
ilumi = 5e32
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
