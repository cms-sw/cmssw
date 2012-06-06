from ROOT import *

top = []
bot = []
file = TFile("recoErrors.root","UPDATE")
#file  = TFile.Open("recoErrors.root")
for key in file.GetListOfKeys():
    if key.GetClassName() != "TH1F": continue
    if "denominator" in key.GetName(): bot.append(key.GetName())
    else: top.append(key.GetName())
for t in top:
    if len(bot) != 1: continue
    if "Ratio" in t: continue
    ##Load Denominator histo
    bkey = file.GetListOfKeys().FindObject(bot[0])
    bplot = bkey.ReadObj();
    ##Load Numerator histo
    tkey = file.GetListOfKeys().FindObject(t)
    tplot = tkey.ReadObj();
    ##Make ratio histo
    ratio = tplot.Clone(t.split("_")[1]+"_Ratio")
    ratio.SetTitle("Fraction of Errors in "+t.split("_")[1])
    ratio.GetYaxis().SetTitle(t.split("_")[1])
    ratio.Divide(bplot)
##Write it out
file.Write()
file.Close()
