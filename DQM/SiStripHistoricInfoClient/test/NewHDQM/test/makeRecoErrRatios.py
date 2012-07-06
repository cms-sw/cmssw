from ROOT import *
import sys

top = []
bot = []
##I need to simply read in the appropriate web directory.
##From here I can open the root file...
dir = sys.argv[1]
if dir[-1] != '/': dir += '/'
file = TFile(dir+"RECOError_plots_all_plots.root", "UPDATE")
print dir+ "RECOError_plots_all_plots.root"
for key in file.GetListOfKeys():
    print  key.GetName()
    if key.GetClassName() != "TH1F": continue
    if "denominator" in key.GetName(): bot.append(key.GetName())
    else: top.append(key.GetName())
##Load Denominator histo
bkey = file.GetListOfKeys().FindObject(bot[0])
bplot = bkey.ReadObj();
if len(bot) ==1 and bplot.GetEntries() > 0:
    for t in top:
        if len(bot) != 1: continue
        if "Ratio" in t: continue
        ##Load Numerator histo
        tkey = file.GetListOfKeys().FindObject(t)
        tplot = tkey.ReadObj();
        if not tplot.GetSumOfWeights() > 0.0: continue
        ##Make new TCanvas to print out ratios
        C=TCanvas("C","C",600,600)
        ##Clone and setup ratio histo
        ratio = tplot.Clone(t.split("_")[1]+"_Ratio")
        ratio.SetName(t.split("_")[1]+"_Ratio")
        ratio.SetTitle("Fraction of Errors in "+t.split("_")[1])
        ##Make ratio, then print + save plots
        ratio.Divide(bplot)
        if ratio.Integral(int((ratio.GetNbinsX()*3)/4), ratio.GetNbinsX())/ratio.Integral(1, ratio.GetNbinsX()) > 0.90:
            C.SetFrameLineWidth(10)
            C.SetFrameLineColor(2)
        print ratio.Integral(int((ratio.GetNbinsX()*3)/4), ratio.GetNbinsX())/ratio.Integral(1, ratio.GetNbinsX())
        ##Now temporarily change the bins shown on X axis...
        for i in range(1,pfTrackElec_Ratio->GetNbinsX()+1):
            if i%int(145*0.03) !=  0: ratio.GetXaxis().SetBinLabel(i,"")
        C.cd()
        ratio.Draw()
        C.SaveAs(dir+t.split("_")[1]+"_Ratio.png")
        C.Close()
        ##Now change bin labels back
        for i in range(1,pfTrackElec_Ratio->GetNbinsX()+1): ratio.GetXaxis().SetBinLabel(i,bplot.GetXaxis().GetBinLabel(i))
        file.cd()
        ratio.Write()
file.Close()
