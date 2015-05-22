
#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from scaleBands import *
    
if __name__ == "__main__":
    from sys import argv
    fin = ROOT.TFile(argv[1])
    fbase  = dirname(argv[1])
    fout = ROOT.TFile(fbase+"/systTH.bands.root", "RECREATE")
    ROOT.gROOT.ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc")
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(20.,25.)
    c1 = ROOT.TCanvas("c1","c1")
    plots = [ "nJet25", ]
    if "3l" in argv[1]:
        plots += ["jet1pT","htJet25","htJet25ratio1224Lep","lepEta3","minDrllAFOS","bestMTopHad","finalMVA"]
    else:
        plots += ["lep2Pt" ,"lep2Eta" ,"drl2j" ,"mtW1" ,"htJet25" ,"mhtJet25","MVA_2LSS_4j_6var" ]
    for var in plots:
        #for L,N in ('', False), ('_norm',True):
        (L,N) = ('_norm',True)
        for P in ["TTH",]:
            scales = [ "tune"+X for X in "P11 ProQ20 Z2".split() ]
            bands = {}
            for LR,R in ('', False), ('_ratio',True):
                bandS = scaleBand(fin, var+"_"+P, True, norm=N, scales=scales, basePostfix="_tuneZ2Star", ymin=1e-3, relative=R)
                bandX = scaleBand(fin, var+"_"+P, True, norm=N, scales=scales+["tuneD6T"], basePostfix="_tuneZ2Star", ymin=1e-3, outPostfix="_wD6T", relative=R)
                if not bandS: continue
                bandX.SetFillColor(ROOT.kRed+1); bandS.SetFillStyle(1001);
                bandS.SetFillColor(ROOT.kAzure-9); bandS.SetFillStyle(1001);
                bands["S"+LR] = bandS
                bands["X"+LR] = bandX
                bandS.SetName("%s_%s_bands_%s%s"   %(var,P,L,LR)); fout.WriteTObject(bandS)
                bandX.SetName("%s_%s_bands_%s%s_X" %(var,P,L,LR)); fout.WriteTObject(bandX)
            if len(bands) != 4: continue
            bandN = bands["S"]
            bandN.Sort()
            xmin = bandN.GetX()[0]-bandN.GetErrorXlow(0)
            xmax = bandN.GetX()[bandN.GetN()-1]+bandN.GetErrorXhigh(bandN.GetN()-1)
            ymax = max([bandN.GetY()[i]+1.3*bandN.GetErrorYhigh(i) for i in xrange(bandN.GetN())])
            if var == "nJet25": xmin = 1.5 if "3l" in argv[1] else 3.5
            ## Prepare split screen
            c1 = ROOT.TCanvas("c1", "c1", 600, 750); c1.Draw()
            c1.SetWindowSize(600 + (600 - c1.GetWw()), (750 + (750 - c1.GetWh())));
            p1 = ROOT.TPad("pad1","pad1",0,0.31,1,0.99);
            p1.SetBottomMargin(0);
            p1.Draw();
            p2 = ROOT.TPad("pad2","pad2",0,0,1,0.31);
            p2.SetTopMargin(0);
            p2.SetBottomMargin(0.3);
            p2.SetFillStyle(0);
            p2.Draw();
            p1.cd();
            ## Draw absolute prediction in top frame
            frame = ROOT.TH1F("frame","frame",1,xmin,xmax)
            frame.GetYaxis().SetRangeUser(0,1.4*ymax)
            frame.GetXaxis().SetLabelOffset(999) ## send them away
            frame.GetXaxis().SetTitleOffset(999) ## in outer space
            frame.GetYaxis().SetLabelSize(0.05)
            frame.GetYaxis().SetLabelSize(0.05)
            frame.GetYaxis().SetDecimals(True)
            frame.GetYaxis().SetTitle("Event yield")
            frame.Draw()
            for X in "XS": 
                bands[X].Draw("E2 SAME")
            frame.Draw("AXIS SAME")
            ## Draw relaive prediction in the bottom frame
            p2.cd() 
            rframe = ROOT.TH1F("rframe","rframe",1,xmin,xmax)
            rframe.GetXaxis().SetTitle(bandN.GetXaxis().GetTitle())
            rframe.GetYaxis().SetRangeUser(0.65,1.45);
            rframe.GetXaxis().SetTitleSize(0.14)
            rframe.GetYaxis().SetTitleSize(0.14)
            rframe.GetXaxis().SetLabelSize(0.11)
            rframe.GetYaxis().SetLabelSize(0.11)
            rframe.GetXaxis().SetNdivisions(505 if var == "nJet25" else 510)
            rframe.GetYaxis().SetNdivisions(505)
            rframe.GetYaxis().SetDecimals(True)
            rframe.GetYaxis().SetTitle("Ratio")
            rframe.GetYaxis().SetTitleOffset(0.52);
            rframe.Draw()
            for X in "XS": 
                bands[X+"_ratio"].Draw("E2 SAME")
            line = ROOT.TLine(xmin,1,xmax,1)
            line.SetLineWidth(2);
            line.SetLineStyle(7);
            line.SetLineColor(1);
            line.Draw("L")
            rframe.Draw("AXIS SAME")
            p1.cd()
            leg = doLegend(.20,.77,.92,.91, textSize=0.046)
            leg.AddEntry(bands["S"], "Envelop of Z2*, Z2, ProQ20, P11", "F")
            leg.AddEntry(bands["X"], "Envelop including also D6T tune", "F")
            leg.Draw()
            leg.Draw()
            c1.cd()
            doCMSSpam("CMS Simulation",textSize=0.035)
            c1.Print(fbase+"/systTH_"+P+"_"+var+".png")
            c1.Print(fbase+"/systTH_"+P+"_"+var+".pdf")
            del leg
            del frame
            del rframe
            del c1
    fout.Close()

