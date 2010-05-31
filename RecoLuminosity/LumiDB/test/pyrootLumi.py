#! /usr/bin/env python
import commands
import sys, os, string, fileinput
import Tkinter as tk
import ROOT
from ROOT import TCanvas,TH1F,gROOT,TFile,gStyle,gDirectory,TDatime, TLegend
master=tk.Tk()

def destroy(e) :
    sys.exit()

class Zhen(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self, master=master)
        ROOT.gROOT.SetBatch(ROOT.kFALSE)
        #ROOT.gROOT.SetBatch(ROOT.kTRUE)
        ROOT.gStyle.SetOptStat(0)
        
        da = TDatime(2010,03,30,13,10,00)

        c = TCanvas("Luminosity","",1)
        
        c.SetHighLightColor(2);
        c.Range(-125.6732,-0.1364721,1123.878,1.178117)
        c.SetFillColor(0)
        c.SetBorderMode(0)
        c.SetBorderSize(2)
        c.SetGridx()
        c.SetGridy()
        c.SetFrameFillColor(19)
        c.SetFrameBorderMode(0)
        c.SetFrameBorderMode(0)
        
        
        h1f = TH1F("Luminposity","",1000,0.,1000)
        h1f.GetXaxis().SetNdivisions(-503)
        h1f.GetXaxis().SetTimeDisplay(1)
        h1f.GetXaxis().SetTimeFormat("%d\/%m %H:%M")
        h1f.GetXaxis().SetTimeOffset(da.Convert())       
        h1f.GetXaxis().SetLabelFont(32);
        h1f.GetXaxis().SetLabelSize(0.03);
        h1f.GetXaxis().SetTitleFont(32);
        h1f.GetXaxis().SetTitle("Date");
        
        h1f.GetYaxis().SetLabelFont(32);
        h1f.GetYaxis().SetLabelSize(0.03);
        h1f.GetYaxis().SetTitleFont(32);
        h1f.GetYaxis().SetTitle("L (#mub^{-1})");

 


        for i in range(0,1000):
            #h1f.GetXaxis().FindBin() ## Ricordati di calcolare il bin corretto per il tuo tempo
            h1f.SetBinContent(i,20.2+i)
        
        h1f.Draw()

        h1f2 = TH1F("Luminposity 2","",1000,0.,1000)
        h1f2.GetXaxis().SetNdivisions(-503)
        h1f2.GetXaxis().SetTimeDisplay(1)
        h1f2.GetXaxis().SetTimeFormat("%d\/%m %H:%M")
        h1f2.GetXaxis().SetTimeOffset(da.Convert())       
        h1f2.GetXaxis().SetLabelSize(0.03)


        for i in range(0,1000):
            h1f2.SetBinContent(i,20.2-i)

        h1f2.SetLineColor(ROOT.kRed)
        h1f2.Draw("same")

        leg = TLegend(0.1537356,0.6631356,0.5344828,0.875)
        leg.SetTextFont(32);
        leg.SetFillColor(0);
        leg.SetFillStyle(1001);
        
        leg.AddEntry(h1f,"Delibvered 1.54 nb^{-1}","l")
        leg.AddEntry(h1f2,"Delibvered 1.59 nb^{-1}","l")
        leg.Draw();
        
        #c.Modified();
        #c.cd();
        #c.SetSelected(h1f);
        
        #c.SaveAs("Zhen.jpg")
        
        self.pack()
        button = tk.Button(master=master,text='Quit',command=sys.exit)
        button.pack(side=tk.BOTTOM)
        tk.mainloop()
if __name__ == '__main__':
    Zhen()
