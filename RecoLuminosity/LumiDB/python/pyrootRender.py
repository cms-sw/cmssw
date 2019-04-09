from __future__ import print_function
from builtins import range
import sys
import ROOT
from ROOT import TCanvas,TH1F,gROOT,TFile,gStyle,gDirectory,TDatime,TLegend

batchonly=False
try:
    import Tkinter as Tk
    root=Tk.Tk()
except ImportError:
    print('unable to import GUI backend, switch to batch only mode')
    batchonly=True

def destroy(e) :
    sys.exit()
        
class interactiveRender(Tk.Frame):
    def __init__(self):
        Tk.Frame.__init__(self,master=root)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gROOT.SetBatch(ROOT.kFALSE)
        self.__canvas=TCanvas("Luminosity","",1)
        self.__canvas.SetHighLightColor(2);
        self.__canvas.Range(-125.6732,-0.1364721,1123.878,1.178117)
        self.__canvas.SetFillColor(0)
        self.__canvas.SetBorderMode(0)
        self.__canvas.SetBorderSize(2)
        self.__canvas.SetGridx()
        self.__canvas.SetGridy()
        self.__canvas.SetFrameFillColor(19)
        self.__canvas.SetFrameBorderMode(0)
        self.__canvas.SetFrameBorderMode(0)
    def draw(self,rootobj):
        rootobj.Draw()
        self.pack()
        button=Tk.Button(master=root,text='Quit',command=sys.exit)
        button.pack(side=Tk.BOTTOM)
        Tk.mainloop()
class batchRender():
    def __init__(self,outputfilename):
        ROOT.gStyle.SetOptStat(0)
        ROOT.gROOT.SetBatch(ROOT.kTRUE)
        self.__canvas=TCanvas("Luminosity","",1)
        self.__canvas.SetHighLightColor(2);
        self.__canvas.Range(-125.6732,-0.1364721,1123.878,1.178117)
        self.__canvas.SetFillColor(0)
        self.__canvas.SetBorderMode(0)
        self.__canvas.SetBorderSize(2)
        self.__canvas.SetGridx()
        self.__canvas.SetGridy()
        self.__canvas.SetFrameFillColor(19)
        self.__canvas.SetFrameBorderMode(0)
        self.__canvas.SetFrameBorderMode(0)
        self.__outfile=outputfilename
    def draw(self,rootobj):
        rootobj.Draw()
        self.__canvas.Modified()
        self.__canvas.cd()
        self.__canvas.SetSelected(rootobj)
        self.__canvas.SaveAs(self.__outfile)
if __name__=='__main__':
      
    da = TDatime(2010,3,30,13,10,00)
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
    
    #m=interactiveRender()
    #m.draw(h1f)
    bat=batchRender('testroot.jpg')
    bat.draw(h1f)
