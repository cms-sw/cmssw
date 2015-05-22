from ROOT import gDirectory, TLegend,gPad, TCanvas, TH1F

from CMGTools.RootTools.Style import *

class HistogramComparison:
    def __init__(self, name, title, bins = 100, min = 0, max = 1000):
        self.name = name
        self.h1 = TH1F('h1_'+name, title, bins,min,max)
        self.h2 = TH1F('h2_'+name, title, bins,min,max)
        self.h2 = TH1F('h3_'+name, title, bins,min,max)
        self.eff = None
        self.canSup = TCanvas('canSup'+name, 'canSup'+name)
        self.canEff = TCanvas('canEff'+name, 'canEff'+name)
        self.canSup.SetLogy()
        self.canEff.SetLogy()
        # self.h1.SetLineWidth(2)
        sBlack.formatHisto( self.h1 )
        sBlueSquares.formatHisto( self.h2 )
        self.histoFinalized = False
    def reset(self):
        self.h1.Reset()
        self.h2.Reset()
        self.eff.Reset()
    def computeEff(self):
        self.eff = self.h2.Clone( 'eff_'+ self.name)
        self.eff.GetYaxis().SetTitle('efficiency')
        # self.eff.Divide( self.h1 )
        self.eff.Divide( self.eff, self.h1, 1, 1, 'B')
        self.eff.SetStats(0)
        return self.eff
    def setUpLegend(self, caption1 = 'all events', caption2='selected', xmin=None, ymin=None, xmax=None,ymax=None ):
        if xmin==None:
            xmin = 0.5
        if ymin==None:
            ymin = 0.5
        if xmax==None:
            xmax = 0.85
        if ymax==None:
            ymax = 0.8
        self.legend = TLegend(xmin,ymin,xmax,ymax)
        self.legend.AddEntry(self.h1,caption1)
        self.legend.AddEntry(self.h2,caption2)
    def finalizeHistograms(self):
        self.h1.GetYaxis().SetRangeUser(0.1, self.h1.GetEntries() )
        self.h1.Sumw2()
        self.h2.Sumw2()
        self.computeEff()
        self.histoFinalized = True
    def draw(self):
        if not self.histoFinalized:
            self.finalizeHistograms()
        self.canSup.cd()
        self.h1.Draw()
        self.h2.Draw('same')   
        if self.legend == None:
            self.setUpLegend()
        self.legend.Draw()
        gPad.RedrawAxis()
        self.canSup.SaveAs(self.canSup.GetName()+'.png')
        self.canEff.cd()
        # if self.eff == None:
        #     self.computeEff()
        self.eff.GetYaxis().SetRangeUser(0.0001,2)
        self.eff.Draw()
        self.canEff.SaveAs(self.canEff.GetName()+'.png')
    def savePlots(self):
        self.canSup.SaveAs(self.canSup.GetName()+'.png')
        self.canEff.SaveAs(self.canEff.GetName()+'.png')
