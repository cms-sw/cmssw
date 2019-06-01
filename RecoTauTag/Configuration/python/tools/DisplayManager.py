from builtins import range
import ROOT
import copy
import math

def applyLegendSettings(leg):
    leg.SetBorderSize(0)
    leg.SetFillColor(10)
    leg.SetLineColor(0)
    leg.SetFillStyle(0)
#   leg.SetTextSize(0.035)
#   leg.SetTextFont(42)

def createRatioCanvas(name, errorBandFillColor=14, errorBandStyle=3354):
    cv = ROOT.TCanvas(name.replace('.pdf', ''), name.replace('.pdf', ''), 10, 10, 700, 600)
    ## this is the tricky part...
    ## Divide with correct margins
    cv.Divide(1, 2, 0.0, 0.0)
    ## Set Pad sizes
    cv.GetPad(1).SetPad(0.0, 0.32, 1., 1.0)
    cv.GetPad(2).SetPad(0.0, 0.00, 1., 0.34)
    cv.GetPad(1).SetFillStyle(4000)
    cv.GetPad(2).SetFillStyle(4000)
    ## Set pad margins 1
    cv.cd(1)
    ROOT.gPad.SetTopMargin(0.08)
    ROOT.gPad.SetLeftMargin(0.12)
    ROOT.gPad.SetBottomMargin(0.03)
    ROOT.gPad.SetRightMargin(0.1)
    ## Set pad margins 2
    cv.cd(2)
    ROOT.gPad.SetBottomMargin(0.35)
    ROOT.gPad.SetLeftMargin(0.12)
    ROOT.gPad.SetRightMargin(0.1)

    bogyHist = ROOT.TH1F("legendPseudoHist", "", 1, 1., 2.)
    bogyHist.SetFillColor(errorBandFillColor)
    bogyHist.SetFillStyle(errorBandStyle)
    bogyHist.SetLineColor(0)
    
    cv.cd(1)
    return cv

def checkDifferences(histos):
    rel_diff = 0
    if len(histos)>0:
        for ib in range(0, histos[0].GetNbinsX()):
            for ih, h in enumerate(histos):
                if not ih == 0:
                    if not histos[0].GetBinContent(ib+1) == 0:
                        rel_diff+=((h.GetBinContent(ib+1)-histos[0].GetBinContent(ib+1))/histos[0].GetBinContent(ib+1))*((h.GetBinContent(ib+1)-histos[0].GetBinContent(ib+1))/histos[0].GetBinContent(ib+1))
    return math.sqrt(rel_diff)

class DisplayManager(object):
    def __init__(self, name, ratio):
        if ratio:
            self.canvas = createRatioCanvas(name.replace('pdf', ''))
        else:
            self.canvas = ROOT.TCanvas(name.replace('.pdf', ''))
        self.name = name
        self.draw_ratio = ratio
        self.histos = []
        self.Legend = ROOT.TLegend(0.15, 0.79, 0.5, 0.89)
        applyLegendSettings(self.Legend)
        self.draw_ratioLegend = ROOT.TLegend(0.15, 0.79, 0.5, 0.89)
        applyLegendSettings(self.draw_ratioLegend)
        self.pullRange = 0.5
        self.box = ROOT.TPave(0.93,0.85,0.98,0.90)
        self.canvas.Print(self.name + '[')
        
    def __del__(self):
        self.canvas.Print(self.name + ']')

    def Draw(self, histos, titles):
        self.histos = histos
        ymax = max(h.GetMaximum() for h in self.histos)
        self.Legend.Clear()
        self.draw_ratioLegend.Clear()
        for i, h in enumerate(self.histos):
            title = titles[i]
            h.GetYaxis().SetRangeUser(0., ymax * 1.3)
            self.Legend.AddEntry(h, title + ': ' + str(h.Integral()))
            if i == 0:
                h.Draw('HIST E')
            else:
                h.Draw('SAME HIST E')
        self.Legend.Draw()
        ## add an assessment on the agreement
        rel_diff = checkDifferences(self.histos)
        self.box.SetLineColor(1)
        self.box.SetLineWidth(1)
        self.box.SetShadowColor(0)
        if rel_diff <= 0.01:
            self.box.SetFillColor(416) ## kGreen
        elif rel_diff <= 0.10:
            self.box.SetFillColor(797) ## kOrange
        else:
            self.box.SetFillColor(632) ## kRed
        self.box.ConvertNDCtoPad()
        self.box.Draw()
        
        pull_histos = []
        if self.draw_ratio:
            self.canvas.cd(2)
            for ihist in range(1, len(self.histos)):
                histPull = copy.deepcopy(self.histos[ihist])
                pull_histos.append(histPull)
                histPull.Divide(self.histos[0])
                histPull.UseCurrentStyle()
                histPull.SetLineColor(self.histos[ihist].GetLineColor())
                histPull.SetMarkerColor(self.histos[ihist].GetLineColor())
                histPull.SetLineStyle(self.histos[ihist].GetLineStyle())
                histPull.SetLineWidth(self.histos[ihist].GetLineWidth())
                histPull.GetYaxis().SetRangeUser(-self.pullRange + 1., self.pullRange + 1.)
                # defaultYtoPixel = 408.  # height in pixels of default canvas
                defaultYtoPixel = self.canvas.GetPad(1).YtoPixel(0.)
                pad2YtoPixel = float(self.canvas.GetPad(2).YtoPixel(0))
                pad2XaxisFactor = defaultYtoPixel / pad2YtoPixel
                histPull.GetXaxis().SetLabelSize(self.histos[0].GetXaxis().GetLabelSize()*pad2XaxisFactor)
                histPull.GetXaxis().SetLabelOffset(self.histos[0].GetXaxis().GetLabelOffset()*pad2XaxisFactor)
                histPull.GetXaxis().SetTitleSize(self.histos[0].GetXaxis().GetTitleSize()*pad2XaxisFactor)
                histPull.GetXaxis().SetTitleOffset(self.histos[0].GetXaxis().GetTitleOffset()/pad2XaxisFactor*2.5)
                histPull.GetYaxis().SetLabelSize(self.histos[0].GetYaxis().GetLabelSize()*pad2XaxisFactor)
                histPull.GetYaxis().SetLabelOffset(self.histos[0].GetYaxis().GetLabelOffset()*pad2XaxisFactor)
                histPull.GetYaxis().SetTitleSize(self.histos[0].GetYaxis().GetTitleSize()*pad2XaxisFactor)
                histPull.GetYaxis().SetTitleOffset(self.histos[0].GetYaxis().GetTitleOffset()/pad2XaxisFactor)
                histPull.GetYaxis().CenterTitle()
                histPull.GetXaxis().SetTickLength(histPull.GetXaxis().GetTickLength()*pad2XaxisFactor)
                histPull.GetYaxis().SetNdivisions(306)
                histPull.GetYaxis().SetTitle("Ratio to " + titles[0])
                histPull.SetTitle('')
                if ihist == 1:
                    histPull.Draw("ep")
                else:
                    histPull.Draw("same ep")
                self.draw_ratioLegend.AddEntry(histPull, titles[ihist])
            # This is a little bit ugly though ...
            for i, h in enumerate(self.histos):
                h.GetXaxis().SetLabelSize(0)
            self.canvas.cd(1)
        self.canvas.Update()
        self.canvas.Print(self.name)
