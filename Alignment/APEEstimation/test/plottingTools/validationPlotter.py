import ROOT
ROOT.gROOT.SetBatch(True)
from setTDRStyle import setTDRStyle


from granularity import *


class ValidationPlotter:
    def __init__(self):
        setTDRStyle()
        self.inFiles = {}
        self.labels = {}
        self.colors = {}
        self.markers = {}
        self.outPath = None
        self.granularity = standardGranularity
        self.order = []
        
    def addInputFile(self, label, inFile, color=None, marker=20):
        self.order.append(label)
        self.inFiles[label] = inFile
        self.labels[label] = label
        self.markers[label] = marker
        if color != None:
            self.colors[label] = color
        else:
            # choose first not occupied color (other than white)
            for autoColor in range(1,100):
                if autoColor not in self.colors.values() and not autoColor == 10:
                    self.colors[label] = autoColor
                    break
    
    def setGranularity(self, granularity):
        self.granularity = granularity
    
    def setOutputPath(self, outPath):
        self.outPath = outPath
        
    def convertName(self, name):
        out = name.replace("Bpix", "BPIX")
        out = out.replace("Fpix", "FPIX")
        out = out.replace("Plus", "+")
        out = out.replace("Minus", "-")
        out = out.replace("Fpix", "FPIX")
        out = out.replace("Tib", "TIB")
        out = out.replace("Tob", "TOB")
        out = out.replace("Tid", "TID")
        out = out.replace("Tec", "TEC")
        out = out.replace("Layer", " L")
        out = out.replace("Ring", " R")
        out = out.replace("Stereo", "S")
        out = out.replace("Rphi", "R") # different from Ring, this one does not add a space in front
        out = out.replace("In", "i")
        out = out.replace("Out", "o")
        return out
    
    def plotHist(self, folder, name, title, hists, twoDimensional=False):
        self.canvas = ROOT.TCanvas("canvas", "canvas", ROOT.gStyle.GetCanvasDefW(),ROOT.gStyle.GetCanvasDefH())
        ROOT.gPad.SetRightMargin(0.10)
        if twoDimensional:
            ROOT.gPad.SetRightMargin(0.2)
            
        
        legend = ROOT.TLegend(0.2,0.7,0.9,0.90)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.03)
        legend.SetMargin(0.15)
        legend.SetNColumns(2)
        legend.SetBorderSize(0)
        
        normalize = False
        if len (hists) > 1:
            normalize = True
        

        firstHist = True
        scaleHist = None
        maximum = 0
        for hist, label in hists:
            
            n = int(hist.Integral())
            mu = hist.GetMean()
            sigma = hist.GetRMS()
            
            if normalize:
                hist.Scale(1./hist.Integral())
            
            if hist.GetMaximum() > maximum:
                maximum = hist.GetMaximum()
            
            if firstHist:
                scaleHist = hist
                addDraw = ""
                firstHist = False
            else:
                addDraw = "same"
            
            if not twoDimensional:
                if self.markers[label] != 0:
                    drawMode = "P0%s"%(addDraw)
                else:
                    drawMode = "hist%s"%(addDraw)
            else:
                drawMode = "COLZ"
                
            scaleHist.SetMaximum(maximum*1.5)
            scaleHist.SetMinimum(0)
            
            hist.Draw(drawMode)
            
            legText = "#splitline{{{label}}}{{N={n:.2E},#mu={mu:.2f},RMS={sigma:.2f}}}".format(label=label,n=n, mu=mu,sigma=sigma)
            
            if self.markers[label] != 0:
                legend.AddEntry(hist, legText, "p")
            else:
                legend.AddEntry(hist, legText, "l")
        
        if not twoDimensional:
            legend.Draw()
        
        cmsText = ROOT.TLatex(0.16,0.96,title)
        cmsText.SetTextFont(42)
        cmsText.SetNDC()
        cmsText.Draw("same")
        
        import os
        if not os.path.isdir(self.outPath):
            os.makedirs(self.outPath)
        if not os.path.isdir(self.outPath+"/"+folder):
            os.makedirs(self.outPath+"/"+folder)
        self.canvas.SaveAs("{}/{}/{}.pdf".format(self.outPath, folder, name))
        self.canvas = None
            
        
        
    def makeResidualPlot(self, sectorNumber, coordinate):
        # residual
        hists = []
        for label in self.order:
            fi = ROOT.TFile(self.inFiles[label], "READ")
            hist = fi.Get("ApeEstimator1/Sector_{sectorNumber}/Results/h_Res{coordinate}".format(sectorNumber=sectorNumber, coordinate=coordinate))
            hist.SetLineColor(self.colors[label])
            hist.SetMarkerColor(self.colors[label])
            hist.SetMarkerStyle(self.markers[label])
            hist.SetDirectory(0)
            if self.markers[label] == 0:
                hist.SetMarkerSize(0)
                hist.SetLineWidth(2)
                
            hists.append((hist, label))
            nameHist = fi.Get("ApeEstimator1/Sector_{sectorNumber}/z_name".format(sectorNumber=sectorNumber))
            nameHist.SetDirectory(0)
            title = self.convertName(nameHist.GetTitle())
            fi.Close()
            
        name = "Sector_{sectorNumber}_Res{coordinate}".format(sectorNumber=sectorNumber, coordinate=coordinate)
        
        self.plotHist("residuals", name, title, hists)
        
        
        # normalized residual
        hists = []
        for label in self.order:
            fi = ROOT.TFile(self.inFiles[label], "READ")
            hist = fi.Get("ApeEstimator1/Sector_{sectorNumber}/Results/h_NorRes{coordinate}".format(sectorNumber=sectorNumber, coordinate=coordinate))
            hist.SetLineColor(self.colors[label])
            hist.SetMarkerColor(self.colors[label])
            hist.SetMarkerStyle(self.markers[label])
            hist.SetDirectory(0)
            if self.markers[label] == 0:
                hist.SetMarkerSize(0)
                hist.SetLineWidth(2)
                
            hists.append((hist, label))
            nameHist = fi.Get("ApeEstimator1/Sector_{sectorNumber}/z_name".format(sectorNumber=sectorNumber))
            nameHist.SetDirectory(0)
            title = self.convertName(nameHist.GetTitle())
            fi.Close()
        name = "Sector_{sectorNumber}_NorRes{coordinate}".format(sectorNumber=sectorNumber, coordinate=coordinate)
        
        self.plotHist("residuals", name, title, hists)
        
    def makeTrackPlot(self,histName, twoDimensional=False):
        hists = []
        for label in self.order:
            fi = ROOT.TFile(self.inFiles[label], "READ")
            hist = fi.Get("ApeEstimator2/TrackVariables/{histName}".format(histName=histName))
            hist.SetLineColor(self.colors[label])
            hist.SetMarkerColor(self.colors[label])
            hist.SetMarkerStyle(self.markers[label])
            hist.SetDirectory(0)
            if self.markers[label] == 0:
                hist.SetMarkerSize(0)
                hist.SetLineWidth(2)
            
            if twoDimensional:
                self.plotHist("tracks", histName+"_"+label, label, [(hist, label),], twoDimensional=True)
            else:
                hists.append((hist, label))
        if len(hists) > 0:
            self.plotHist("tracks", histName, histName, hists, twoDimensional=twoDimensional)
            
    
    
    def draw(self):
        for coordinate in self.granularity.sectors.keys():
            rangeList = self.granularity.sectors[coordinate]
            for first, last in rangeList:
                for i in range(first, last+1):
                    self.makeResidualPlot(i, coordinate)
                    
        self.makeTrackPlot("h_hitsSize")
        self.makeTrackPlot("h_hitsValid")
        self.makeTrackPlot("h_hitsInvalid")
        self.makeTrackPlot("h_hits2D")
        self.makeTrackPlot("h_layersMissed")
        self.makeTrackPlot("h_hitsPixel")
        self.makeTrackPlot("h_hitsStrip")
        self.makeTrackPlot("h_charge")
        self.makeTrackPlot("h_chi2")
        self.makeTrackPlot("h_ndof")
        self.makeTrackPlot("h_norChi2")
        self.makeTrackPlot("h_prob")
        self.makeTrackPlot("h_eta")
        self.makeTrackPlot("h_etaErr")
        self.makeTrackPlot("h_theta")
        self.makeTrackPlot("h_phi")
        self.makeTrackPlot("h_phiErr")
        self.makeTrackPlot("h_d0Beamspot")
        self.makeTrackPlot("h_d0BeamspotErr")
        self.makeTrackPlot("h_dz")
        self.makeTrackPlot("h_dzErr")
        self.makeTrackPlot("h_pt")
        self.makeTrackPlot("h_ptErr")
        self.makeTrackPlot("h_meanAngle")
        self.makeTrackPlot("h_hitsGood")
        # these don't have several histograms in one plot
        self.makeTrackPlot("h2_meanAngleVsHits", twoDimensional=True)
        self.makeTrackPlot("h2_hitsGoodVsHitsValid", twoDimensional=True)
        self.makeTrackPlot("h2_hitsPixelVsEta", twoDimensional=True)
        self.makeTrackPlot("h2_hitsPixelVsTheta", twoDimensional=True)
        self.makeTrackPlot("h2_hitsStripVsEta", twoDimensional=True)
        self.makeTrackPlot("h2_hitsStripVsTheta", twoDimensional=True)
        self.makeTrackPlot("h2_ptVsTheta", twoDimensional=True)
        
        

def main():
    pass


if __name__ == "__main__":
    main()
