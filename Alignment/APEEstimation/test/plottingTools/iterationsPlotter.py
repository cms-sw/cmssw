import ROOT
ROOT.gROOT.SetBatch(True)
from setTDRStyle import setTDRStyle

# load some default things form there
from granularity import *


class IterationsPlotter:
    def __init__(self):
        setTDRStyle()
        self.inFile = None
        self.outPath = None
        self.granularity = standardGranularity
        self.title = ""
        
    def setInputFile(self, inFile):
        self.inFile = inFile
    
    def setGranularity(self, granularity):
        self.granularity = granularity
    
    def setOutputPath(self, outPath):
        self.outPath = outPath
    
    def setTitle(self, title):
        self.title = title
    
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
    
    def makeHists(self, sectorRange, coordinate):
        sectors = range(sectorRange[0],sectorRange[1]+1)
        numSectors = len(sectors)
        
        fi = ROOT.TFile(self.inFile, "READ")
        nameTree = fi.Get("nameTree")
        nameTree.GetEntry(0)
        apeTree = fi.Get("iterTree{}".format(coordinate))
        noEntries = apeTree.GetEntries()
        
        hists = []
        names = []
        maximum = 10
        for i,sector in enumerate(sectors):
            hist = ROOT.TH1F("hist{}_{}".format(coordinate, sector), "", noEntries, 0-0.5, noEntries-0.5)
            hist.SetTitle(";iteration number;#sigma_{align," + coordinate.lower() + "}  [#mum]")
            #~ hist.SetAxisRange(0.,100.,"Y")
            hist.SetMarkerStyle(20+i)
            hist.SetDirectory(0)
            hists.append(hist)
            no_it = 1
            for it in apeTree:
                hist.SetBinContent(no_it, 10000. * (float(getattr(it,  "Ape_Sector_{}".format(sector))))**0.5)
                no_it += 1
            if hist.GetMaximum() > maximum:
                maximum = hist.GetMaximum()
                
            sectorName = self.convertName(str(getattr(nameTree, "Ape_Sector_{}".format(sector))))
            names.append(sectorName)
        

        
        fi.Close()
        return hists, names, maximum
        
    def draw(self):
        for coordinate in self.granularity.sectors.keys():
            rangeList = self.granularity.sectors[coordinate]
            for j, sectorRange in enumerate(rangeList):
                self.canvas = ROOT.TCanvas("canvas", "canvas", int(ROOT.gStyle.GetCanvasDefW()*15/10.),ROOT.gStyle.GetCanvasDefH())
                ROOT.gPad.SetRightMargin(0.10)
                
                legend = ROOT.TLegend(0.2,0.73,0.85,0.93)
                legend.SetFillColor(0)
                legend.SetFillStyle(0)
                legend.SetTextSize(0.025)
                legend.SetMargin(0.30)
                legend.SetBorderSize(0)
                legend.SetNColumns(4)
                
                hists, names, maximum = self.makeHists(sectorRange, coordinate)
                for i, hist in enumerate(hists):
                    if i == 0:
                        drawOption = "P0L"
                    else:
                        drawOption = "P0Lsame"
                    hist.SetMaximum(maximum*1.5)
                    hist.Draw(drawOption)
                    legend.AddEntry(hist, names[i], "PL")
                legend.Draw()
                
                cmsText = ROOT.TLatex(0.16,0.96,self.title)
                cmsText.SetTextFont(42)
                cmsText.SetNDC()
                cmsText.Draw("same")
                
                granularityText = ROOT.TLatex(0.9,0.96,self.granularity.names[coordinate][j])
                granularityText.SetTextAlign(31)
                granularityText.SetTextFont(42)
                granularityText.SetNDC()
                granularityText.Draw("same")
                
                import os
                if not os.path.isdir(self.outPath):
                    os.makedirs(self.outPath)
                
                self.canvas.SaveAs("{}/iterations_{}_{}.pdf".format(self.outPath, coordinate, self.granularity.names[coordinate][j]))
                self.canvas = None
def main():
    pass

if __name__ == "__main__":
    main()
