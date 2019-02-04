import ROOT
ROOT.gROOT.SetBatch(True)
from setTDRStyle import setTDRStyle

from systematicErrors import *
from granularity import *


class ResultPlotter:
    def __init__(self):
        setTDRStyle()
        self.names = {}
        self.inFiles = {}
        self.labels = {}
        self.colors = {}
        self.outPath = None
        self.hasSystematics = {}
        self.systematics = {}
        self.granularity = standardGranularity
        self.title = ""
        
    def addInputFile(self, name, inFile, label, color=None):
        self.names[label] = name
        self.inFiles[label] = inFile
        self.labels[label] = label
        self.systematics[name] = []
        self.hasSystematics[name] = False
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
    
    def setTitle(self, title):
        self.title = title
    
    def doSystematics(self, name):
        self.hasSystematics[name] = True
    
    def addSystematics(self, name, systematics, additive=True):
        self.hasSystematics[name] = True
        if not additive:
            self.systematics[name] = []
        self.systematics[name].append(systematics)
    
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
    
    def makeHist(self, name, sectorRange, coordinate, number):
        sectors = range(sectorRange[0],sectorRange[1]+1)
        numSectors = len(sectors)
        
        hist = ROOT.TH1F("{}hist{}_{}".format(name, number, coordinate), "", numSectors, 0, numSectors)
        hist.SetTitle(";;#sigma_{align," + coordinate.lower() + "}  [#mum]")
        hist.SetAxisRange(0.,100.,"Y")
        
        syst = None
        if self.hasSystematics[name]:
            syst = ROOT.TGraphAsymmErrors()
        
        fi = ROOT.TFile(self.inFiles[name], "READ")
        nameTree = fi.Get("nameTree")
        apeTree = fi.Get("iterTree{}".format(coordinate))
        # Get last entries in branches (for apeTree) to retrieve result of this iteration
        # in iterTreeX/Y, the results of the previous iterations are also stored
        nameTree.GetEntry(0)
        apeTree.GetEntry(apeTree.GetEntries()-1)
        iBin = 1
        for sector in sectors:
            sectorApe  = 10000. * (float(getattr(apeTree,  "Ape_Sector_{}".format(sector))))**0.5
            sectorName = self.convertName(str(getattr(nameTree, "Ape_Sector_{}".format(sector))))
            hist.SetBinContent(iBin, sectorApe)
            hist.SetBinError(iBin, 0.0000001)
            hist.GetXaxis().SetBinLabel(iBin, sectorName)
            
            if self.hasSystematics[name]:
                sysErrUp = 0
                sysErrDn = 0
                # add up errors quadratically
                for partError in self.systematics[name]:
                    scaleFac = 1.0
                    if partError.isRelative[sector-1]:
                        scaleFac = sectorApe
        
                    if partError.direction[sector-1] == DIR_BOTH:
                        sysErrUp += (scaleFac*partError[coordinate][sector-1])**2
                        sysErrDn += (scaleFac*partError[coordinate][sector-1])**2
                    elif partError.direction[sector-1] == DIR_DOWN:
                        sysErrDn += (scaleFac*partError[coordinate][sector-1])**2
                    elif partError.direction[sector-1] == DIR_UP:
                        sysErrUp += (scaleFac*partError[coordinate][sector-1])**2
                sysErrUp = sysErrUp**0.5
                sysErrDn = sysErrDn**0.5
                binWidth = hist.GetXaxis().GetBinCenter(iBin) - hist.GetXaxis().GetBinLowEdge(iBin)
                syst.SetPoint(iBin, hist.GetXaxis().GetBinCenter(iBin), sectorApe)
                syst.SetPointError(iBin, binWidth, binWidth, sysErrDn, sysErrUp)
            
            iBin += 1
        hist.SetDirectory(0)
        fi.Close()
        return hist, syst
        
    def draw(self):
        for coordinate in self.granularity.sectors.keys():
            plotNumber = 0
            rangeList = self.granularity.sectors[coordinate]
            for sectorRange in rangeList:
                self.canvas = ROOT.TCanvas("canvas", "canvas", int(ROOT.gStyle.GetCanvasDefW()*len(range(sectorRange[0],sectorRange[1]+1))/10.),ROOT.gStyle.GetCanvasDefH())
                ROOT.gPad.SetRightMargin(0.10)
                
                legend = ROOT.TLegend(0.2,0.65,0.5,0.85)
                legend.SetFillColor(0)
                legend.SetFillStyle(0)
                legend.SetTextSize(0.04)
                legend.SetMargin(0.30)
                legend.SetBorderSize(0)
                
                firstHist = True
                histos = [] # need to save histos or they will be deleted right after variable is set to something else
                systGraphs = [] # same for systematics errors
                for name in self.inFiles.keys():
                    if firstHist:
                        drawMode = "E0"
                        firstHist = False
                    else:
                        drawMode = "E0same"
                    histo, syst = self.makeHist(name, sectorRange, coordinate, plotNumber) 
                    histo.SetMarkerColor(self.colors[name])
                    histo.Draw(drawMode)
                    histos.append(histo)
                    legend.AddEntry(histo, self.labels[name], "p")
                    
                    if self.hasSystematics[name]:
                        syst.SetFillColor(self.colors[name])
                        syst.SetFillStyle(3354)
                        syst.Draw("02same")
                        systGraphs.append(syst)
                    
                legend.Draw()
                self.canvas.Update()
                
                cmsText = ROOT.TLatex(0.16,0.96,self.title)
                cmsText.SetTextFont(42)
                cmsText.SetNDC()
                cmsText.Draw("same")
                
                import os
                if not os.path.isdir(self.outPath):
                    os.makedirs(self.outPath)
                
                self.canvas.SaveAs("{}/result_{}_{}.pdf".format(self.outPath, coordinate, self.granularity.names[coordinate][plotNumber]))
                self.canvas = None
                legend = None
                histos = None
                plotNumber += 1

def main():
    pass


if __name__ == "__main__":
    main()
