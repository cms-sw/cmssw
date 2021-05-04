import ROOT
ROOT.gROOT.SetBatch(True)
from setTDRStyle import setTDRStyle

from systematicErrors import *
from granularity import *


class ResultPlotter:
    def __init__(self):
        setTDRStyle()
        self.inFiles = {}
        self.hitNumbers = {}
        self.labels = {}
        self.colors = {}
        self.markers = {}
        self.outPath = None
        self.hasSystematics = {}
        self.systematics = {}
        self.granularity = standardGranularity
        self.title = ""
        self.order = []
        
    def addInputFile(self, label, inFile, color=None, marker=20, hitNumbers=None):
        self.order.append(label)
        self.inFiles[label] = inFile
        self.labels[label] = label
        self.systematics[label] = []
        self.hasSystematics[label] = False
        self.markers[label] = marker
        self.hitNumbers[label] = hitNumbers
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
    
    def doSystematics(self, label):
        self.hasSystematics[label] = True
    
    def addSystematics(self, label, systematics, additive=True):
        self.hasSystematics[label] = True
        if not additive:
            self.systematics[label] = []
        self.systematics[label].append(systematics)
    
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
    
    def makeHitNumbers(self, label, sectorRange, coordinate):
        self.numHitCounters += 1
        sectors = list(range(sectorRange[0],sectorRange[1]+1))
        numSectors = len(sectors)
        
        fi = ROOT.TFile(self.hitNumbers[label], "READ")
        align = 22
        size = 0.02
        font = 42
        labels = []
        for i,sector in enumerate(sectors):
            hitHist = fi.Get("ApeEstimator1/Sector_{}/Results/h_NorRes{}".format(sector, coordinate.upper()))
            num = hitHist.GetEntries()
            posX = (float(i)+0.5)/numSectors*(1-ROOT.gPad.GetLeftMargin()-ROOT.gPad.GetRightMargin())+ROOT.gPad.GetLeftMargin()
            posY = (1-ROOT.gPad.GetTopMargin()-size)-1.2*size*self.numHitCounters
            
            label = ROOT.TLatex(posX, posY, "%.2E"%(num))
            label.SetNDC(True)
            label.SetTextColor(self.colors[label])
            label.SetTextSize(size)
            label.SetTextFont(font)
            label.SetTextAngle(45)
            label.SetTextAlign(align)
            labels.append(label)
        return labels
            
    def makeHist(self, label, sectorRange, coordinate, number):
        sectors = list(range(sectorRange[0],sectorRange[1]+1))
        numSectors = len(sectors)
        
        hist = ROOT.TH1F("{}hist{}_{}".format(label, number, coordinate), "", numSectors, 0, numSectors)
        hist.SetTitle(";;#sigma_{align," + coordinate.lower() + "}  [#mum]")
        hist.SetAxisRange(0.,100.,"Y")
        
        syst = None
        if self.hasSystematics[label]:
            syst = ROOT.TGraphAsymmErrors()
        
        fi = ROOT.TFile(self.inFiles[label], "READ")
        nameTree = fi.Get("nameTree")
        apeTree = fi.Get("iterTree{}".format(coordinate.upper()))
        # Get last entries in branches (for apeTree) to retrieve result of this iteration
        # in iterTreeX/Y, the results of the previous iterations are also stored
        nameTree.GetEntry(0)
        apeTree.GetEntry(apeTree.GetEntries()-1)
        iBin = 1
        for sector in sectors:
            sectorApe  = 10000. * (float(getattr(apeTree,  "Ape_Sector_{}".format(sector))))**0.5
            sectorName = self.convertName(str(getattr(nameTree, "Ape_Sector_{}".format(sector))))
            binWidth = hist.GetXaxis().GetBinCenter(iBin) - hist.GetXaxis().GetBinLowEdge(iBin)
            hist.SetBinContent(iBin, sectorApe)
            hist.SetBinError(iBin, 0.0000001)
            hist.GetXaxis().SetBinLabel(iBin, sectorName)
            
            if self.hasSystematics[label]:
                sysErrUp = 0
                sysErrDn = 0
                # add up errors quadratically
                for partError in self.systematics[label]:
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
                
                syst.SetPoint(iBin, hist.GetXaxis().GetBinCenter(iBin), sectorApe)
                syst.SetPointError(iBin, binWidth, binWidth, sysErrDn, sysErrUp)
            
            iBin += 1
        hist.SetDirectory(0)
        fi.Close()
        return hist, syst
        
    def draw(self):
        allLabels = []
        for coordinate in self.granularity.sectors.keys():
            plotNumber = 0
            rangeList = self.granularity.sectors[coordinate]
            for sectorRange in rangeList:
                self.canvas = ROOT.TCanvas("canvas", "canvas", int(ROOT.gStyle.GetCanvasDefW()*len(list(range(sectorRange[0],sectorRange[1]+1)))/10.),ROOT.gStyle.GetCanvasDefH())
                ROOT.gPad.SetRightMargin(0.10)
                
                legend = ROOT.TLegend(0.2,0.62,0.5,0.82)
                legend.SetFillColor(0)
                legend.SetFillStyle(0)
                legend.SetTextSize(0.04)
                legend.SetMargin(0.30)
                legend.SetBorderSize(0)
                
                firstHist = True
                histos = [] # need to save histos or they will be deleted right after variable is set to something else
                systGraphs = [] # same for systematics errors
                self.numHitCounters = 0
                for name in self.order:
                    if firstHist:
                        addDraw = ""
                        firstHist = False
                    else:
                        addDraw = "same"
                    
                    if self.markers[name] != 0:
                        drawMode = "P0%s"%(addDraw)
                    else:
                        drawMode = "hist%s"%(addDraw)
                    
                    histo, syst = self.makeHist(name, sectorRange, coordinate, plotNumber) 
                    histo.SetMarkerColor(self.colors[name])
                    histo.SetMarkerStyle(self.markers[name])
                    if self.markers[name] == 0:
                        histo.SetMarkerSize(0)
                        histo.SetLineColor(self.colors[name])
                        histo.SetLineWidth(2)
                        
                    histo.Draw(drawMode)
                    histos.append(histo)
                    if self.markers[name] != 0:
                        legend.AddEntry(histo, self.labels[name], "p")
                    else:
                        legend.AddEntry(histo, self.labels[name], "l")
                        
                    if self.hasSystematics[name]:
                        syst.SetFillColor(self.colors[name])
                        syst.SetFillStyle(3354)
                        syst.Draw("02same")
                        systGraphs.append(syst)
                    
                    if self.hitNumbers[name] != None:
                        labels = self.makeHitNumbers(name, sectorRange, coordinate) 
                        allLabels.extend(labels)
                        for label in labels:
                            label.Draw("same")
                legend.Draw()
                self.canvas.Update()
                
                cmsText = ROOT.TLatex(0.16,0.96,self.title)
                cmsText.SetTextFont(42)
                cmsText.SetNDC()
                cmsText.Draw("same")
                
                import os
                if not os.path.isdir(self.outPath):
                    os.makedirs(self.outPath)
                
                self.canvas.SaveAs("{}/results_{}_{}.pdf".format(self.outPath, coordinate, self.granularity.names[coordinate][plotNumber]))
                self.canvas = None
                legend = None
                histos = None
                plotNumber += 1

def main():
    pass


if __name__ == "__main__":
    main()
