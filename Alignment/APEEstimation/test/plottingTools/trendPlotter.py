from __future__ import print_function
import ROOT
ROOT.gROOT.SetBatch(True)
from setTDRStyle import setTDRStyle
from granularity import *
import os

try:
    base = os.environ['CMSSW_BASE']
except KeyError:
    base = "../../../../.."

# 2016 is missing here
lumiFiles = {
    2016: "{base}/src/Alignment/APEEstimation/data/lumiperrun2016.txt",
    2017: "{base}/src/Alignment/APEEstimation/data/lumiperrun2017.txt",
    2018: "{base}/src/Alignment/APEEstimation/data/lumiperrun2018.txt",
}
pixelIOVs = {
    2016: [271866, 276315, 278271, 280928],
    2017: [297281, 298653, 299443, 300389, 301046, 302131, 303790, 303998, 304911],
    2018: [316758, 317527, 317661, 317664, 318227, 320377],
}

lumis = {}
intLumis = {}
runs = {}
years = []

for year in lumiFiles.keys():
    runs[year] = []
    years.append(year)
    intLumis[year] = 0
    with open(lumiFiles[year].format(base=base), "r") as fi:
        for line in fi:
            in len(line) == 0:
                continue
            line = line.strip()
            run, lumi = line.split(" ")
            run = int(run)
            lumi = float(lumi)*0.001 # convert to fb^-1
            lumis[run] = lumi
            runs[year].append(run)
            intLumis[year] += lumi
years.sort()

def runToLumi(whichRun, fromYear, inclusive=False):
    lumiResult = 0.0
    for year in years:
        if year < fromYear:
            continue
        for run in runs[year]:
            if run > whichRun:
                break
            if run < whichRun:
                lumiResult += lumis[run]
            if run == whichRun and inclusive:
                lumiResult += lumis[run]
    return lumiResult


def whichYear(run):
    thisYear = -1
    years = runs.keys()
    years.sort()
    for year in years:
        if min(runs[year]) <= run:
            thisYear = year
    return thisYear
    print("Run %d not in range of any year"%(run))
    return -1
    

class TrendPlotter:
    def __init__(self):
        setTDRStyle()
        self.names = {}
        self.outPath = None
        self.granularity = standardGranularity
        self.title = ""
        self.points = []
        self.years = []
        self.doLumi = True
        self.colors = []
        self.log = False
        
    def addTrend(self, label, points, dashed=False, color=None, marker=None):
        self.points.append( (label, points, dashed, color, marker) )
        if color:
            self.colors.append(color)
    
    def setGranularity(self, granularity):
        self.granularity = granularity
    
    def setOutputPath(self, outPath):
        self.outPath = outPath
    
    def setTitle(self, title):
        self.title = title
    
    def setLog(self, log=True):
        self.log = log
   
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
        out = out.replace("Rphi", "R") # other than Ring, this one does not add a space in front
        out = out.replace("In", "i")
        out = out.replace("Out", "o")
        return out
    
    def drawTrendPlot(self, sector, coordinate, number):         
        self.canvas = ROOT.TCanvas("canvas%s_%s"%(sector, coordinate), "canvas", int(ROOT.gStyle.GetCanvasDefW()*3),ROOT.gStyle.GetCanvasDefH())
        ROOT.gPad.SetLeftMargin(0.06)
        ROOT.gPad.SetRightMargin(0.04)
        
        iTrend = 0
        
        if self.log:
            minApe = 0.9
            maxApe = 7000.0
            ROOT.gPad.SetLogy()
        else:
            minApe = 0
            maxApe = 100
        
        
        # calibrate runrange
        firstRun = 999999
        lastRun = 0
        for label, points, dashed, color, marker in self.points:
            firstRun = min(min(points, key=lambda x:x[0])[0], firstRun)
            lastRun = max(max(points, key=lambda x:x[1])[1], lastRun)
        theFirstRun = firstRun
        theLastRun = lastRun
        
        firstYear = whichYear(firstRun)
        lastYear = whichYear(lastRun)
        minLumi = 0
        
        maxLumi = 0
        for year in intLumis.keys():
            if year >= firstYear and year <= lastYear:
                maxLumi += intLumis[year]
        
        verticalLines = []
        lineLabels = []
        i = 0
        for year in range(firstYear, lastYear+1):
            for position in pixelIOVs[year]:
                if self.doLumi:
                    posLumi = runToLumi(position, firstYear, False)
                else:
                    posLumi = position
                vLine = ROOT.TLine(posLumi,minApe,posLumi,maxApe)
                vLine.SetLineStyle(9)
                vLine.SetLineColor(ROOT.kRed)
                verticalLines.append(vLine)
                
                posApe = 70+3.5*(maxApe-minApe)/100*(i % 5)
            
                text = ROOT.TLatex(posLumi + (maxLumi-minLumi)*0.003 , posApe, str(position))
                text.SetTextFont(42)
                text.SetTextSize(0.035)
                text.SetTextColor(ROOT.kRed+2)
                lineLabels.append(text)
                i += 1
        
        
        legend = ROOT.TLegend(0.07, 0.89, 0.935, 0.96)
        legend.SetTextFont(42)
        legend.SetTextSize(0.045)
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)
        legend.SetNColumns(5)
        
        if self.doLumi:
            hAxisLumi = ROOT.TH2F("hAxisRun%s_%s"%(sector, coordinate),"", 10, float(minLumi), float(maxLumi), 10, minApe, maxApe)
            hAxisLumi.SetTitle(";integrated luminosity [fb^{-1}];#sigma_{align," + coordinate.lower() + "}  [#mum]")
        else:
            hAxisLumi = ROOT.TH2F("hAxisRun%s_%s"%(sector, coordinate),"", 10, theFirstRun, theLastRun, 10, minApe, maxApe)
            hAxisLumi.SetTitle(";Run number;#sigma_{align," + coordinate.lower() + "}  [#mum]")
        hAxisLumi.GetYaxis().SetTitleOffset(0.4)
        hAxisLumi.GetXaxis().SetNdivisions(510)
        hAxisLumi.Draw("AXIS")
        trends = []
        useColor = 1
        for label, points, dashed, color, marker in self.points:
            iTrend += 1
            graphLumi = ROOT.TGraphErrors()
            trends.append(graphLumi)
            
            if color:
                graphLumi.SetLineColor(color)
                graphLumi.SetMarkerColor(color)
            else:
                while True:
                    if useColor not in self.colors and useColor not in [0,10]:
                        self.colors.append(useColor)
                        graphLumi.SetLineColor(useColor)
                        graphLumi.SetMarkerColor(useColor)
                        break
                    useColor += 1
            
            if marker:
                graphLumi.SetLineWidth(0)
                graphLumi.SetMarkerSize(1.3)
                graphLumi.SetMarkerStyle(marker)
            else:
                graphLumi.SetLineWidth(2)
                graphLumi.SetMarkerSize(0)
                graphLumi.SetMarkerStyle(20)
            
            
            if dashed:
                graphLumi.SetLineStyle(2)
            
            
            iPoint = 0
            for firstRun, lastRun, file in points:
                fi = ROOT.TFile(file, "READ")
                nameTree = fi.Get("nameTree")
                apeTree = fi.Get("iterTree{}".format(coordinate))
                nameTree.GetEntry(0)
                apeTree.GetEntry(apeTree.GetEntries()-1)
                
                sectorApe  = 10000. * (float(getattr(apeTree,  "Ape_Sector_{}".format(sector))))**0.5
                sectorName = str(getattr(nameTree, "Ape_Sector_{}".format(sector)))
                
                # this could be done centrally for each trend and then not be redone for each sector
                # but it does not take too much time (most time is spent reading out ROOT files)
                if self.doLumi:
                    lumiStart = runToLumi(firstRun, firstYear, False)
                    lumiEnd = runToLumi(lastRun, firstYear, True)
                else:
                    lumiStart = firstRun
                    lumiEnd = lastRun
                    
                xPosLumi = (lumiStart+lumiEnd) / 2
                xErrLumi = -(lumiStart-lumiEnd) / 2
                graphLumi.SetPoint(iPoint, xPosLumi, sectorApe)
                graphLumi.SetPointError(iPoint, xErrLumi,0)
                
                iPoint += 1
                fi.Close()
            graphLumi.Draw("PZ same")
            if marker:
                legend.AddEntry(graphLumi, label, "pl")
            else:
                legend.AddEntry(graphLumi, label, "l")
        cmsText = ROOT.TLatex(0.16,0.96,self.title)
        cmsText.SetTextFont(42)
        cmsText.SetNDC()
        cmsText.Draw("same")
        
        sectorText = ROOT.TLatex(0.9,0.96,sectorName)
        sectorText.SetTextAlign(31)
        sectorText.SetTextFont(42)
        sectorText.SetNDC()
        sectorText.Draw("same")
        
        
        
        for vLine in verticalLines:
            vLine.Draw("same")
        for llabel in lineLabels:
            llabel.Draw("same")
            
        legend.Draw("same")
        
        ROOT.gPad.RedrawAxis()
        
        import os
        if not os.path.isdir("{}/{}".format(self.outPath, self.granularity.names[coordinate][number])):
            os.makedirs("{}/{}".format(self.outPath, self.granularity.names[coordinate][number]))
        
        app = ""
        if not self.doLumi:
            app = "_byRun"
        
        self.canvas.SaveAs("{}/{}/trend_{}_{}{}.pdf".format(self.outPath, self.granularity.names[coordinate][number], coordinate, sectorName, app))
        self.canvas = None
        
        
        
    def draw(self):
        for coordinate in self.granularity.sectors.keys():
            plotNumber = 0
            rangeList = self.granularity.sectors[coordinate]
            for sectorRange in rangeList:
                for sector in range(sectorRange[0], sectorRange[1]+1):
                    self.drawTrendPlot(sector, coordinate, plotNumber)
                plotNumber += 1
                

def main():
    pass


if __name__ == "__main__":
    main()
