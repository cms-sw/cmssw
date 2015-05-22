#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *

import os, ROOT, string

class AbsCollectionDrawer(Module):
    def __init__(self,name,label,cut=None):
        Module.__init__(self,name,booker=None)
        self._label = label
        self._cut = cut
    def beginJob(self):
        pass
    def beginEvent(self,event):
        pass
    def drawOne(self,index,object):
        pass
    def analyze(self,event):
        self.beginEvent(event)
        objs = Collection(event,self._label)
        for i,o in enumerate(objs):
            if self._cut != None and not eval(self._cut, globals(), o): 
                continue
            self.drawOne(i+1,o)
        return True

class MarkerDrawer(AbsCollectionDrawer):
    def __init__(self,name,label,markerStyle=20,markerColor=1,markerSize=2,cut=None,maxItems=99):
        AbsCollectionDrawer.__init__(self,name,label,cut)
        self._marker = ROOT.TMarker(0,0,markerStyle)
        self._marker.SetMarkerStyle(markerStyle)
        self._marker.SetMarkerColor(markerColor)
        self._marker.SetMarkerSize(markerSize)
        self._maxItems = maxItems
    def drawOne(self,index,object):
        if (index > self._maxItems): return
        self._marker.DrawMarker(object.eta,object.phi)

class LabelDrawer(AbsCollectionDrawer):
    def __init__(self,name,label,text,textFont=42,textSize=0.035,textColor=1,cut=None,maxItems=99,xoffs=0.4,yoffs=0.4):
        AbsCollectionDrawer.__init__(self,name,label,cut)
        self._maxItems = maxItems
        self._text = text
        self._latex = ROOT.TLatex(0,0,text)
        self._latex.SetTextFont(textFont)
        self._latex.SetTextColor(textColor)
        self._latex.SetTextSize(textSize)
        self._latex.SetTextAlign(22)
        self._xoffs = xoffs
        self._yoffs = yoffs
    def drawOne(self,index,object):
        if (index > self._maxItems): return
        x,y = object.eta, object.phi
        align = 22;
        x += self._xoffs if object.eta < 0 else -self._xoffs
        y += self._yoffs if object.phi < 0 else -self._yoffs
        align += 10 if object.eta > 0 else -10
        align +=  1 if object.phi > 0 else  -1
        self._latex.SetTextAlign(align)
        text = string.Formatter().vformat(self._text,[],{'i':index, 'o':object})
        self._latex.DrawLatex(x,y,text)

class CircleDrawer(AbsCollectionDrawer):
    def __init__(self,name,label,radius,lineColor=1,lineWidth=2,lineStyle=1,cut=None,maxItems=99):
        AbsCollectionDrawer.__init__(self,name,label,cut)
        self._radius = radius
        self._circle = ROOT.TEllipse(0,0,radius)
        self._circle.SetLineColor(lineColor)
        self._circle.SetLineWidth(lineWidth)
        self._circle.SetLineStyle(lineStyle)
        self._circle.SetFillStyle(0)
        self._maxItems = maxItems
    def drawOne(self,index,object):
        if (index >= self._maxItems): return
        self._circle.DrawEllipse(object.eta,object.phi,self._radius,self._radius,0.,360.,0.)



class CanvasMaker(Module):
    def __init__(self,name,maxEvents=100):
        Module.__init__(self,name,booker=None)
        ROOT.gROOT.ProcessLine(".x tdrstyle.cc")
        self._canvas = ROOT.TCanvas("c1","c1");
        self._canvas.SetGridx(1)
        self._canvas.SetGridy(1)
        self._frame = ROOT.TH1F("frame","frame;#eta;#phi",1,-5,5);
        self._frame.GetYaxis().SetRangeUser(-3.1416,3.1416);
        self._frame.SetStats(0);
        self._maxEvents = maxEvents
        self._currEvents = 0
    def canvas(self): 
        return self._canvas
    def analyze(self,event):
        self._currEvents += 1;
        if self._currEvents > self._maxEvents: return False
        self._canvas.Clear()
        self._frame.Draw()
        self._frame.SetTitle("Run %d, Lumi %d, Event %d" % (event.run, event.lumi, event.evt))
        return True

class CanvasPrinter(Module):
    def __init__(self,name,canvasMaker,fileTemplates):
        Module.__init__(self,name,booker=None)
        self._canvasMaker = canvasMaker
        self._fileTemplate = fileTemplates[:]
    def analyze(self,event):
        for t in self._fileTemplate:
            fname = string.Formatter().vformat(t,[],event)
            self._canvasMaker.canvas().Print(fname)
            print fname

class LeptonMatchDrawer(AbsCollectionDrawer):
    def __init__(self,name,label,markerStyle=20,markerColors={0:99,1:1,2:4},markerSize=2,cut=None,maxItems=99):
        AbsCollectionDrawer.__init__(self,name,label,cut)
        self._marker = ROOT.TMarker(0,0,markerStyle)
        self._marker.SetMarkerStyle(markerStyle)
        self._marker.SetMarkerSize(markerSize)
        self._maxItems = maxItems
        self._markerColors = markerColors
    def beginEvent(self,event):
        self._glep = Collection(event,"GenLep")
        self._gtau = Collection(event,"GenLepFromTau")
    def drawOne(self,index,object):
        if (index > self._maxItems): return
        flag = 0
        (gmatch,dr) = closest(object,self._glep)
        if dr < 0.7 and abs(gmatch.pdgId) == abs(object.pdgId):
           flag = 2  
        else:
            (gmatch,dr) = closest(object,self._gtau)
            if dr < 0.7 and abs(gmatch.pdgId) == abs(object.pdgId):
                flag = 1
        self._marker.SetMarkerColor(self._markerColors[flag])
        self._marker.DrawMarker(object.eta,object.phi)


from sys import argv
f = ROOT.TFile.Open(argv[1])
ROOT.gROOT.ProcessLine(".x functions.cc+");
t = f.Get("ttHLepTreeProducerBase")
#t.AddFriend("newMC/t","/data/gpetrucc/8TeV/ttH/TREES_030113_CRIS_HADD/0_leptonMC_v1/lepMCFriend_TTJets.root")
#t.AddFriend("newMC/t","/afs/cern.ch/user/g/gpetrucc/ttH/CMGTools/CMSSW_5_3_5/src/CMGTools/TTHAnalysis/macros/leptons/0_leptonMC_v2_030113/lepMCFriend_TTJets.root")
print "Reading %s (%d entries)" % (argv[1], t.GetEntries())

cMaker   = CanvasMaker("cm",maxEvents=40)
cPrinter = CanvasPrinter("cp",cMaker,[ "/afs/cern.ch/user/g/gpetrucc/public_html/drop/plots/ttH/sparks/v3/3l/r{run}_ls{lumi}_ev{evt}.png" ])
leps  = MarkerDrawer("leps","LepGood", markerColor=62)
#lepsXX  = MarkerDrawer("leps","LepGood", markerColor=92, markerSize=4, cut="mcMatchId == 0")
jets  = CircleDrawer("leps","Jet", 0.5, lineColor=210, lineStyle=1, cut="btagCSV <= 0.244")
jetsL = CircleDrawer("leps","Jet", 0.5, lineColor=92,  lineStyle=1, cut="btagCSV <= 0.679 and btagCSV > 0.244")
jetsB = CircleDrawer("leps","Jet", 0.5, lineColor=100, lineStyle=1, cut="btagCSV > 0.679")
jetsF = CircleDrawer("leps","JetFwd", 0.5, lineColor=202, lineStyle=1)
glepsH = MarkerDrawer("leps","GenLep", markerStyle=34, markerSize=3.0, markerColor=214, cut="sourceId == 25")
glepsHt = MarkerDrawer("leps","GenLepFromTau", markerStyle=34, markerSize=2.0, markerColor=223, cut="sourceId == 25")
glepsT = MarkerDrawer("leps","GenLep", markerStyle=34, markerSize=3.0, markerColor=1,   cut="sourceId == 6")
glepsTt = MarkerDrawer("leps","GenLepFromTau", markerStyle=34, markerSize=2.0, markerColor=28,   cut="sourceId == 6")
quarksB = MarkerDrawer("leps","GenBQuark", markerStyle=33, markerSize=3.0, markerColor=206)
quarksH = MarkerDrawer("leps","GenQuark",  markerStyle=33, markerSize=3.0, markerColor=223, cut="sourceId == 25")
quarksT = MarkerDrawer("leps","GenQuark",  markerStyle=33, markerSize=3.0, markerColor=1, cut="sourceId != 25")
quarks25 = MarkerDrawer("leps","GenQuark",  markerStyle=33, markerSize=5.0, markerColor=66, cut="pt > 25")
quarksPt  = LabelDrawer("leps","GenQuark", '{o.pt:.0f}', cut=None, textSize=0.03, xoffs=0.2, yoffs=0)
quarksBPt  = LabelDrawer("leps","GenBQuark", '{o.pt:.0f}', cut=None, textSize=0.03, xoffs=0.2, yoffs=0, textColor=206)
#jetsPt    = LabelDrawer("leps","Jet", '{o.pt:.0f}', textColor=4,xoffs=0.0,yoffs=0.0, textSize=0.03)
#cut = "nLepGood == 2 && nJet25 >= 6 && nBJetMedium25 >= 1 && abs(LepGood1_pdgId+LepGood2_pdgId)==26 && GenHiggsDecayMode == 24"
cut = "nLepGood == 3 && nJet25 >= 4  && abs(mass_2(Jet3_pt,Jet3_eta,Jet3_phi,Jet3_mass,Jet4_pt,Jet4_eta,Jet4_phi,Jet4_mass)-80)<10 && nBJetMedium25 >= 2 && GenHiggsDecayMode == 24 && minMllAFAS > 12 && minMllAFOS < 80 && abs(bestMTopHad-173) < 20"
#cut = "LepGood1_mcMatchId == 0 || LepGood2_mcMatchId == 0 || LepGood3_mcMatchId == 0"
el = EventLoop([ 
    cMaker, 
    #lepsXX, 
    jetsB, jetsL, jetsF, jets, 
    leps, 
    glepsH, glepsT, 
    glepsHt, glepsTt, 
    quarks25,
    quarksB, quarksH, quarksT,
    #jetsPt, 
    quarksPt,
    quarksBPt,
    cPrinter, 
])
el.loop([t], cut=cut, maxEvents=2000000)
