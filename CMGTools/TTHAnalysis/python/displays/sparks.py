#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.plotter.tree2yield import CutsFile, scalarToVector
from optparse import OptionParser

import os, os.path, ROOT, string
from math import *

if "/functions_cc.so" not in ROOT.gSystem.GetLibraries(): 
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/functions.cc+" % os.environ['CMSSW_BASE']);

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
            if self._cut and not eval(self._cut, globals(), o): 
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
    def __init__(self,name,maxEvents=100,maxEta=5):
        Module.__init__(self,name,booker=None)
        ROOT.gROOT.ProcessLine(".x %s/src/CMGTools/TTHAnalysis/python/plotter/tdrstyle.cc" % os.environ['CMSSW_BASE']);
        self._canvas = ROOT.TCanvas("c1","c1");
        self._canvas.SetGridx(1)
        self._canvas.SetGridy(1)
        self._frame = ROOT.TH1F("frame","frame;#eta;#phi",1,-maxEta,maxEta);
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
        self._glep = Collection(event,"genLep")
        self._gtau = Collection(event,"genLepFromTau")
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


def addSparksOptions(parser):
    parser.add_option("-c", "--cut-file",  dest="cutfile", default=None, type="string", help="Cut file to apply")
    parser.add_option("-C", "--cut",  dest="cut", default=None, type="string", help="Cut to apply")
    parser.add_option("-n", "--maxEvents",  dest="maxEvents", default=-1, type="int", help="Max events")
    parser.add_option("-N", "--maxEventsToPrint",  dest="maxEventsToPrint", default=40, type="int", help="Max events")
    parser.add_option("-t", "--tree",          dest="tree", default='ttHLepTreeProducerTTH', help="Pattern for tree name");
    parser.add_option("-V", "--vector",  dest="vectorTree", action="store_true", default=True, help="Input tree is a vector");
    parser.add_option("-F", "--add-friend",    dest="friendTrees",  action="append", default=[], nargs=2, help="Add a friend tree (treename, filename). Can use {name}, {cname} patterns in the treename") 
    parser.add_option("--pdir", "--print-dir", dest="pdir", type="string", default="plots", help="print out plots in this directory");
    ### CUT-file options
    parser.add_option("-S", "--start-at-cut",   dest="startCut",   type="string", help="Run selection starting at the cut matched by this regexp, included.") 
    parser.add_option("-U", "--up-to-cut",      dest="upToCut",   type="string", help="Run selection only up to the cut matched by this regexp, included.") 
    parser.add_option("-X", "--exclude-cut", dest="cutsToExclude", action="append", default=[], help="Cuts to exclude (regexp matching cut name), can specify multiple times.") 
    parser.add_option("-I", "--invert-cut",  dest="cutsToInvert",  action="append", default=[], help="Cuts to invert (regexp matching cut name), can specify multiple times.") 
    parser.add_option("-R", "--replace-cut", dest="cutsToReplace", action="append", default=[], nargs=3, help="Cuts to invert (regexp of old cut name, new name, new cut); can specify multiple times.") 
    parser.add_option("-A", "--add-cut",     dest="cutsToAdd",     action="append", default=[], nargs=3, help="Cuts to insert (regexp of cut name after which this cut should go, new name, new cut); can specify multiple times.") 
    parser.add_option("--s2v", "--scalar2vector",     dest="doS2V",    action="store_true", default=False, help="Do scalar to vector conversion") 

if __name__ == "__main__": 
    parser = OptionParser(usage="usage: %prog [options] rootfile [what] \nrun with --help to get list of options")
    addSparksOptions(parser)
    (options, args) = parser.parse_args()
    f = ROOT.TFile.Open("%s/%s/%s_tree.root" % (args[0], options.tree, options.tree))
    name = os.path.basename(args[0])
    t = f.Get(options.tree)
    t.vectorTree = options.vectorTree
    t.friends = []
    for tf_tree,tf_file in options.friendTrees:
        tf = t.AddFriend(tf_tree, tf_file.format(cname=name,name=name))
        t.friends.append(tf) # to make sure pyroot does not delete them
    cut = None
    if options.cutfile:
        cut = CutsFile(options.cutfile,options).allCuts()
    elif options.cut:
        cut = options.cut
    if options.doS2V:
        cut = scalarToVector(cut)
    print "Reading %s (%d entries)" % (args[0], t.GetEntries())
    if not os.path.exists((options.pdir)):
        print "mkdir?"
        os.system("mkdir -p "+(options.pdir))
        if os.path.exists("/afs/cern.ch"): os.system("cp /afs/cern.ch/user/g/gpetrucc/php/index.php "+(options.pdir))
    ### CONFIG
    cMaker   = CanvasMaker("cm", maxEvents=options.maxEventsToPrint, maxEta=3)
    cPrinter = CanvasPrinter("cp",cMaker,[ options.pdir+"/r{run}_ls{lumi}_ev{evt}.png" ])
    leps  = MarkerDrawer("leps","LepGood", markerColor=62)
    jetsL = CircleDrawer("leps","Jet", 0.4, lineColor=92,  lineStyle=1, cut="btagCSV <= 0.679")
    jetsB = CircleDrawer("leps","Jet", 0.4, lineColor=100, lineStyle=1, cut="btagCSV > 0.679")
    jetsPt    = LabelDrawer("leps","Jet", '{o.pt:.0f}', textColor=4,xoffs=0.0,yoffs=0.0, textSize=0.025)
    gleps = MarkerDrawer("leps","genLep", markerStyle=34, markerSize=2.0, markerColor=214)#, cut="mcMatchId > 0")
    SVs_soft  = MarkerDrawer("svs","SV", markerColor=ROOT.kMagenta+1,cut="pt > 5 and fabs(dxy) < 2.5 and jetPt < 25 and mva > 0.7")
    SVs_untag = MarkerDrawer("svs","SV", markerColor=ROOT.kRed-3,cut="pt > 5 and fabs(dxy) < 2.5 and (jetBTag < 0.679 and jetPt > 25) and mva > 0.7")
    SVs_hard  = MarkerDrawer("svh","SV", markerColor=ROOT.kMagenta-8,cut="pt > 5 and fabs(dxy) < 2.5 and (jetBTag > 0.679 and jetPt > 25) and mva > 0.7")
    SVs_soft  = MarkerDrawer("svs","SV", markerColor=ROOT.kMagenta+1,cut="pt > 5 and fabs(dxy) < 2.5 and jetPt < 25 and mva > 0.7")
    SVs_true  = MarkerDrawer("svt","SV", markerStyle=34, markerSize=1.0, markerColor=ROOT.kMagenta+3, cut="mcFlavHeaviest == 5")
    SVs_vals  = LabelDrawer("svl","SV", '{o.pt:.1f}/{o.mva:.2f}', textColor=1,xoffs=0.02,yoffs=0.0, textSize=0.025, cut="mcFlavHeaviest == 5 and pt > 3 and jetPt < 40")
    #SVs_2     = LabelDrawer("svl","SV", '{o.jetPt:.0f}', textColor=2,xoffs=-0.05,yoffs=-0.05, textSize=0.025, cut="pt > 5 and fabs(dxy) < 2.5 and jetPt < 25 and mva > 0.7")
    el = EventLoop([ 
        cMaker, 
        jetsB, jetsL, 
        leps, 
        gleps, 
        SVs_hard,SVs_untag,SVs_soft,SVs_true,SVs_vals,#SVs_2,
        jetsPt, 
        cPrinter, 
    ])
    el.loop([t], cut=cut, maxEvents=options.maxEvents)
