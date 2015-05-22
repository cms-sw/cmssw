#!/usr/bin/env python
from CMGTools.TTHAnalysis.displays.sparks import *

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
        os.system("mkdir -p "+(options.pdir))
        if os.path.exists("/afs/cern.ch"): os.system("cp /afs/cern.ch/user/g/gpetrucc/php/index.php "+(options.pdir))
    ### CONFIG
    cMaker   = CanvasMaker("cm", maxEvents=options.maxEventsToPrint, maxEta=3)
    cPrinter = CanvasPrinter("cp",cMaker,[ options.pdir+"/r{run}_ls{lumi}_ev{evt}.png" ])
    leps  = MarkerDrawer("leps","LepGood", markerColor=ROOT.kAzure+2, markerSize=0.8)
    top   = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kRed+2,    markerSize=2.6, markerStyle=24, cut="abs(pdgId)==6")
    stop  = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kGreen+2,  markerSize=2.6, markerStyle=25, cut="abs(pdgId)==1000006")
    btop  = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kRed+2,    markerSize=2.0, markerStyle=33, cut="abs(pdgId)==5 and abs(motherId)==6")
    bstop = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kGreen+3,  markerSize=2.0, markerStyle=33, cut="abs(pdgId)==5 and abs(motherId)==1000006")
    wtop  = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kOrange+8, markerSize=2.3, markerStyle=27, cut="abs(pdgId)==24 and abs(motherId)==6")
    wdec  = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kRed+1,    markerSize=1.3, markerStyle=20, cut="abs(motherId)==24 and charge != 0")
    xdec  = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kGreen-3,  markerSize=1.3, markerStyle=20, cut="abs(motherId)==1000024 and pdgId != 1000022")
    isr   = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kGray+1,   markerSize=1.3, markerStyle=20, cut="(abs(pdgId) <= 5 or abs(pdgId)==21) and (abs(motherId) != 6 and abs(motherId) < 24)")
    sdec  = MarkerDrawer("leps","GenP6StatusThree", markerColor=ROOT.kViolet,   markerSize=1.3, markerStyle=20, cut="(abs(pdgId) <= 5 or abs(pdgId)==21) and (abs(motherId) == 1000021)")
    jets8  = CircleDrawer("leps","FatJet", 0.8, lineColor=ROOT.kOrange+7,  lineStyle=1, cut="")
    jets   = CircleDrawer("leps","Jet", 0.4, lineColor=ROOT.kPink-6,    lineStyle=1, cut="")
    jetsM  = CircleDrawer("leps","Jet", 0.44, lineColor=ROOT.kRed,    lineStyle=2, cut="mass > 40 and pt > 70")
    jetsT  = LabelDrawer("leps","Jet", '{o.pt:.0f}',          textColor=ROOT.kPink-6,xoffs=-0.0,yoffs=-0.0, textSize=0.02)
    jets8T = LabelDrawer("leps","FatJet", '{o.pt:.0f},{o.prunedMass:.0f}', textColor=ROOT.kOrange+9,xoffs=0.,yoffs=+0.85, textSize=0.025)
    jetsMT = LabelDrawer("leps","Jet", '{o.pt:.0f},{o.mass:.0f}', textColor=ROOT.kRed,xoffs=0.,yoffs=+0.50, textSize=0.025, cut="mass > 40 and pt > 70")
    el = EventLoop([ 
        cMaker, 
        jets8, 
        jets, jetsM,
        top,stop,btop,bstop,wtop,wdec,xdec,isr,sdec,
        leps, 
        jets8T, jetsT, jetsMT,
        cPrinter, 
    ])
    el.loop([t], cut=cut, maxEvents=options.maxEvents)
