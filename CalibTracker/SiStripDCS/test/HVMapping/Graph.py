#! /usr/bin/env python
# Author: Jacob Herman
# Purpose: This script will make noise histograms, or histograms of various noise differences and ratios for the pedestal run info supplied by the user. Optionally an assignment dictionary can be read in to draw additional colored histos on top of the noise (or diff ect.) histo to see where assignments end up.
#TO DO: Implement legend, and double check for logic errors derived from copy pasting.


from optparse import OptionParser
import pickle, os, sys
sys.path.append('Classes/')

#Intialize command line interface
parser = OptionParser(usage ='''Usage: ./Graph.py [Options] [PickledNoise]
Examples:
   ./Graph.py 2011-TECM-ON-PEAK-000001.root 2011-TECP-ON-PEAK-000002.root
       Can plot raw noise for individually specified files
   ./Graph.py PedestalRuns/*
       Can do many files at once using bash wildcard expansions
   ./Graph.py -d PedestalRuns/*
       Some options will tell the script to make plots of noise ratios or difference
   ./Graph.py -a AssignmentDictionary.pkl 2011-TECM-ON-PEAK-000001.root 2011-TECP-ON-PEAK-000002.root
       can augment graphs (noise, diff, ect.) with colors indicating assignments''')
   


parser.add_option('-b', '--batch', action = 'store_true', dest = 'batch', help = 'Will run pyroot in batch mode with out graphics')
parser.add_option('-L', '--noiselim', type = 'float', nargs = 2, default = (0,20), dest = 'noiselim', help = "Sets  x limits on the noise histo to that passed by the user, takes two arguments with the min first, defaults to (0, 20)")
parser.add_option('-p', default = '', dest = 'path', help = "set the name of the sub-directory that the resulting histogram will be saved to in 'Ouput/<dataset>/Histos/'")
parser.add_option('-d', action = 'store_true', default = False , dest = 'Diff', help = 'Tells script to make a difference histogram')
parser.add_option('-l','--dlim', type = 'float', nargs = 2, default = (-2, 2), dest = 'difflim', help = 'Sets the x limits on the diff hist to that passed by user, takes two arguments with the min first, defaults to (-2,2)')
parser.add_option('-o','--On1', action = 'store_true', default = False, dest = 'On1', help = 'Tells script to make an ON/HV1 histogram')
parser.add_option('-i','--on1lim', type = 'float', nargs = 2, default= (0, 2), dest = 'on1lim', help = 'Sets the x limits on the on/hv1 hist to that passed by user, takes two arguments with the min first, defaults to (0,2)')
parser.add_option('-O','--On2', action = 'store_true', default = False, dest = 'On2', help = 'Tells script to make an ON/HV2 histogram')
parser.add_option('-m','--on2lim', type = 'float', nargs = 2, default=(0, 2), dest = 'on2lim', help = 'Sets the x limits on the on/hv2 hist to that passed by user, takes two arguments with the min first, defaults to (0,2)')
parser.add_option('-x','--Off1', action = 'store_true', default = False, dest = 'Off1', help = 'Tells script to make an HV1/OFF histogram')
parser.add_option('-t','--off1lim', type = 'float', nargs = 2, default=(0, 2), dest = 'off1lim', help = 'Sets the x limits on the hv1/off hist to that passed by user, takes two arguments with the min first, defaults to (0,2)')
parser.add_option('-X','--Off2', action = 'store_true', default = False, dest = 'Off2', help = 'Tells script to make an HV2/OFF histogram')
parser.add_option('-s','--off2lim', type = 'float', nargs = 2, default=(0, 2), dest = 'off2lim', help = 'Sets the x limits on the hv2/off hist to that passed by user, takes two arguments with the min first, defaults to (0,2)')
parser.add_option('-f', action = 'store_true', default = False , dest = 'OnOff', help = 'Tells script to make an (Off - On) Histogram')
parser.add_option('-I','--onofflim', type = 'float', nargs = 2, default=(-.5, 2), dest = 'onofflim', help = 'Sets the x limits on the off - on hist to that passed by user, takes two arguments with the min first, defaults to (-.5,2)')
parser.add_option('-a', dest = 'assignpath', help = 'Opens the assignment dictionary at the path passed by the user and augments the dictionary with historgrams showing the locations of assigments.')
parser.add_option('-c', dest = 'catagories', help = "Opens the Undetermined Catagories assignments at the path specified and augments On Off histogram with colors showing location of assigments including 'No-HV' and 'Cross-Talking' , must be executed with '-f'")

(Commands, args) = parser.parse_args()

if Commands.batch:
    sys.argv.append('-b')
    
from HVMapToolsClasses import HVMapNoise, HVAnalysis
from ROOT import TH1F, TCanvas, TAttLine, TLegend

#Intialize and fill noise class
Noise = HVMapNoise('Pedestal Run Info')

#format path
if not Commands.path.endswith('/') and Commands.path != '':
    Commands.path = Commands.path + '/'

#get dataset
try:
    dataset = args[0].split('/')[1]
except:
    print "Warning invalid positional argument passed:", args[0]
    
for path in args:
    Noise.AddFromPickle(path)
    try:
        if path.split('/')[1] != dataset:
            dataset = path.split('/')[1]
            print "Warning you are using noise from multiple datasets"
    except:
        print "Warning invalid positional argument passed:", path


#make directory structure if necessary
if Commands.path.replace('/','') not in os.listdir('Output/' + dataset + '/Histos/') and Commands.path != '':
    os.system('mkdir %s'%('Output/' + dataset +'/Histos/' + Commands.path))

#assemble path
Commands.path = 'Output/' + dataset+ '/Histos/' + Commands.path 

 
if Commands.Diff or Commands.On1 or Commands.On2 or Commands.Off1 or Commands.Off2 or Commands.OnOff:

    #intialize HVAnalysis with appropriate methods
    methodlist = []

    if Commands.Diff:
        methodlist.append('DIFF')

    if Commands.On1 or Commands.On2:
        methodlist.append('RON')

    if Commands.Off1 or Commands.Off1:
        methodlist.append('ROFF')

    if Commands.OnOff:
        methodlist.append('OF')

    Analysis = HVAnalysis(str(methodlist),Noise,methodlist)
        
    
    if Commands.Diff:

        canvas = TCanvas('Diff Canvas', 'Diff  Canvas',200,10,700,500)
        
        
        bins = len(Analysis.Diff.keys())/15
        DiffHist = Analysis.MkHisto('DIFF',Commands.difflim[0], Commands.difflim[1],bins,'HV1 - HV2')
        DiffHist.SetYTitle("# of APVs")
        DiffHist.SetXTitle("Noise(HV1) - Noise(HV2)")
        DiffHist.Draw()
        
        if Commands.assignpath is None:
            canvas.Print(Commands.path + 'Diff.png')

        else:
            try:
            
                file = open(Commands.assignpath)
                Assignments = pickle.load(file)

                DiffHist.SetStats(0)
                
                HV1hist = TH1F('HV1','HV1 - HV2', bins, Commands.difflim[0], Commands.difflim[1])
                HV1hist.SetLineColor(2)
                HV1hist.SetStats(0)
                
                HV2hist = TH1F('HV2','HV1 - HV2', bins, Commands.difflim[0], Commands.difflim[1])
                HV2hist.SetLineColor(4)
                HV2hist.SetStats(0)
                
                Otherhist = TH1F('Other','HV1 - HV2', bins, Commands.difflim[0], Commands.difflim[1])
                Otherhist.SetLineColor(3)
                Otherhist.SetStats(0)

                checklist = []
                for detID in Assignments.keys():
                    if detID in Analysis.Diff.keys():
                        if 'HV1' in Assignments[detID]:
                            for APVID in Analysis.Diff[detID].keys():
                                HV1hist.Fill(Analysis.Diff[detID][APVID])

                        elif 'HV2' in Assignments[detID]:
                            for APVID in Analysis.Diff[detID].keys():
                                HV2hist.Fill(Analysis.Diff[detID][APVID])

                        else:
                            for APVID in Analysis.Diff[detID].keys():
                                Otherhist.Fill(Analysis.Diff[detID][APVID])

                HV1hist.Draw("sameL")
                HV2hist.Draw("sameL")
                
                Otherhist.SetYTitle("# of APVs")
                Otherhist.SetXTitle("Noise(HV1) - Noise(HV2)")
                Otherhist.Draw("sameL")

                DiffLegend = TLegend(.6,.7,.9,.9, 'Legend')
                DiffLegend.AddEntry(DiffHist,"All","l")
                DiffLegend.AddEntry(HV1hist,"HV1", "l")
                DiffLegend.AddEntry(HV2hist, "HV2","l")
                DiffLegend.AddEntry(Otherhist, "Other","l")
                DiffLegend.Draw()
                
                canvas.Print(Commands.path + 'Diff_WithAssignments.png')

            except:
                print "Failed to make HV1 - HV2 histo with assignments"
                
    if Commands.On1:

        canvas = TCanvas('HV1 Ratio On Canvas', 'HV1 Ratio On Canvas',200,10,700,500)

        
        bins = len(Analysis.RatioOn1.keys())/15
        On1Hist = Analysis.MkHisto('ON1',Commands.on1lim[0],Commands.on1lim[1],bins,'ON/HV1')
        On1Hist.SetYTitle("# of APVs")
        On1Hist.SetXTitle("Noise(ON)/Noise(HV1)")
        On1Hist.Draw()
        
        if Commands.assignpath is None:
            canvas.Print(Commands.path + 'ON_HV1Ratio.png')

        else:
            try:
                file = open(Commands.assignpath)
                Assignments = pickle.load(file)

                On1Hist.SetStats(0)

                HV1hist = TH1F('HV1_1','ON/HV1', bins, Commands.on1lim[0], Commands.on1lim[1])
                HV1hist.SetLineColor(2)
                HV1hist.SetStats(0)
                
                HV2hist = TH1F('HV2_1','ON/HV1', bins, Commands.on1lim[0], Commands.on1lim[1])
                HV2hist.SetLineColor(4)
                HV2hist.SetStats(0)
                
                Otherhist = TH1F('Other_1','ON/HV1', bins, Commands.on1lim[0], Commands.on1lim[1])
                Otherhist.SetLineColor(3)
                Otherhist.SetStats(0)
                
                for detID in Assignments.keys():
                    if detID in Analysis.RatioOn1.keys():
                        
                        if 'HV1' in Assignments[detID]:
                            for APVID in Analysis.RatioOn1[detID].keys():
                                HV1hist.Fill(Analysis.RatioOn1[detID][APVID])

                        elif 'HV2' in Assignments[detID]:
                            for APVID in Analysis.RatioOn1[detID].keys():
                                HV2hist.Fill(Analysis.RatioOn1[detID][APVID])

                        else:
                            for APVID in Analysis.RatioOn1[detID].keys():
                                Otherhist.Fill(Analysis.RatioOn1[detID][APVID])

                HV1hist.Draw("sameL")
                HV2hist.Draw("sameL")
            
                Otherhist.SetYTitle("# of APVs")
                Otherhist.SetXTitle("Noise(ON)/Noise(HV1)")
                Otherhist.Draw("sameL")

                On1Legend = TLegend(.6,.7,.9,.9, 'Legend')
                On1Legend.AddEntry(On1Hist, "All","l")
                On1Legend.AddEntry(HV1hist,"HV1", "l")
                On1Legend.AddEntry(HV2hist,"HV2", "l")
                On1Legend.AddEntry(Otherhist,"Other", "l")
                On1Legend.Draw()
                
                canvas.Print(Commands.path + 'ON_HV1Ratio_WithAssignments.png')

            except:
                print "Failed to make ON/HV1 histo with assignments"

    if Commands.On2:

        canvas = TCanvas('HV2 Ratio On Canvas', 'HV2 Ratio On Canvas',200,10,700,500)
        
        
        bins = len(Analysis.RatioOn2.keys())/15
        On2Hist = Analysis.MkHisto('ON2',Commands.on2lim[0],Commands.on2lim[1],bins,'ON/HV2')
        On2Hist.SetYTitle("# of APVs")
        On2Hist.SetXTitle("Noise(ON)/Noise(HV2)")
        On2Hist.Draw()
        
        if Commands.assignpath is None:
            canvas.Print(Commands.path + 'ON_HV2Ratio.png')

        else:
            try:
                file = open(Commands.assignpath)
                Assignments = pickle.load(file)

                On2hHist.SetStats(0)
                
                HV1hist = TH1F('HV1_2','ON/HV2', bins, Commands.on2lim[0], Commands.on2lim[1])
                HV1hist.SetLineColor(2)
                HV1hist.SetStats(0)
                
                HV2hist = TH1F('HV2_2','ON/HV2', bins, Commands.on2lim[0], Commands.on2lim[1])
                HV2hist.SetLineColor(4)
                HV2hist.SetStats(0)
                
                Otherhist = TH1F('Other_2','ON/HV2', bins, Commands.on2lim[0], Commands.on2lim[1])
                Otherhist.SetLineColor(3)
                Otherhist.SetStats(0)
                
                for detID in Assignments.keys():
                    if detID in Analysis.RatioOn2.keys():
                        
                        if 'HV1' in Assignments[detID]:
                            for APVID in Analysis.RatioOn2[detID].keys():
                                HV1hist.Fill(Analysis.RatioOn2[detID][APVID])

                        elif 'HV2' in Assignments[detID]:
                            for APVID in Analysis.RatioOn2[detID].keys():
                                HV2hist.Fill(Analysis.RatioOn2[detID][APVID])

                        else:
                            for APVID in Analysis.RatioOn2[detID].keys():
                                Otherhist.Fill(Analysis.RatioOn2[detID][APVID])

                HV1hist.Draw("sameL")
                HV2hist.Draw("sameL")
                Otherhist.SetYTitle("# of APVs")
                Otherhist.SetXTitle("Noise(ON)/Noise(HV2)")
                Otherhist.Draw("sameL")

                On2Legend = TLegend(.6,.7,.9,.9, 'Legend')
                On2Legend.AddEntry(On2Hist, "All","l")
                On2Legend.AddEntry(HV1hist,"HV1", "l")
                On2Legend.AddEntry(HV2hist,"HV2", "l")
                On2Legend.AddEntry(Otherhist,"Other", "l")
                On2Legend.Draw()
                
                canvas.Print(Commands.path + 'ON_HV2Ratio_WithAssignments.png')

            except:
                print "Failed to make ON/HV2 histo with assignments"
                
    if Commands.Off1:

        canvas = TCanvas('HV1 Ratio Off Canvas', 'HV1 Ratio Off Canvas',200,10,700,500)
        
        
        bins = len(Analysis.RatioOn2.keys())/15
        Off1Hist = Analysis.MkHisto('OFF1',Commands.off1lim[0],Commands.off1lim[1],bins,'HV1/OFF')
        Off1Hist.SetYTitle("# of APVs")
        Off1Hist.SetXTitle("Noise(HV1)/Noise(Off)")
        Off1Hist.Draw()
        
        if Commands.assignpath is None:
            canvas.Print(Commands.path + 'OFF_HV1Ratio.png')

        else:
            try:
                file = open(Commands.assignpath)
                Assignments = pickle.load(file)

                Off1Hist.SetStats(0)
                
                HV1hist = TH1F('HV1_3','HV1/OFF', bins, Commands.off1lim[0], Commands.off1lim[1])
                HV1hist.SetLineColor(2)
                HV1hist.SetStats(0)
                
                HV2hist = TH1F('HV2_3','HV1/OFF', bins, Commands.off1lim[0], Commands.off1lim[1])
                HV2hist.SetLineColor(4)
                HV2hist.SetStats(0)
                
                Otherhist = TH1F('Other_2','HV1/OFF', bins, Commands.off1lim[0], Commands.off1lim[1])
                Otherhist.SetLineColor(3)
                Otherhist.SetStats(0)

                for detID in Assignments.keys():
                    if detID in Analysis.RatioOff1.keys():
                        
                        if 'HV1' in Assignments[detID]:
                            for APVID in Analysis.RatioOff1[detID].keys():
                                HV1hist.Fill(Analysis.RatioOff1[detID][APVID])

                        elif 'HV2' in Assignments[detID]:
                            for APVID in Analysis.RatioOff1[detID].keys():
                                HV2hist.Fill(Analysis.RatioOff1[detID][APVID])

                        else:
                            for APVID in Analysis.RatioOff1[detID].keys():
                                Otherhist.Fill(Analysis.RatioOff1[detID][APVID])

                HV1hist.Draw("sameL")
                HV2hist.Draw("sameL")
                Otherhist.SetYTitle("# of APVs")
                Otherhist.SetXTitle("Noise(HV1)/Noise(Off)")
                Otherhist.Draw("sameL")

                Off1Legend = TLegend(.6,.7,.9,.9, 'Legend')
                Off1Legend.AddEntry(Off1Hist, "All","l")
                Off1Legend.AddEntry(HV1hist,"HV1", "l")
                Off1Legend.AddEntry(HV2hist,"HV2", "l")
                Off1Legend.AddEntry(Otherhist,"Other", "l")
                Off1Legend.Draw()
                
                canvas.Print(Commands.path + 'OFF_HV1Ratio_WithAssignments.png')

            except:
                print "Failed to make HV1/OFF histo with assignments"
                
    if Commands.Off2:

        canvas = TCanvas('HV2 Ratio Off Canvas', 'HV2 Ratio Off Canvas',200,10,700,500)
        
        
        bins = len(Analysis.RatioOn2.keys())/15
        Off2Hist = Analysis.MkHisto('OFF2',Commands.off2lim[0],Commands.off2lim[1],bins,'HV2/OFF')
        Off2Hist.SetYTitle("# of APVs")
        Off2Hist.SetXTitle("Noise(HV2)/Noise(Off)")
        Off2Hist.Draw()
        
        if Commands.assignpath is None:
            canvas.Print(Commands.path + 'OFF_HV2Ratio.png')

        else:
            try:
                file = open(Commands.assignpath)
                Assignments = pickle.load(file)

                Off2Hist.SetStats(0)

                HV1hist = TH1F('HV1_4','HV2/OFF', bins, Commands.off2lim[0], Commands.off2lim[1])
                HV1hist.SetLineColor(2)
                HV1hist.SetStats(0)
                
                HV2hist = TH1F('HV2_4','HV2/OFF', bins, Commands.off2lim[0], Commands.off2lim[1])
                HV2hist.SetLineColor(4)
                HV2hist.SetStats(0)

                Otherhist = TH1F('Other_4','HV2/OFF', bins, Commands.off2lim[0], Commands.off2lim[1])
                Otherhist.SetLineColor(3)
                Otherhist.SetStats(0)
                
                for detID in Assignments.keys():
                    if detID in Analysis.RatioOff2.keys():
                        
                        if 'HV1' in Assignments[detID]:
                            for APVID in Analysis.RatioOff2[detID].keys():
                                HV1hist.Fill(Analysis.RatioOff2[detID][APVID])

                        elif 'HV2' in Assignments[detID]:
                            for APVID in Analysis.RatioOff2[detID].keys():
                                HV2hist.Fill(Analysis.RatioOff2[detID][APVID])

                        else:
                            for APVID in Analysis.RatioOff2[detID].keys():
                                Otherhist.Fill(Analysis.RatioOff2[detID][APVID])

                HV1hist.Draw("sameL")
                HV2hist.Draw("sameL")
                
                Otherhist.SetYTitle("# of APVs")
                Otherhist.SetXTitle("Noise(HV2)/Noise(Off)")
                Otherhist.Draw("sameL")

                Off2Legend = TLegend(.6,.7,.9,.9, 'Legend')
                Off2Legend.AddEntry(Off2Hist, "All","l")
                Off2Legend.AddEntry(HV1hist,"HV1", "l")
                Off2Legend.AddEntry(HV2hist,"HV2", "l")
                Off2Legend.AddEntry(Otherhist,"Other", "l")
                Off2Legend.Draw()
                
                canvas.Print(Commands.path + 'OFF_HV2Ratio_WithAssignments.png')

            except:
                print "Failed to make HV2/OFF histo with assignments"

    if Commands.OnOff:

        canvas = TCanvas('Off - On Canvas', 'Off - On Canvas',200,10,700,500)
        
        
        bins = len(Analysis.OnOffDiff.keys())/15
        OnOffHist = Analysis.MkHisto('ONOFF',Commands.onofflim[0],Commands.onofflim[1],bins,'OFF - ON')
        OnOffHist.SetXTitle("Noise(Off) - Noise(On)")
        OnOffHist.SetYTitle("# of APVs")
        OnOffHist.Draw()
        
        if Commands.assignpath is None and Commands.catagories is None:
            canvas.Print(Commands.path + 'ONOFFDiff.png')

        else:
            try:
           

                if Commands.assignpath is not None:
                    file = open(Commands.assignpath)
                    Assignments = pickle.load(file)
                    file.close()

                if Commands.catagories is not None:
                    file = open(Commands.catagories)
                    Catagories  = pickle.load(file)
                    file.close()

                OnOffHist.SetStats(0)
                
                HV1hist = TH1F('HV1_5','OFF-ON', bins, Commands.onofflim[0], Commands.onofflim[1])
                HV1hist.SetLineColor(2)
                HV1hist.SetStats(0)
                
                HV2hist = TH1F('HV2_5','OFF-ON', bins, Commands.onofflim[0], Commands.onofflim[1])
                HV2hist.SetLineColor(4)
                HV2hist.SetStats(0)
                
                NoHVhist = TH1F('NoHV','OFF-ON', bins, Commands.onofflim[0], Commands.onofflim[1])
                NoHVhist.SetLineColor(6)
                NoHVhist.SetStats(0)
                
                XTalk = TH1F('XTalk','OFF-ON', bins, Commands.onofflim[0], Commands.onofflim[1])
                XTalk.SetLineColor(7)
                XTalk.SetStats(0)
                
                Otherhist = TH1F('Other','OFF - ON', bins, Commands.onofflim[0], Commands.onofflim[1])
                Otherhist.SetLineColor(3)
                Otherhist.SetStats(0)

                if 'Assignments' in dir():
                    for detID in Assignments.keys():
                        if detID in Analysis.OnOffDiff.keys():
                            if 'HV1' in Assignments[detID]:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    HV1hist.Fill(Analysis.OnOffDiff[detID][APVID])

                            elif 'HV2' in Assignments[detID]:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    HV2hist.Fill(Analysis.OnOffDiff[detID][APVID])

                            else:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    Otherhist.Fill(Analysis.OnOffDiff[detID][APVID])

                if 'Catagories' in dir():
                    for detID in Catagories.keys():
                        if detID in Analysis.OnOffDiff.keys():
                            if 'HV1' in Catagories[detID]:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    HV1hist.Fill(Analysis.OnOffDiff[detID][APVID])

                            elif 'HV2' in Catagories[detID]:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    HV2hist.Fill(Analysis.OnOffDiff[detID][APVID])

                    
                            elif 'NO-HV' in Catagories[detID] and Commands.catagories is not None:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    NoHVhist.Fill(Analysis.OnOffDiff[detID][APVID])

                            elif 'Cross-Talking' in Catagories[detID] and Commands.catagories is not None:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    XTalk.Fill(Analysis.OnOffDiff[detID][APVID])
                            
                            else:
                                for APVID in Analysis.OnOffDiff[detID].keys():
                                    Otherhist.Fill(Analysis.OnOffDiff[detID][APVID])

                OnOffLegend = TLegend(.6,.7,.9,.9, 'Legend')
                
                if Commands.assignpath is not None:
                    HV1hist.Draw("sameL")
                    HV2hist.Draw("sameL")
                    Otherhist.SetXTitle("Noise(Off) - Noise(On)")
                    Otherhist.SetYTitle("# of APVs")
                    Otherhist.Draw("sameL")

               
                    
                    OnOffLegend.AddEntry(OnOffHist, "All","l")
                    OnOffLegend.AddEntry(HV1hist,"HV1", "l")
                    OnOffLegend.AddEntry(HV2hist,"HV2", "l")
                    OnOffLegend.AddEntry(Otherhist,"Other", "l")

                if Commands.catagories is not None:
                    if Commands.assignpath is None:
                        HV1hist.Draw("sameL")
                        HV2hist.Draw("sameL")
                        Otherhist.SetXTitle("Noise(Off) - Noise(On)")
                        Otherhist.SetYTitle("# of APVs")
                        Otherhist.Draw("sameL")

               
                    
                        OnOffLegend.AddEntry(OnOffHist, "All","l")
                        OnOffLegend.AddEntry(HV1hist,"HV1", "l")
                        OnOffLegend.AddEntry(HV2hist,"HV2", "l")
                        OnOffLegend.AddEntry(Otherhist,"Other", "l")


                    OnOffLegend.AddEntry(XTalk,"Cross Talking", "l")
                    OnOffLegend.AddEntry(NoHVhist,"No HV", "l")

                    NoHVhist.Draw("sameL")
                    XTalk.SetXTitle("Noise(Off) - Noise(On)")
                    XTalk.SetYTitle("# of APVs")
                    XTalk.Draw("sameL")
                    
                OnOffLegend.Draw()

                if Commands.assignpath is not None and Commands.catagories is not None:
                    canvas.Print(Commands.path + 'ONOFFDIFF_WithAssignmentsAndCatagories.png')
                elif Commands.assignpath is not None:
                    canvas.Print(Commands.path + 'ONOFFDIFF_WithAssignments.png')
                else:
                    canvas.Print(Commands.path + 'ONOFFDIFF_WithAssignmentCatagories.png')

            except:
                print "Failed to make OFF - ON histogram with catagories or assignments"

else:

    #intialize histo

    canvas = TCanvas('Graph Canvas', 'Graph Canvas',200,10,700,500)
    
    hist = Noise.IntializeHisto(Commands.noiselim[0], Commands.noiselim[1], 700, 'Pedestal Noise')

    for RunT in Noise.Noise.keys():
        Noise.AddToHisto(hist,RunT,'All')
    hist.SetXTitle("Pedestal Noise")
    hist.SetYTitle("# of APVs")
    hist.Draw()

    if Commands.assignpath is None:
        canvas.Print(Commands.path + 'PedestalNoise.png')

    else:
        try:
        
            file = open(Commands.assignpath)
            assign = pickle.load(file)

            hist.SetStats(0)
            
            hist1 = Noise.IntializeHisto(Commands.noiselim[0], Commands.noiselim[1], 700,'Pedestal Noise1')

            for RunT in Noise.Noise.keys():
                Noise.AddToHisto(hist1, RunT, 'All', 2, 'HV1',assign)

            hist.SetStats(0)
            hist1.Draw("sameL")
            
            hist2 = Noise.IntializeHisto(Commands.noiselim[0], Commands.noiselim[1], 700, 'Pedestal Noise2')

            for RunT in Noise.Noise.keys():
                Noise.AddToHisto(hist2, RunT , 'All', 4,'HV2', assign)

            hist2.SetStats(0)
            hist2.Draw("sameL")

            histU = Noise.IntializeHisto(Commands.noiselim[0], Commands.noiselim[1], 700, 'Pedestal Noise and Assignments')

            for RunT in Noise.Noise.keys():
                Noise.AddToHisto(histU, RunT , 'All', 3,'Undetermined', assign)

            histU.SetStats(0)
            histU.SetXTitle("Pedestal Noise")
            histU.SetYTitle("# of APVs")
            histU.Draw("sameL")

            Legend = TLegend(.6,.7,.9,.9, 'Legend')
            Legend.AddEntry(hist, "All","l")
            Legend.AddEntry(hist1,"HV1", "l")
            Legend.AddEntry(hist2,"HV2", "l")
            Legend.AddEntry(histU,"Other", "l")
            Legend.Draw()

            canvas.Print(Commands.path + "PedestalNoise_WithAssignments.png")

        except:
            print "Failed to draw Pedestal Noise with assignments"
    
