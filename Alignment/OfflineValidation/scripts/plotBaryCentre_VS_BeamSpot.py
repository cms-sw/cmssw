#!/usr/bin/env python3

import sys, os
from array import array
import optparse
from collections import OrderedDict
import json

import ROOT
ROOT.gSystem.Load("libFWCoreFWLite.so")

import CondCore.Utilities.conddblib as conddb
import Alignment.OfflineValidation.TkAlAllInOneTool.findAndChange as fnc

# 1/lumiScaleFactor to go from 1/pb to 1/fb
lumiScaleFactor = 1000

grootargs = []
def callback_rootargs(option, opt, value, parser):
    grootargs.append(opt)

def vararg_callback(option, opt_str, value, parser):
    assert value is None
    value = []

    def floatable(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not floatable(arg):
            break
        value.append(arg)

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)

def parseOptions():
    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    parser.add_option("--inputFileName", dest="inputFileName", default="PixelBaryCentre.root",help="name of the ntuple file that contains the barycentre tree")
    parser.add_option("--plotConfigFile", dest="plotConfigFile", default="PixelBaryCentrePlotConfig.json",help="json file that configs the plotting")

    parser.add_option("--usePixelQuality",action="store_true", dest="usePixelQuality", default=False,help="whether use SiPixelQuality")
    parser.add_option("--showLumi",action="store_true", dest="showLumi", default=False,help="whether use integrated lumi as x-axis")
    parser.add_option("--years", dest="years", default = [2017], action="callback", callback=vararg_callback, help="years to plot")

    parser.add_option("-l",action="callback",callback=callback_rootargs)
    parser.add_option("-q",action="callback",callback=callback_rootargs)
    parser.add_option("-b",action="callback",callback=callback_rootargs)

    return parser


def findRunIndex(run, runs) :
    #runs has to be sorted
    if(len(runs)==0) :
      print("Empty run list!")
      return -1
    elif(len(runs)==1) :
       if(run>=runs[0]) :
         return 0
       else :
         print("Only one run but the run requested is before the run!")
         return -1
    else :
       # underflow
       if(run <= runs[0]) :
          return 0
       # overflow
       elif(run >= runs[len(runs)-1]) :
          return len(runs)-1
       else :
          return ROOT.TAxis(len(runs)-1,array('d',runs)).FindBin(run) - 1


def readBaryCentreAnalyzerTree(t, branch_names, accumulatedLumiPerRun, showLumi, isEOY) :
    # to store lumi sections info for each run
    run_maxlumi = {}
    run_lumis = {}

    # y-axis of TGraph
    # runs for all years // integrated luminosity as a function of run
    runs = list(accumulatedLumiPerRun.keys())
    runs.sort()

    current_run = 0
    for iov in t :
        # skip runs out-of-range
        if(iov.run>runs[len(runs)-1] or iov.run<runs[0]):
          continue

        if(iov.run!=current_run) : # a new run, initialize lumi sections
          run_lumis[iov.run] = [iov.ls]
          run_maxlumi[iov.run] = iov.ls
        else : # current run, append lumi sections
          run_lumis[iov.run].append(iov.ls)
          if(run_maxlumi[iov.run]<iov.ls):
             run_maxlumi[iov.run] = iov.ls
        # update current run
        current_run = iov.run

    # initialize store barycentre
    pos = {}
    for branch_name in branch_names :
        for coord in ["x","y","z"] :
            pos[coord+"_"+branch_name] = array('d',[])
            # max and min to determine the plotting range
            pos[coord+"max_"+branch_name] = -9999
            pos[coord+"min_"+branch_name] = 9999

    # y-errors
    zeros = array('d',[])

    # x-axis of TGraph
    runlumi = array('d',[])
    runlumiplot = array('d',[])
    runlumiplot_error = array('d',[])

    max_run = 0
    # loop over IOVs
    for iov in t :
        # skip runs out-of-range
        if(iov.run>runs[len(runs)-1] or iov.run<runs[0]):
          continue
        # exclude 2018D for EOY rereco
        if(isEOY and iov.run>=320413 and iov.run<=325175):
          continue

        # if x-axis is luminosity
        if(showLumi) :
          run_index = findRunIndex(iov.run,runs)
          instLumi = 0
          if(run_index==0) :
            instLumi = accumulatedLumiPerRun[ runs[run_index] ]
          if(run_index>0) :
            instLumi = accumulatedLumiPerRun[ runs[run_index] ] - accumulatedLumiPerRun[ runs[run_index-1] ]
          # remove runs with zero luminosity if x-axis is luminosity
          if( instLumi==0 ) : #and accumulatedLumiPerRun[ runs[run_index] ]==0 ) :
            continue

          if(len(run_lumis[iov.run])>1) :  # lumi-based conditions
            if(run_index==0) :
              runlumi.append(0.0+instLumi*iov.ls*1.0/run_maxlumi[iov.run])
            else :
              runlumi.append(accumulatedLumiPerRun[ runs[run_index-1] ]+instLumi*iov.ls*1.0/run_maxlumi[iov.run])

          else : # run-based or only one-IOV in the run
              runlumi.append(accumulatedLumiPerRun[ runs[run_index] ])

        else: # else x-axis is run number
          if(len(run_lumis[iov.run])>1) :#lumi-based conditions
               runlumi.append(iov.run+iov.ls*1.0/run_maxlumi[iov.run])

          else : # run-based or only one-IOV in the run
               runlumi.append(iov.run)

        zeros.append(0)

        #10000 is to translate cm to micro-metre
        for branch_name in branch_names :
            pos_ = {"x":10000*getattr(iov, branch_name).x(),
                    "y":10000*getattr(iov, branch_name).y(),
                    "z":10000*getattr(iov, branch_name).z()}

            for coord in ["x","y","z"] :
                pos[coord+"_"+branch_name].append(pos_[coord])
                # max/min
                if(pos_[coord]>pos[coord+"max_"+branch_name]) :
                   pos[coord+"max_"+branch_name] = pos_[coord]
                   max_run = iov.run
                if(pos_[coord]<pos[coord+"min_"+branch_name]) :
                   pos[coord+"min_"+branch_name] = pos_[coord]

    # x-axis : run/lumi or integrtated luminosity
    for iov in range(len(runlumi)-1) :
        runlumiplot.append(0.5*(runlumi[iov]+runlumi[iov+1]))
        runlumiplot_error.append(0.5*(runlumi[iov+1]-runlumi[iov]))

    runlumiplot.append(runlumiplot[len(runlumiplot_error)-1]+2*runlumiplot_error[len(runlumiplot_error)-1])
    runlumiplot_error.append(runlumiplot_error[len(runlumiplot_error)-1])

    v_runlumiplot = ROOT.TVectorD(len(runlumiplot),runlumiplot)
    v_runlumiplot_error = ROOT.TVectorD(len(runlumiplot_error),runlumiplot_error)

    # y-axis error
    v_zeros = ROOT.TVectorD(len(zeros),zeros)

    # store barycentre into a dict
    barryCentre = {}
    v_pos = {}
    for branch_name in branch_names :
        for coord in ["x","y","z"] :
            v_pos[coord] = ROOT.TVectorD(len(pos[coord+"_"+branch_name]),pos[coord+"_"+branch_name])

            barryCentre[coord+'_'+branch_name] = ROOT.TGraphErrors(v_runlumiplot, v_pos[coord], v_runlumiplot_error, v_zeros)
            barryCentre['a_'+coord+'_'+branch_name] = pos[coord+"_"+branch_name]

            barryCentre[coord+'max_'+branch_name] = pos[coord+"max_"+branch_name]
            barryCentre[coord+'min_'+branch_name] = pos[coord+"min_"+branch_name]

    barryCentre['v_runlumiplot'] = v_runlumiplot
    barryCentre['v_runlumierror'] = v_runlumiplot_error
    barryCentre['v_zeros'] = v_zeros

    return barryCentre


def blackBox(x1, y1, x2, y2):
    x = array('d',[x1, x2, x2, x1, x1])
    y = array('d',[y1, y1, y2, y2, y1])
    v_x = ROOT.TVectorD(len(x),x)
    v_y = ROOT.TVectorD(len(y),y)

    gr = ROOT.TGraph(v_x,v_y)
    gr.SetLineColor(ROOT.kBlack)

    return gr


def plotbarycenter(bc,coord,plotConfigJson, substructure,runsPerYear,pixelLocalRecos,accumulatedLumiPerRun, withPixelQuality,showLumi) :
    runs = list(accumulatedLumiPerRun.keys())
    runs.sort()
    years = list(runsPerYear.keys())
    years.sort()
    labels = list(bc.keys())

    can = ROOT.TCanvas("barycentre_"+substructure+"_"+coord, "", 2000, 900)
    can.cd()

    range_ = 0
    width_ = 0
    upper = 0
    lower = 0
    xmax = 0

    gr = {}
    firstGraph = True
    for label in labels :
        gr[label] = ROOT.TGraph()
        gr[label] = bc[label][coord+"_"+substructure]

        gr[label].SetMarkerStyle(8)
        gr[label].SetMarkerSize(0)
        gr[label].SetMarkerStyle(8)
        gr[label].SetMarkerSize(0)
        gr[label].SetLineColor(plotConfigJson["colorScheme"][label])

        width_ = gr[label].GetXaxis().GetXmax() - gr[label].GetXaxis().GetXmin()
        xmax   = gr[label].GetXaxis().GetXmax()

        if firstGraph :
           upper  = bc[label][coord+"max_"+substructure]
           lower  = bc[label][coord+"min_"+substructure]
           firstGraph = False
        else :
           upper = max(upper, bc[label][coord+"max_"+substructure])
           lower = min(lower, bc[label][coord+"min_"+substructure])

    scale = 1.1
    if(upper>0) :
      upper = upper * scale
    else :
      upper = upper / scale
    if(lower>0) :
      lower = lower / scale
    else :
      lower = lower * scale
    range_ = upper - lower

    firstGraph = True
    for label in labels :
        if(firstGraph) :
            gr[label].GetYaxis().SetRangeUser(lower, upper)
            gr[label].GetYaxis().SetTitle(plotConfigJson["substructures"][substructure]+" barycentre ("+coord+") [#mum]")
            gr[label].GetXaxis().SetTitle("Run Number")
            gr[label].GetYaxis().CenterTitle(True)
            gr[label].GetXaxis().CenterTitle(True)
            gr[label].GetYaxis().SetTitleOffset(0.80)
            gr[label].GetYaxis().SetTitleSize(0.055)
            gr[label].GetXaxis().SetTitleOffset(0.80)
            gr[label].GetXaxis().SetTitleSize(0.055)
            gr[label].GetXaxis().SetMaxDigits(6)
            if(showLumi) :
                gr[label].GetXaxis().SetTitle("Delivered luminosity [1/fb]")

            gr[label].Draw("AP")
            firstGraph = False
        else :
            gr[label].Draw("P")

    # dummy TGraph for pixel local reco changes and first-of-year Run
    gr_dummyFirstRunOfTheYear = blackBox(-999, 10000, -999, -10000)
    gr_dummyFirstRunOfTheYear.SetLineColor(ROOT.kBlack)
    gr_dummyFirstRunOfTheYear.SetLineStyle(1)
    gr_dummyFirstRunOfTheYear.Draw("L")
    gr_dummyPixelReco = blackBox(-999, 10000, -999, -10000)
    gr_dummyPixelReco.SetLineColor(ROOT.kGray+1)
    gr_dummyPixelReco.SetLineStyle(3)
    gr_dummyPixelReco.Draw("L")
    gr_dummyFirstRunOfTheYear.SetTitle("First run of the year")
    gr_dummyPixelReco.SetTitle("Pixel calibration update")

    for label in labels :
        gr[label].SetTitle(plotConfigJson["baryCentreLabels"][label])
    legend = can.BuildLegend()#0.65, 0.65, 0.85, 0.85)
    legend.SetShadowColor(0)
    legend.SetFillColor(0)
    legend.SetLineColor(1)

    for label in labels :
        gr[label].SetTitle("")

    # Add legends
    # and vertical lines
    years_label = ""
    for year in years :
        years_label += str(year)
        years_label += "+"
    years_label = years_label.rstrip("+")

    # CMS logo
    CMSworkInProgress = ROOT.TPaveText( xmax-0.3*width_, upper+range_*0.005,
                                        xmax, upper+range_*0.055, "nb")
    CMSworkInProgress.AddText("CMS #bf{#it{Preliminary} ("+years_label+" pp collisions)}")
    CMSworkInProgress.SetTextAlign(32) #right/bottom aligned
    CMSworkInProgress.SetTextSize(0.04)
    CMSworkInProgress.SetFillColor(10)
    CMSworkInProgress.Draw()

    # vertical lines
    #pixel local reco
    line_pixels = {}
    for since in pixelLocalRecos :
        if showLumi :
           run_index = findRunIndex(since,runs)
           integrated_lumi = accumulatedLumiPerRun[runs[run_index]]
           line_pixels[since] = ROOT.TLine(integrated_lumi, lower, integrated_lumi, upper)

        else :
           line_pixels[since] = ROOT.TLine(since, lower, since, upper)

        line_pixels[since].SetLineColor(ROOT.kGray+1)
        line_pixels[since].SetLineStyle(3)
        line_pixels[since].Draw()

    # years
    line_years = {}
    box_years = {}
    text_years = {}
    if(len(years)>1 or (not showLumi) ) : # indicate begining of the year if more than one year to show or use run number
      for year in years :
          if showLumi :
             #first run of the year
             run_index = findRunIndex(runsPerYear[year][0],runs)
             integrated_lumi = accumulatedLumiPerRun[runs[run_index]]
             line_years[year] = ROOT.TLine(integrated_lumi, lower, integrated_lumi, upper)
             text_years[year] = ROOT.TPaveText( integrated_lumi+0.01*width_, upper-range_*0.05,
                                              integrated_lumi+0.05*width_,  upper-range_*0.015, "nb")
             box_years[year] = blackBox(integrated_lumi+0.005*width_, upper-range_*0.01, integrated_lumi+0.055*width_, upper-range_*0.055)
          else :
             line_years[year] = ROOT.TLine(runsPerYear[year][0], lower, runsPerYear[year][0], upper)
             text_years[year] = ROOT.TPaveText( runsPerYear[year][0]+0.01*width_, upper-range_*0.05,
                                              runsPerYear[year][0]+0.05*width_,  upper-range_*0.015, "nb")
             box_years[year] = blackBox(runsPerYear[year][0]+0.01*width_, upper-range_*0.015, runsPerYear[year][0]+0.05*width_, upper-range_*0.05)


          box_years[year].Draw("L")
          line_years[year].Draw()

          # Add TextBox at the beginning of each year
          text_years[year].AddText(str(year))
          text_years[year].SetTextAlign(22)
          text_years[year].SetTextSize(0.025)
          text_years[year].SetFillColor(10)
          text_years[year].Draw()

    #legend.Draw()
    can.Update()

    if(showLumi) :
      can.SaveAs("baryCentre"+withPixelQuality+"_"+coord+"_"+substructure+"_"+years_label+"_IntegratedLumi.pdf")
      can.SaveAs("baryCentre"+withPixelQuality+"_"+coord+"_"+substructure+"_"+years_label+"_IntegratedLumi.png")
    else :
      can.SaveAs("baryCentre"+withPixelQuality+"_"+coord+"_"+substructure+"_"+years_label+"_RunNumber.pdf")
      can.SaveAs("baryCentre"+withPixelQuality+"_"+coord+"_"+substructure+"_"+years_label+"_RunNumber.png")

    #####################################################################################################################


# main call
def Run():

    #ROOT.gSystem.Load("libFWCoreFWLite.so")
    parser=parseOptions()
    (options,args) = parser.parse_args()
    sys.argv = grootargs

    inputFileName = options.inputFileName
    if os.path.isfile(inputFileName) == False :
       print ("File "+inputFileName+" not exist!")
       return -1

    plotConfigFile = open(options.plotConfigFile)
    plotConfigJson = json.load(plotConfigFile)
    plotConfigFile.close()

    usePixelQuality = options.usePixelQuality
    withPixelQuality = ""
    if(usePixelQuality) :
       withPixelQuality = "WithPixelQuality"
    showLumi = options.showLumi
    # order years from old to new
    years = options.years
    years.sort()

    # runs per year
    runsPerYear = {}
    for year in years :
        runsPerYear[year] = []
    # integrated lumi vs run
    accumulatedLumiPerRun = {}

    run_index = 0
    lastRun = 1

    # get lumi per IOV
    for year in years :
        inputLumiFile = fnc.digest_path("Alignment/OfflineValidation/data/lumiperrun"+str(year)+".txt")
        if os.path.isfile(inputLumiFile) == False :
           print ("File "+inputLumiFile+" not exist!")
           return -1
        lumiFile = open(inputLumiFile,'r')
        lines = lumiFile.readlines()

        for line in lines :
            # line = "run inst_lumi"
            run = int(line.split()[0])
            integrated_lumi = float(line.split()[1])/lumiScaleFactor # 1/pb to 1/fb

            # runs per year
            runsPerYear[year].append(run)
            # integrated luminosity per run
            # run number must be ordered from small to large in the text file
            if(run_index == 0) :
              accumulatedLumiPerRun[run] = integrated_lumi
            else :
              accumulatedLumiPerRun[run] = accumulatedLumiPerRun[lastRun]+integrated_lumi

            run_index+=1
            lastRun = run

        # close file
        lumiFile.close()

    # order by key (year)
    runsPerYear = OrderedDict(sorted(runsPerYear.items(), key=lambda t: t[0]))
    # order by key (run number)
    accumulatedLumiPerRun = OrderedDict(sorted(accumulatedLumiPerRun.items(), key=lambda t: t[0]))

    #pixel local reco update (IOVs/sinces)
    pixelLocalRecos = []
    # connnect to ProdDB to access pixel local reco condition change
    db = plotConfigJson["pixelDataBase"]
    pixel_template = plotConfigJson["pixelLocalReco"]
    db = db.replace("sqlite_file:", "").replace("sqlite:", "")
    db = db.replace("frontier://FrontierProd/CMS_CONDITIONS", "pro")
    db = db.replace("frontier://FrontierPrep/CMS_CONDITIONS", "dev")

    con = conddb.connect(url = conddb.make_url(db))
    session = con.session()
    # get IOV table
    IOV = session.get_dbtype(conddb.IOV)
    iovs = set(session.query(IOV.since).filter(IOV.tag_name == pixel_template).all())
    session.close()
    pixelLocalRecos = sorted([int(item[0]) for item in iovs])
    #pixelLocalRecos = [1, 186500, 195360, 197749, 200961, 203368, 204601, 206446, 238341, 246866, 253914, 255655, 271866, 276315, 278271, 280928, 290543, 297281, 298653, 299443, 300389, 301046, 302131, 303790, 303998, 304911, 313041, 314881, 316758, 317475, 317485, 317527, 317661, 317664, 318227, 320377, 321831, 322510, 322603, 323232, 324245]

    # substructures to plot
    substructures = list(plotConfigJson["substructures"].keys())

    # start barycentre plotter
    bc = {}
    try:
       f = ROOT.TFile(inputFileName,"READ")
       # read TTrees
       for label in list(plotConfigJson["baryCentreLabels"].keys()) :
           isEOY = False
           t = ROOT.TTree()
           if label == "" :
              t = f.Get("PixelBaryCentreAnalyzer"+withPixelQuality+"/PixelBarycentre")
           else :
              t = f.Get("PixelBaryCentreAnalyzer"+withPixelQuality+"/PixelBarycentre_"+label)
              if(label=="EOY") :
                 isEOY = True

           bc[label] = readBaryCentreAnalyzerTree(t, substructures, accumulatedLumiPerRun, showLumi, isEOY)

    except IOError:
       print("File "+inputFileName+" not accessible")

    # plot
    for substructure in substructures :
        for coord in ['x','y','z'] :
            plotbarycenter(bc,coord,plotConfigJson,substructure, runsPerYear,pixelLocalRecos,accumulatedLumiPerRun, withPixelQuality,showLumi)


if __name__ == "__main__":
    Run()
