#!/bin/env python

import sys, os

from array import array
import optparse
#import argparse
from collections import OrderedDict
import ROOT

import CondCore.Utilities.conddblib as conddb

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

    parser.add_option("--usePixelQuality",action="store_true", dest="usePixelQuality", default=False,help="whether use SiPixelQuality")
    parser.add_option("--showLumi",action="store_true", dest="showLumi", default=False,help="whether use integrated lumi as x-axis")

    parser.add_option("--years", dest="years", default = [2016,2017,2018], action="callback", callback=vararg_callback, help="years to plot")
    parser.add_option("--substructures", dest="substructures", default = ['BPIX','PIX'], action="callback", callback=vararg_callback, help="substrcutures to plot")

    parser.add_option("-l",action="callback",callback=callback_rootargs)
    parser.add_option("-q",action="callback",callback=callback_rootargs)
    parser.add_option("-b",action="callback",callback=callback_rootargs)

    return parser


# 1/lumiScaleFactor to go from 1/pb to 1/fb
lumiScaleFactor = 1000

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
    runs = accumulatedLumiPerRun.keys()
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
        if(isEOY and iov.run>=320413):
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
            pos_ = {"x":10000*getattr(iov, branch_name).X(), 
                    "y":10000*getattr(iov, branch_name).Y(),
                    "z":10000*getattr(iov, branch_name).Z()}

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


def plotbarycenter(bc, coord,substructure, runsPerYear, pixelLocalRecos, accumulatedLumiPerRun, withPixelQuality, showLumi) :

    substructureTitle=""
    if(substructure=="PIX") :
       substructureTitle="Pixel"
    if(substructure=="BPIX") :
       substructureTitle="Pixel barrel"

    runs = accumulatedLumiPerRun.keys()
    runs.sort()
    years = runsPerYear.keys()
    years.sort()

    can = ROOT.TCanvas("barycentre_"+substructure+"_"+coord, "", 2000, 900)
    can.cd()

    bc_rereco = bc["rereco"]
    bc_prompt = bc["prompt"]
    bc_EOY = bc["EOY"]

    gr_rereco = bc_rereco[coord+"_"+substructure]
    gr_prompt = bc_prompt[coord+"_"+substructure]
    gr_EOY = bc_EOY[coord+"_"+substructure]

    gr_rereco.SetMarkerStyle(8)
    gr_rereco.SetMarkerSize(0)
    gr_EOY.SetMarkerStyle(8)
    gr_EOY.SetMarkerSize(0)
    gr_prompt.SetMarkerStyle(8)
    gr_prompt.SetMarkerSize(0)
    gr_rereco.SetLineColor(ROOT.kGreen+3)
    gr_EOY.SetLineColor(ROOT.kRed)
    gr_prompt.SetLineColor(ROOT.kBlue)

    upper = max(bc_rereco[coord+"max_"+substructure], bc_EOY[coord+"max_"+substructure], bc_prompt[coord+"max_"+substructure])
    lower = min(bc_rereco[coord+"min_"+substructure], bc_EOY[coord+"min_"+substructure], bc_prompt[coord+"min_"+substructure])

    scale = 1.1

    if(upper>0) :
      upper = upper * scale
    else :
      upper = upper / scale

    lower = min(bc_rereco[coord+"min_"+substructure], bc_EOY[coord+"min_"+substructure], bc_prompt[coord+"min_"+substructure])
    if(lower>0) :
      lower = lower / scale
    else :
      lower = lower * scale

    range_ = upper - lower
    width_ = gr_prompt.GetXaxis().GetXmax() - gr_prompt.GetXaxis().GetXmin()

    gr_prompt.GetYaxis().SetRangeUser(lower, upper)
    gr_prompt.GetYaxis().SetTitle(substructureTitle+" barycentre ("+coord+") [#mum]")
    gr_prompt.GetXaxis().SetTitle("Run Number")
    gr_prompt.GetYaxis().CenterTitle(True)
    gr_prompt.GetXaxis().CenterTitle(True)
    gr_prompt.GetYaxis().SetTitleOffset(0.80)
    gr_prompt.GetYaxis().SetTitleSize(0.055)
    gr_prompt.GetXaxis().SetTitleOffset(0.80)
    gr_prompt.GetXaxis().SetTitleSize(0.055)
    gr_prompt.GetXaxis().SetMaxDigits(6)
    gr_prompt.Draw("AP")

    if(showLumi) :
      gr_prompt.GetXaxis().SetTitle("Delivered luminosity [1/fb]")

    gr_EOY.Draw("P")
    gr_rereco.Draw("P")         

    gr_dummyFirstRunOfTheYear = blackBox(-999, 10000, -999, -10000)
    gr_dummyFirstRunOfTheYear.SetLineColor(ROOT.kBlack)
    gr_dummyFirstRunOfTheYear.SetLineStyle(1)
    gr_dummyFirstRunOfTheYear.Draw("L")

    gr_dummyPixelReco = blackBox(-999, 10000, -999, -10000)
    gr_dummyPixelReco.SetLineColor(ROOT.kGray+1)
    gr_dummyPixelReco.SetLineStyle(3)
    gr_dummyPixelReco.Draw("L")

    gr_prompt.SetTitle("Alignment during data taking" )
    gr_EOY.SetTitle("End-of-Year Re-reconstruction" )
    gr_rereco.SetTitle("Legacy reprocessing" )
    gr_dummyFirstRunOfTheYear.SetTitle("First run of the year")
    gr_dummyPixelReco.SetTitle("Pixel calibration update")

    #legend = can.BuildLegend()#0.65, 0.65, 0.85, 0.85)
    #legend.SetShadowColor(0)
    #legend.SetFillColor(0)
    #legend.SetLineColor(0)
   
    gr_EOY.SetTitle("")
    gr_rereco.SetTitle("")
    gr_prompt.SetTitle("")
    gr_dummyFirstRunOfTheYear.SetTitle("")
    gr_dummyPixelReco.SetTitle("")

    # Add legends
    # and vertical lines
    years_label = ""
    for year in years :
        years_label += str(year)
        years_label += "+"
    years_label = years_label.rstrip("+")

    # CMS logo
    CMSworkInProgress = ROOT.TPaveText( gr_rereco.GetXaxis().GetXmax()-0.3*width_, upper+range_*0.005,
                                        gr_rereco.GetXaxis().GetXmax(), upper+range_*0.055, "nb")
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

    # plot diff EOY - prompt and Rereco - prompt
    canDiff = ROOT.TCanvas("barycentreDiff_"+substructure+"_"+coord, "", 2000, 900)
    canDiff.cd()

    xmin = gr_rereco.GetXaxis().GetXmin()
    xmax = gr_rereco.GetXaxis().GetXmax()

    # get diff between prompt and rereco
    a_diff_rereco = array('d',[])
    a_diff_EOY = array('d',[])
    a_rereco = bc_rereco['a_'+coord+'_'+substructure]
    a_EOY = bc_EOY['a_'+coord+'_'+substructure]
    a_prompt = bc_prompt['a_'+coord+'_'+substructure]

    for i in range(len(a_rereco)):
        a_diff_rereco.append(a_rereco[i]-a_prompt[i])
    for i in range(len(a_EOY)):
        a_diff_EOY.append(a_EOY[i]-a_prompt[i])

    v_diff_rereco = ROOT.TVectorD(len(a_diff_rereco),a_diff_rereco)
    v_diff_EOY    = ROOT.TVectorD(len(a_diff_EOY),a_diff_EOY)

    gr_diff_rereco = ROOT.TGraphErrors(bc_rereco["v_runlumiplot"], v_diff_rereco,
                                       bc_rereco["v_runlumierror"],bc_rereco["v_zeros"])
    gr_diff_EOY   = ROOT.TGraphErrors(bc_EOY["v_runlumiplot"], v_diff_EOY,
                                      bc_EOY["v_runlumierror"],bc_EOY["v_zeros"])

    diffmax_rereco = max(a_diff_rereco)
    diffmax_EOY = max(a_diff_EOY)

    diffmax = max( max(a_diff_rereco), max(a_diff_EOY))
    diffmin = min( min(a_diff_rereco), min(a_diff_EOY))

    diffmax = diffmax + 50
    diffmin = diffmin - 50

    upper = diffmax
    range_ = diffmax - diffmin

    gr_diff_rereco.SetMarkerStyle(8)
    gr_diff_rereco.SetMarkerSize(0)
    gr_diff_rereco.SetLineColor(ROOT.kGreen+3)
    gr_diff_EOY.SetMarkerStyle(8)
    gr_diff_EOY.SetMarkerSize(0)
    gr_diff_EOY.SetLineColor(ROOT.kRed)

    gr_diff_EOY.GetYaxis().SetRangeUser(diffmin, diffmax)
    gr_diff_EOY.GetYaxis().SetTitle(substructureTitle+" barycentre ("+coord+") diff w.r.t. alignment during data taking[#mum]")
    gr_diff_EOY.GetXaxis().SetTitle("Run Number")
    gr_diff_EOY.GetYaxis().CenterTitle(True)
    gr_diff_EOY.GetXaxis().CenterTitle(True)
    gr_diff_EOY.GetYaxis().SetTitleOffset(0.80)
    gr_diff_EOY.GetYaxis().SetTitleSize(0.055)
    gr_diff_EOY.GetXaxis().SetTitleOffset(0.80)
    gr_diff_EOY.GetXaxis().SetTitleSize(0.055)
    gr_diff_EOY.GetXaxis().SetMaxDigits(6)
    gr_diff_EOY.Draw("AP")

    if(showLumi) :
      gr_diff_EOY.GetXaxis().SetTitle("Delivered luminosity [1/fb]")
    gr_diff_rereco.Draw("P")

    gr_dummyFirstRunOfTheYear.Draw("L")
    gr_dummyPixelReco.Draw("L")

    gr_diff_EOY.SetTitle("End-of-Year Re-reconstruction" )
    gr_diff_rereco.SetTitle("Legacy reprocessing" )
    gr_dummyFirstRunOfTheYear.SetTitle("First run of the year")
    gr_dummyPixelReco.SetTitle("Pixel calinbration update")

    #legendDiff = canDiff.BuildLegend()#0.65, 0.65, 0.85, 0.85)
    #legendDiff.SetShadowColor(0)
    #legendDiff.SetFillColor(0)
    #legendDiff.SetLineColor(0)

    gr_diff_EOY.SetTitle("")
    gr_diff_rereco.SetTitle("")
    gr_dummyFirstRunOfTheYear.SetTitle("")
    gr_dummyPixelReco.SetTitle("")

    #draw horizontal line at zero
    line_diff = ROOT.TLine(xmin,0,xmax,0)
    line_diff.SetLineColor(ROOT.kBlue)
    line_diff.Draw()

    # CMS logo
    CMSworkInProgress.Draw()

    # vertical lines
    #pixel local reco
    line_diff_pixels = {}
    for since in pixelLocalRecos :
        if showLumi :
           run_index = findRunIndex(since,runs)
           integrated_lumi = accumulatedLumiPerRun[runs[run_index]]
           line_diff_pixels[since] = ROOT.TLine(integrated_lumi, diffmin, integrated_lumi, diffmax)

        else :
           line_diff_pixels[since] = ROOT.TLine(since, diffmin, since, diffmax)

        line_diff_pixels[since].SetLineColor(ROOT.kGray+1)
        line_diff_pixels[since].SetLineStyle(3)
        line_diff_pixels[since].Draw()

    # years
    line_diff_years = {}
    box_diff_years = {}
    text_diff_years = {}
    if(len(years)>1 or (not showLumi) ) : # indicate begining of the year if more than one year to show or use run number
      for year in years :
          if showLumi :
             #first run of the year
             run_index = findRunIndex(runsPerYear[year][0],runs)
             integrated_lumi = accumulatedLumiPerRun[runs[run_index]]
             line_diff_years[year] = ROOT.TLine(integrated_lumi, diffmin, integrated_lumi, diffmax)
             text_diff_years[year] = ROOT.TPaveText( integrated_lumi+0.01*width_, upper-range_*0.05,
                                              integrated_lumi+0.05*width_,  upper-range_*0.015, "nb")
             box_diff_years[year] = blackBox(integrated_lumi+0.005*width_, upper-range_*0.01, integrated_lumi+0.055*width_, upper-range_*0.055)
          else :
             line_diff_years[year] = ROOT.TLine(runsPerYear[year][0], diffmin, runsPerYear[year][0], diffmax)
             text_diff_years[year] = ROOT.TPaveText( runsPerYear[year][0]+0.01*width_, upper-range_*0.05,
                                              runsPerYear[year][0]+0.05*width_,  upper-range_*0.015, "nb")
             box_diff_years[year] = blackBox(runsPerYear[year][0]+0.01*width_, upper-range_*0.015, runsPerYear[year][0]+0.05*width_, upper-range_*0.05)

          box_diff_years[year].Draw("L")
          line_diff_years[year].Draw("same")

          # Add TextBox at the beginning of each year
          text_diff_years[year].AddText(str(year))
          text_diff_years[year].SetTextAlign(32)
          text_diff_years[year].SetTextSize(0.025)
          text_diff_years[year].SetFillColor(10)
          text_diff_years[year].Draw()

    #legendDiff.Draw()
    canDiff.Update()

    if(showLumi) :
       canDiff.SaveAs("baryCentreDiff_"+coord+"_"+substructure+"_"+years_label+"_IntegratedLumi.pdf")
       canDiff.SaveAs("baryCentreDiff_"+coord+"_"+substructure+"_"+years_label+"_IntegratedLumi.png")
    else :
       canDiff.SaveAs("baryCentreDiff_"+coord+"_"+substructure+"_"+years_label+"_RunNumber.pdf")
       canDiff.SaveAs("baryCentreDiff_"+coord+"_"+substructure+"_"+years_label+"_RunNumber.png")

# main call
def Run():

    parser=parseOptions()
    (options,args) = parser.parse_args()
    sys.argv = grootargs

    print(options)

    usePixelQuality=options.usePixelQuality
    showLumi=options.showLumi

    # order years from old to new
    years=options.years
    years.sort()
    print("Years:")
    print(years)

    # Global variables
    # runs per year
    runsPerYear = {}
    for year in years :
        runsPerYear[year] = []

    # integrated lumi vs run
    accumulatedLumiPerRun = {}

    run_index = 0
    lastRun = 1
    CMSSW_Dir = os.getenv("CMSSW_BASE")
    for year in years :

        lumiFile = open(CMSSW_Dir + "/src/Alignment/OfflineValidation/data/lumiperrun"+str(year)+".txt",'r')
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

    # order by key :
    # order by key (year)
    runsPerYear = OrderedDict(sorted(runsPerYear.items(), key=lambda t: t[0]))
    # order by key (run number)
    accumulatedLumiPerRun = OrderedDict(sorted(accumulatedLumiPerRun.items(), key=lambda t: t[0]))

    #pixel local reco update (IOVs/sinces)
    pixelLocalRecos = []
    # connnect to ProdDB to access pixel local reco condition change
    #db = "frontier://FrontierProd/CMS_CONDITIONS"
    pixel_template = "SiPixelTemplateDBObject_38T_v16_offline"
    con = conddb.connect(url = conddb.make_url("pro"))
    session = con.session()
    # get IOV table
    IOV = session.get_dbtype(conddb.IOV)
    iovs = set(session.query(IOV.since).filter(IOV.tag_name == pixel_template).all())
    session.close()

    pixelLocalRecos = sorted([int(item[0]) for item in iovs])
    print("Pixel template updates:")
    print(pixelLocalRecos)

    f = ROOT.TFile("PixelBaryCentre_2016_2017_2018.root","READ")

    withPixelQuality="" 
    if(usePixelQuality) :
      withPixelQuality="WithPixelQuality"

    t_rereco = f.Get("PixelBaryCentreAnalyzer"+withPixelQuality+"/PixelBarycentre")
    t_prompt = f.Get("PixelBaryCentreAnalyzer"+withPixelQuality+"/PixelBarycentre_prompt")
    t_EOY = f.Get("PixelBaryCentreAnalyzer"+withPixelQuality+"/PixelBarycentre_EOY")

    # substructures to plot
    substructures=options.substructures

    bc = {}
    isEOY = False
    bc["rereco"] = readBaryCentreAnalyzerTree(t_rereco, substructures, accumulatedLumiPerRun, showLumi, isEOY)
    bc["prompt"] = readBaryCentreAnalyzerTree(t_prompt, substructures, accumulatedLumiPerRun, showLumi, isEOY)
    isEOY = True
    bc["EOY"] = readBaryCentreAnalyzerTree(t_EOY, substructures, accumulatedLumiPerRun, showLumi, isEOY)

    for substructure in substructures :

        for coord in ['x','y','z'] :

            plotbarycenter(bc, coord,substructure, runsPerYear,pixelLocalRecos,accumulatedLumiPerRun,withPixelQuality,showLumi)


if __name__ == "__main__":
    Run()
