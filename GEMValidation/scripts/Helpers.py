from ROOT import *
from cuts import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def drawCscLabel(title, x=0.17, y=0.35, font_size=0.):
    tex = TLatex(x, y,title)
    if font_size > 0.:
      tex.SetTextSize(font_size)
      tex.SetTextSize(0.05)
      tex.SetNDC()
      tex.Draw()
      return tex


#_______________________________________________________________________________
def drawEtaLabel(minEta, maxEta, x=0.17, y=0.35, font_size=0.):
    tex = TLatex(x, y,"%.2f < |#eta| < %.2f"%(minEta,maxEta))
    if font_size > 0.:
      tex.SetTextSize(font_size)
      tex.SetTextSize(0.05)
      tex.SetNDC()
      tex.Draw()
      return tex


#_______________________________________________________________________________
def drawPuLabel(pu, x=0.17, y=0.35, font_size=0.):
    tex = TLatex(x, y,"<PU> = %d"%(pu))
    if font_size > 0.:
      tex.SetTextSize(font_size)
      tex.SetTextSize(0.05)
      tex.SetNDC()
      tex.Draw()
      return tex
  

#_______________________________________________________________________________
def draw_eff(t,title, h_name, h_bins, to_draw, denom_cut, extra_num_cut, 
             color = kBlue, marker_st = 20):
    """Make an efficiency plot"""
    
    ## total numerator selection cut
    num_cut = AND(denom_cut,extra_num_cut)

    t.Draw(to_draw + ">>num_" + h_name + h_bins, num_cut, "goff")
    num = TH1F(gDirectory.Get("num_" + h_name).Clone("num_" + h_name))
    t.Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff")
    den = TH1F(gDirectory.Get("denom_" + h_name).Clone("denom_" + h_name))

    useTEfficiency = True
    if useTEfficiency:
        eff = TEfficiency(num, den)
    else:
        eff = TGraphAsymmErrors(num, den)

    eff.SetTitle(title)
    eff.SetLineWidth(2)
    eff.SetLineColor(color)
    eff.SetMarkerStyle(marker_st)
    eff.SetMarkerColor(color)
    eff.SetMarkerSize(.5)
    return eff


#_______________________________________________________________________________
def draw_geff(t, title, h_bins, to_draw, den_cut, extra_num_cut, 
              opt = "", color = kBlue, marker_st = 1, marker_sz = 1.):
    """Make an efficiency plot"""
    
    ## total numerator selection cut 
    ## the extra brackets around the extra_num_cut are necessary !!
    num_cut = AND(den_cut,extra_num_cut)
    debug = False
    if debug:
        print "Denominator cut", den_cut
        print "Numerator cut", num_cut
 
    ## PyROOT works a little different than ROOT when you are plotting 
    ## histograms directly from tree. Hence, this work-around
    nBins  = int(h_bins[1:-1].split(',')[0])
    minBin = float(h_bins[1:-1].split(',')[1])
    maxBin = float(h_bins[1:-1].split(',')[2])
    
    num = TH1F("num", "", nBins, minBin, maxBin) 
    den = TH1F("den", "", nBins, minBin, maxBin)

    t.Draw(to_draw + ">>num", num_cut, "goff")
    t.Draw(to_draw + ">>den", den_cut, "goff")

    ## check if the number of passed entries larger than total entries
    doConsistencyCheck = False
    if doConsistencyCheck:
        for i in range(0,nBins):
            print i, num.GetBinContent(i), den.GetBinContent(i)
            if num.GetBinContent(i) > den.GetBinContent(i):
                print ">>>Error: passed entries > total entries" 

    eff = TEfficiency(num, den)

    ## plotting options
    if not "same" in opt:
        num.Reset()
        num.GetYaxis().SetRangeUser(0.0,1.1)
        num.SetStats(0)
        num.SetTitle(title)
        num.Draw()
        
    eff.SetLineWidth(2)
    eff.SetLineColor(color)
    eff.Draw(opt + " same")
    eff.SetMarkerStyle(marker_st)
    eff.SetMarkerColor(color)
    eff.SetMarkerSize(marker_sz)

    SetOwnership(eff, False)
    return eff
