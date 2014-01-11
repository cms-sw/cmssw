import sys
import math
from ROOT import *
from cuts import *


def draw_occ(target_dir, c_title, ext, t, title, h_name, h_bins, to_draw, cut, opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas("c","c",600,600)
  c.Clear()
  t.Draw(to_draw + ">>" + h_name + h_bins, cut)
  h = TH2F(gDirectory.Get(h_name))
  if not h:
    sys.exit('h does not exist')
  h = TH2F(h.Clone(h_name))
  h.SetTitle(title)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  c.SaveAs(target_dir + c_title + ext)

def draw_1D(target_dir, c_title, ext, t, title, h_name, h_bins, to_draw, cut, opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas("c","c",600,600)
  c.Clear()
  t.Draw(to_draw + ">>" + h_name + h_bins, cut) 
  h = TH1F(gDirectory.Get(h_name).Clone(h_name))
  if not h:
    sys.exit('h does not exist')
  h.SetTitle(title)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  h.SetMinimum(0.)
  c.SaveAs(target_dir + c_title + ext)

def fill_hist(hist, array):
  [hist.Fill(_) for _ in array] 

def draw_1D_adv(target_dir, c_title, ext, t, title, h_name, h_bins, to_draw, cut, opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas("c","c",600,600)
  c.Clear()
  t.Draw(to_draw + ">>" + h_name + h_bins, cut)
  h = TH1F(gDirectory.Get(h_name).Clone(h_name))
  h.Reset()
  if not h:
    sys.exit('h does not exist')
  cutString = cut.GetTitle()
  t1 = t.CopyTree(Form(cutString))
  for entry in t1:
    vector = range(entry.firstClusterStrip, entry.firstClusterStrip + entry.clusterSize)
    fill_hist(h, vector)
  h.SetTitle(title)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  h.SetMinimum(0.)
  c.SaveAs(target_dir + c_title + ext)

def fill_hist2(hist, array, arg):
  [hist.Fill(_, arg) for _ in array] 

def draw_2D_adv(target_dir, c_title, ext, t, title, h_name, h_bins, to_draw, cut, opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas("c","c",600,600)
  c.Clear()
  t.Draw(to_draw + ">>" + h_name + h_bins, cut)
  h = TH2F(gDirectory.Get(h_name).Clone(h_name))
  h.Reset()
  if not h:
    sys.exit('h does not exist')
  cutString = cut.GetTitle()
  t1 = t.CopyTree(Form(cutString))
  for entry in t1:
    vector = range(entry.firstClusterStrip, entry.firstClusterStrip + entry.clusterSize)
    fill_hist2(h, vector, t.roll)
  h.SetTitle(title)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  h.SetMinimum(0.)
  c.SaveAs(target_dir + c_title + ext)
  
def draw_geff(target_dir, c_title, ext, t, title, h_name, h_bins, to_draw,
              denom_cut, extra_num_cut, opt, color, marker_st = 1, marker_sz = 1):
  c = TCanvas("c","c",600,600)
  c.Clear()
  gPad.SetGrid(1)
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(0)
  gStyle.SetOptFit(0)
  t.Draw(to_draw + ">>num_" + h_name + h_bins,extra_num_cut + denom_cut, "goff")
  num = TH1F(gDirectory.Get("num_" + h_name))
  if not num:
    sys.exit('num does not exist')
  num = TH1F(num.Clone("eff_" + h_name))
  
  t.Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff")
  den = TH1F(gDirectory.Get("denom_" + h_name).Clone("denom_" + h_name))
  eff = TGraphAsymmErrors(num, den)
  if not "same" in opt:
    num.Reset()
    num.GetYaxis().SetRangeUser(0.,1.05)
    num.SetStats(0)
    num.SetTitle(title)
    num.Draw()
    
  eff.SetLineWidth(2)
  eff.SetLineColor(color)
  eff.SetMarkerStyle(marker_st)
  eff.SetMarkerColor(color)
  eff.SetMarkerSize(marker_sz)
  eff.Draw(opt + " same")
  
  ## Do fit in the flat region
  if "eta" in c_title:
    xmin = eta_min
    xmax = eta_max
  else:
    xmin = -999.
    xmax = 999.
    
  f1 = TF1("fit1","pol0", xmin, xmax)
  r = eff.Fit("fit1","RQS")
  ptstats = TPaveStats(0.25,0.35,0.75,0.55,"brNDC")
  ptstats.SetName("stats")
  ptstats.SetBorderSize(0)
  ptstats.SetLineWidth(0)
  ptstats.SetFillColor(0)
  ptstats.SetTextAlign(11)
  ptstats.SetTextFont(42)
  ptstats.SetTextSize(.05)
  ptstats.SetTextColor(kRed)
  ptstats.SetOptStat(0)
  ptstats.SetOptFit(1111)
  chi2 = int(r.Chi2())
  ndf = int(r.Ndf())
    ##   prob = r.Prob()
  round(2.675, 2)
  p0 = f1.GetParameter(0)
  p0e = f1.GetParError(0)
  ptstats.AddText("#chi^{2} / ndf: %d/%d" %(chi2,ndf))
  ##   ptstats.AddText("Fit probability: %f %" %(prob))
  ptstats.AddText("Efficiency: %f #pm %f %%"%(p0,p0e))
  ptstats.Draw("same")
  pt = TPaveText(0.09899329,0.9178322,0.8993289,0.9737762,"blNDC")
  pt.SetName("title")
  pt.SetBorderSize(1)
  pt.SetFillColor(0)
  pt.SetFillStyle(0)
  pt.SetTextFont(42)
  pt.AddText(eff.GetTitle())
  pt.Draw("same")
  c.Modified()
  c.SaveAs(target_dir + c_title + ext) 

  
def draw_geff2(target_dir, c_title, ext, t1, t2, title, h_name, h_bins, to_draw,
              denom_cut, extra_num_cut, opt, color, marker_st = 1, marker_sz = 1):
  c = TCanvas("c","c",600,600)
  c.Clear()
  gPad.SetGrid(1)
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(0)
  gStyle.SetOptFit(0)
  t1.Draw(to_draw + ">>num_" + h_name + h_bins,extra_num_cut + denom_cut, "goff")
  num = TH1F(gDirectory.Get("num_" + h_name))
  if not num:
    sys.exit('num does not exist')
  num = TH1F(num.Clone("eff_" + h_name))
  
  t2.Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff")
  den = TH1F(gDirectory.Get("denom_" + h_name).Clone("denom_" + h_name))
  eff = TGraphAsymmErrors(num, den)
  if not "same" in opt:
    num.Reset()
    num.GetYaxis().SetRangeUser(0.,1.05)
    num.SetStats(0)
    num.SetTitle(title)
    num.Draw()
    
  eff.SetLineWidth(2)
  eff.SetLineColor(color)
  eff.SetMarkerStyle(marker_st)
  eff.SetMarkerColor(color)
  eff.SetMarkerSize(marker_sz)
  eff.Draw(opt + " same")

  ## Do fit in the flat region
  if "eta" in c_title:
    xmin = eta_min
    xmax = eta_max
  else:
    xmin = -999.
    xmax = 999.
    
  f1 = TF1("fit1","pol0", xmin, xmax)
  r = eff.Fit("fit1","RQS")
  ptstats = TPaveStats(0.25,0.35,0.75,0.55,"brNDC")
  ptstats.SetName("stats")
  ptstats.SetBorderSize(0)
  ptstats.SetLineWidth(0)
  ptstats.SetFillColor(0)
  ptstats.SetTextAlign(11)
  ptstats.SetTextFont(42)
  ptstats.SetTextSize(.05)
  ptstats.SetTextColor(kRed)
  ptstats.SetOptStat(0)
  ptstats.SetOptFit(1111)
  chi2 = int(r.Chi2())
  ndf = int(r.Ndf())
    ##   prob = r.Prob()
  round(2.675, 2)
  p0 = f1.GetParameter(0)
  p0e = f1.GetParError(0)
  ptstats.AddText("#chi^{2} / ndf: %d/%d" %(chi2,ndf))
  ##   ptstats.AddText("Fit probability: %f %" %(prob))
  ptstats.AddText("Efficiency: %f #pm %f %%"%(p0,p0e))
  ptstats.Draw("same")
  pt = TPaveText(0.09899329,0.9178322,0.8993289,0.9737762,"blNDC")
  pt.SetName("title")
  pt.SetBorderSize(1)
  pt.SetFillColor(0)
  pt.SetFillStyle(0)
  pt.SetTextFont(42)
  pt.AddText(eff.GetTitle())
  pt.Draw("same")
  c.Modified()
  c.SaveAs(target_dir + c_title + ext) 
    
def draw_bx(target_dir, c_title, ext, t, title, h_name, h_bins, to_draw, cut, opt = ""):
  gStyle.SetStatStyle(0)
  gStyle.SetOptStat(1110)
  c = TCanvas("c","c",600,600)
  t.Draw(to_draw + ">>" + h_name + h_bins, cut) 
  gPad.SetLogy()
  h = TH1F(gDirectory.Get(h_name).Clone(h_name))
  h.SetTitle(title)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw(opt)
  h.SetMinimum(1.)
  c.SaveAs(target_dir + c_title + ext)
