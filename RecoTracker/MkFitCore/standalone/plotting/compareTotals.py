import os,sys
import ROOT
import copy

doNorm = True

def getCanvasMainPad( logY ):
  pad1 = ROOT.TPad("pad1", "pad1", 0, 0.2, 1, 1)
  pad1.SetBottomMargin(0.15)
  if( logY ):
    pad1.SetLogy()
  return pad1

def getCanvasRatioPad( logY ):
  pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.21)
  pad2.SetTopMargin(0.05)
  pad2.SetBottomMargin(0.1)
  return pad2

def getRatioAxes( xMin, xMax, yMin, yMax ):
  h2_axes_ratio = ROOT.TH2D("axes_ratio", "", 10, xMin, xMax, 10, yMin, yMax )
  h2_axes_ratio.SetStats(0)
  h2_axes_ratio.GetXaxis().SetLabelSize(0.00)
  h2_axes_ratio.GetXaxis().SetTickLength(0.09)
  h2_axes_ratio.GetYaxis().SetNdivisions(5,5,0)
  h2_axes_ratio.GetYaxis().SetTitleSize(0.13)
  h2_axes_ratio.GetYaxis().SetTitleOffset(0.37)
  h2_axes_ratio.GetYaxis().SetLabelSize(0.13)
  h2_axes_ratio.GetYaxis().SetTitle("Ratio")
  return h2_axes_ratio


dirnames = []
dirnames.append(sys.argv[1]) #original (1)
dirnames.append(sys.argv[2]) #updated (2)

outdir = sys.argv[3]
if not os.path.exists(outdir):
  os.makedirs(outdir)
  os.system("cp web/index.php %s"%(outdir))

names = ["Reference", "Other"]
if len(sys.argv)==5 and sys.argv[4]=="vsCMSSW":
  names = ["CMSSW", "mkFit"]

if len(sys.argv)==6:
  names = [sys.argv[4], sys.argv[5]]

colors = [1,2]

fnames = []
fs = []
for d in range(0, len(dirnames)):
  fnames.append(dirnames[d]+"/plots.root")
  fs.append(ROOT.TFile.Open(fnames[d]))

subdirs=[]

eff_obj   = []
eff_pass  = []
eff_tot   = []
eff_ratio = []
#
hist      = []
#
hist_xratio = []
eff_xratio  = []

for d in range(0, len(fnames)):
  
  fs[d].cd()

  for dkey in ROOT.gDirectory.GetListOfKeys():
    if not dkey.IsFolder():
      continue
    if d<=0:
      subdirs.append(dkey.GetName())

  eff_obj_s   = []
  eff_pass_s  = []
  eff_tot_s   = []
  eff_ratio_s = []
  #
  hist_s      = []
  
  subhist = []
  subrate = []
  
  for subdir in subdirs:
    print "In subdir %s:"%subdir
    fs[d].cd(subdir)

    thiseff_obj  = []
    thiseff_pass = []
    thiseff_tot  = []
    thiseff      = []      
    #
    thishist = []
    #
    nh=0
    ne=0
    #
    for key in ROOT.gDirectory.GetListOfKeys():
      obj = key.ReadObj()
      if obj.IsA().InheritsFrom("TH1"):
        h = obj
        #print "Found TH1 %s"%h.GetName()
          
        thishist.append(h)
        thishist[nh].Sumw2()
        thishist[nh].SetLineColor(colors[d])
        thishist[nh].SetMarkerColor(colors[d])
        thishist[nh].SetMarkerSize(0.3)
        thishist[nh].SetMarkerStyle(20)
        thishist[nh].SetStats(0)
        
        nh=nh+1

      if obj.IsA().InheritsFrom("TEfficiency"):
        e = obj
        #print "Found TEfficiency %s"%e.GetName()
        
        thiseff_obj .append(e)
        thiseff_obj[ne].SetLineColor(colors[d])
        thiseff_obj[ne].SetMarkerColor(colors[d])
        thiseff_obj[ne].SetMarkerSize(0.3)
        thiseff_obj[ne].SetMarkerStyle(20)
        thiseff_pass.append(e.GetPassedHistogram())
        thiseff_tot .append(e.GetTotalHistogram())
        thiseff_pass[ne].Sumw2()
        thiseff_tot[ne] .Sumw2()
        effname = "%s_rate"%(thiseff_pass[ne].GetName())
        auxeff = thiseff_pass[ne].Clone(effname)
        auxeff.Divide(thiseff_pass[ne],thiseff_tot[ne],1.0,1.0,"B")
        thiseff.append(auxeff)
        thiseff[ne].SetLineColor(colors[d])
        thiseff[ne].SetMarkerColor(colors[d])
        thiseff[ne].SetMarkerSize(0.3)
        thiseff[ne].SetMarkerStyle(20)
        thiseff[ne].SetStats(0)
        
        ne=ne+1

    hist_s     .append(thishist)
    #
    eff_ratio_s.append(thiseff)
    eff_pass_s .append(thiseff_pass)
    eff_tot_s  .append(thiseff_tot)
    eff_obj_s  .append(thiseff_obj)
        
  hist       .append(hist_s)
  hist_xratio.append(hist_s)
  #
  eff_ratio .append(eff_ratio_s)
  eff_xratio.append(eff_ratio_s)
  eff_pass  .append(eff_pass_s)
  eff_tot   .append(eff_tot_s)
  eff_obj   .append(eff_obj_s)
  
ratios_hist = []
ratios_eff  = []
for dd in range(len(subdirs)):
  
  thisratio = []
  for r in range(len(hist_xratio[0][dd])):
    auxratio = hist_xratio[1][dd][r].Clone("num")
    auxden   = hist_xratio[0][dd][r].Clone("den")
    intnum   = auxratio.Integral(0,-1)
    intden   = auxden  .Integral(0,-1)
    if intnum>0:
      auxratio.Scale(1.0/intnum)
    if intden>0:
      auxden.Scale(1.0/intden)
      auxratio.Divide(auxden)
    auxratio.SetName("ratio")
    thisratio.append(auxratio)
    thisratio[r].GetYaxis().SetTitle("Ratio")
    thisratio[r].SetLineColor(colors[1])
    thisratio[r].SetMarkerColor(colors[1])
    thisratio[r].SetMarkerSize(0)
    thisratio[r].SetStats(0)
  ratios_hist.append(thisratio)
  
  thisratio = []
  for r in range(len(eff_xratio[0][dd])):
    auxratio = eff_xratio[1][dd][r].Clone(  "numerator")
    auxden   = eff_xratio[0][dd][r].Clone("denominator")
    auxratio.Divide(auxden)
    auxratio.SetName("ratio")
    thisratio.append(auxratio)
    thisratio[r].GetYaxis().SetTitle("Ratio")
    thisratio[r].SetLineColor(colors[1])
    thisratio[r].SetMarkerColor(colors[1])
    thisratio[r].SetMarkerSize(0)
    thisratio[r].SetStats(0)
  ratios_eff.append(thisratio)

### Drawing
ROOT.gStyle.SetOptStat(0)

outsubdir = []
for ns,subdir in enumerate(subdirs):
  thisdir = "%s/%s"%(outdir,subdir)
  outsubdir.append(thisdir)
  if not os.path.exists(thisdir):
    os.mkdir(thisdir)
    os.system("cp web/index.php %s"%(thisdir))

  for r in range(len(eff_xratio[0][ns])): 
      
    outname = eff_obj[0][ns][r].GetName()
    
    can = ROOT.TCanvas("can_%s"%outname, "", 600, 600)
    can.cd()
    
    pad1 = getCanvasMainPad(0)
    pad1.SetTickx()
    pad1.SetTicky()
    
    pad2 = getCanvasRatioPad(0)
    pad2.SetTickx()
    pad2.SetTicky()
    
    can.cd()
    pad1.Draw()
    pad1.cd()
    
    ttitle = eff_obj[0][ns][r].GetTitle()
    xmin = ratios_eff[ns][r].GetXaxis().GetBinLowEdge(1)
    xmax = ratios_eff[ns][r].GetXaxis().GetBinUpEdge(ratios_eff[ns][r].GetNbinsX())
    yminM = 0.0
    ymaxM = 1.2
    #if "dr" in outname or "ineff" in outname:
    #  ymaxM = 0.50
    xtitle = ratios_eff[ns][r].GetXaxis().GetTitle()
    ytitle = "Efficiency"
    if "dr" in outname:
      ytitle = "Duplicate rate"
    elif "fr" in outname:
      ytitle = "Fake rate"
    elif "ineff" in outname:
      ytitle = "Inefficiency"
      
    haxisMain  = ROOT.TH2D("haxisMain" ,ttitle,1,xmin ,xmax,1,yminM,ymaxM)
    
    haxisMain.GetXaxis().SetTitle(xtitle)
    haxisMain.GetXaxis().SetTitleOffset(1.2)
    haxisMain.GetYaxis().SetTitle(ytitle)
    haxisMain.GetYaxis().SetTitleOffset(1.4)
    
    haxisMain.Draw()
    eff_obj[0][ns][r].Draw("PE,same")
    eff_obj[1][ns][r].Draw("PE,same")
    
    legend = ROOT.TLegend(0.7,0.7, 0.87, 0.87);
    legend.SetLineColor(0)
    legend.SetFillColor(0)
    legend.AddEntry(eff_obj[0][ns][r], names[0], "PL")
    legend.AddEntry(eff_obj[1][ns][r], names[1], "PL")
    legend.Draw("same")
    
    can.cd()
    pad2.Draw()
    pad2.cd()
    
    ymin = 0.9*ratios_eff[ns][r].GetMinimum()
    ymax = 1.1*ratios_eff[ns][r].GetMaximum()
    
    if ymin==0:
      ymin=0.75
      if ymax<=ymin:
        ymin=0.75*ymax
        ymax=1.25*ymax
        
    if ymax<=ymin:
      ymin=0.0
      ymax=2.0
    
    hraxes = getRatioAxes(xmin,xmax,ymin,ymax)
    
    line = ROOT.TLine(xmin,1.0,xmax,1.0)
    line.SetLineColor(1)
    line.SetLineStyle(2)
    
    hraxes.Draw("")
    ratios_eff[ns][r].Draw("PE,same")
    line.Draw("same")
    
    can.cd()
    pad1.Draw()
    pad2.Draw()
    
    can.SaveAs("%s/%s.png"%(thisdir,outname));
    can.SaveAs("%s/%s.pdf"%(thisdir,outname));
    
    can.Update()
    can.Clear()

    tot        = [eff_tot[0][ns][r].Integral(), eff_tot[1][ns][r].Integral()]
    passing    = [eff_pass[0][ns][r].Integral(), eff_pass[1][ns][r].Integral()]
    efficiency = []
    reldiff    = []
    for d in range(0,len(tot)):
      if tot[d]>0:
        efficiency.append(passing[d]/tot[d])
      else:
        efficiency.append(0.0)
      if efficiency[0]>0:
        reldiff.append(efficiency[d]/efficiency[0])
      else:
        reldiff.append(0.0)

    fo = open("%s/%s.log"%(thisdir,outname),"w+")
    fo.write( "Totals:" )
    for d in range(0,len(tot)):
      fo.write( " %d " % int(tot[d]) ),
    fo.write( "\nPassing:" )
    for d in range(0,len(tot)):
      fo.write( " %d " % int(passing[d]) ),
    fo.write( "\nRate:" )
    for d in range(0,len(tot)):
      fo.write( " %0.4f " % efficiency[d] ),
    fo.write( "\nRatio(/reference):" )
    for d in range(0,len(tot)):
      fo.write( " %0.4f " % reldiff[d] ),
    fo.write( "\n" )


    if "_pt_" in outname:
      outname = outname+"_logx"
      
      can = ROOT.TCanvas("can_%s"%outname, "", 600, 600)
      can.cd()
      
      pad1 = getCanvasMainPad(0)
      pad1.SetTickx()
      pad1.SetTicky()
      pad1.SetLogx()
      
      pad2 = getCanvasRatioPad(0)
      pad2.SetTickx()
      pad2.SetTicky()
      pad2.SetLogx()
      
      can.cd()
      pad1.Draw()
      pad1.cd()
      
      ttitle = eff_obj[0][ns][r].GetTitle()
      xmin = 0.1
      xmax = ratios_eff[ns][r].GetXaxis().GetBinUpEdge(ratios_eff[ns][r].GetNbinsX())
      yminM = 0.0
      ymaxM = 1.2
      #if "dr" in outname or "ineff" in outname:
      #  ymaxM = 0.50
      xtitle = ratios_eff[ns][r].GetXaxis().GetTitle()
      ytitle = "Efficiency"
      if "dr" in outname:
        ytitle = "Duplicate rate"
      elif "fr" in outname:
        ytitle = "Fake rate"
      elif "ineff" in outname:
        ytitle = "Inefficiency"
        
      haxisMain  = ROOT.TH2D("haxisMain" ,ttitle,1,xmin,xmax,1,yminM,ymaxM)
      
      haxisMain.GetXaxis().SetTitle(xtitle)
      haxisMain.GetXaxis().SetTitleOffset(1.2)
      haxisMain.GetYaxis().SetTitle(ytitle)
      haxisMain.GetYaxis().SetTitleOffset(1.4)
      
      haxisMain.Draw()
      eff_obj[0][ns][r].Draw("PE,same")
      eff_obj[1][ns][r].Draw("PE,same")
      
      legend = ROOT.TLegend(0.7, 0.7, 0.87, 0.87);
      legend.SetLineColor(0)
      legend.SetFillColor(0)
      legend.AddEntry(eff_obj[0][ns][r], names[0], "PL")
      legend.AddEntry(eff_obj[1][ns][r], names[1], "PL")
      legend.Draw("same")
      
      can.cd()
      pad2.Draw()
      pad2.cd()
      
      ymin = 0.9*ratios_eff[ns][r].GetMinimum()
      ymax = 1.1*ratios_eff[ns][r].GetMaximum()
      
      if ymin==0:
        ymin=0.75
        if ymax<=ymin:
          ymin=0.75*ymax
          ymax=1.25*ymax
        
      if ymax<=ymin:
        ymin=0.0
        ymax=2.0
        
      hraxes = getRatioAxes(xmin,xmax,ymin,ymax)
      
      line = ROOT.TLine(xmin,1.0,xmax,1.0)
      line.SetLineColor(1)
      line.SetLineStyle(2)
      
      hraxes.Draw("")
      ratios_eff[ns][r].Draw("PE,same")
      line.Draw("same")
      
      can.cd()
      pad1.Draw()
      pad2.Draw()
      
      can.SaveAs("%s/%s.png"%(thisdir,outname));
      can.SaveAs("%s/%s.pdf"%(thisdir,outname));

      can.Update()
      can.Clear()
      
    del haxisMain
    del hraxes
    del pad1
    del pad2
    del can

  ###

  for r in range(len(hist_xratio[0][ns])): 
      
    outname = hist[0][ns][r].GetName()
    
    can = ROOT.TCanvas("can_%s"%outname, "", 600, 600)
    can.cd()
    
    pad1 = getCanvasMainPad(0)
    pad1.SetTickx()
    pad1.SetTicky()
    
    pad2 = getCanvasRatioPad(0)
    pad2.SetTickx()
    pad2.SetTicky()
    
    can.cd()
    pad1.Draw()
    pad1.cd()
    
    int0   = hist[0][ns][r].Integral(0,-1)
    int1   = hist[1][ns][r].Integral(0,-1)
    if int0>0 and doNorm:
      hist[0][ns][r].Scale(1.0/int0)
    if int1>0 and doNorm:
      hist[1][ns][r].Scale(1.0/int1)
    
    means = [hist[0][ns][r].GetMean(),hist[1][ns][r].GetMean()]

    ttitle = hist[0][ns][r].GetTitle()
    xmin = ratios_hist[ns][r].GetXaxis().GetBinLowEdge(1)
    xmax = ratios_hist[ns][r].GetXaxis().GetBinUpEdge(ratios_hist[ns][r].GetNbinsX())
    yminM = 0.0
    ymaxM = hist[0][ns][r].GetMaximum()
    if hist[1][ns][r].GetMaximum() > ymaxM:
      ymaxM = hist[1][ns][r].GetMaximum()
    ymaxM=1.5*ymaxM
    if ymaxM<=yminM:
      ymaxM = 1.0
    xtitle = hist[0][ns][r].GetXaxis().GetTitle()
    ytitle = "Fraction of tracks"
    if not doNorm:
      ytitle = "Number of tracks"

    haxisMain  = ROOT.TH2D("haxisMain" ,ttitle,1,xmin ,xmax,1,yminM,ymaxM)
    
    haxisMain.GetXaxis().SetTitle(xtitle)
    haxisMain.GetXaxis().SetTitleOffset(1.2)
    haxisMain.GetYaxis().SetTitle(ytitle)
    haxisMain.GetYaxis().SetTitleOffset(1.4)

    haxisMain.Draw()
    hist[0][ns][r].Draw("PE,same")
    hist[1][ns][r].Draw("PE,same")
    
    legend = ROOT.TLegend(0.6, 0.7, 0.87, 0.87);
    legend.SetLineColor(0)
    legend.SetFillColor(0)
    legend.AddEntry(hist[0][ns][r], "%s [#mu=%.2f]"%(names[0],means[0]), "PL")
    legend.AddEntry(hist[1][ns][r], "%s [#mu=%.2f]"%(names[1],means[1]), "PL")
    legend.Draw("same")
    
    can.cd()
    pad2.Draw()
    pad2.cd()
    
    ymin = 0.9*ratios_hist[ns][r].GetMinimum()
    ymax = 1.1*ratios_hist[ns][r].GetMaximum()
    
    if ymin==0:
      ymin=0.75
      if ymax<=ymin:
        ymin=0.75*ymax
        ymax=1.25*ymax
        
    if ymax<=ymin:
      ymin=0.0
      ymax=2.0
    
    hraxes = getRatioAxes(xmin,xmax,ymin,ymax)
    
    line = ROOT.TLine(xmin,1.0,xmax,1.0)
    line.SetLineColor(1)
    line.SetLineStyle(2)
    
    hraxes.Draw("")
    ratios_hist[ns][r].Draw("PE,same")
    line.Draw("same")
    
    can.cd()
    pad1.Draw()
    pad2.Draw()
    
    can.SaveAs("%s/%s.png"%(thisdir,outname));
    can.SaveAs("%s/%s.pdf"%(thisdir,outname));
    
    can.Update()
    can.Clear()

    del haxisMain
    del hraxes
    del pad1
    del pad2
    del can

