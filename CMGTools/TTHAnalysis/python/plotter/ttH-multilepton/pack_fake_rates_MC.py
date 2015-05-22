from math import *
from os.path import basename
import re

import sys
sys.argv.append('-b-')
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv.remove('-b-')
from array import *

def makeH2D(name,xedges,yedges):
    return ROOT.TH2F(name,name,len(xedges)-1,array('f',xedges),len(yedges)-1,array('f',yedges))

def fillSliceY(th2,plot1d,yvalue):
    ybin = th2.GetYaxis().FindBin(yvalue)
    for xbin in xrange(1,th2.GetNbinsX()+1):
        xval = th2.GetXaxis().GetBinCenter(xbin)
        xbin1d = plot1d.GetXaxis().FindBin(xval)
        th2.SetBinContent(xbin,ybin,plot1d.GetBinContent(xbin1d))
def readSliceY(th2,filename,plotname,yvalue):
    slicefile = ROOT.TFile.Open(filename)
    if not slicefile: raise RuntimeError, "Cannot open "+filename
    plot = slicefile.Get(plotname)
    if not plot: 
        slicefile.ls()
        raise RuntimeError, "Cannot find "+plotname+" in "+filename
    fillSliceY(th2,plot,yvalue)
    slicefile.Close()
def assemble2D(out,name,xedges,yedges,filepattern,plotname,yslices):
    out.cd()
    th2 = makeH2D(name,xedges,yedges)
    for yvalue,yname in yslices:
        readSliceY(th2,filepattern%yname,plotname,yvalue)
    out.WriteTObject(th2)
    return th2
    
if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] what path out")
    parser.add_option("-s", "--sel", dest="sel", default=None, help="Select");
    (options, args) = parser.parse_args()
    (what,path,outname) = args
    outfile = ROOT.TFile.Open(outname,"RECREATE")
    if what == "mvaTTH":
       ptbins_c = [ 0,10,15,20,30,45,70,100 ]
       etabins_c_el = [0, 1.479, 2.5]
       etabins_c_mu = [0, 1.5,   2.4]
       etaslices_c  = [ (0.4,"00_15"), (1.8,"15_24") ]
       etaslices_c1 = [ (0.4,"00_15"), (1.8,"15_21") ]
       #for WP in "06", "08":
       #    num = "mvaTTH_"+WP
       #    pt  = "pt_coarse"
       #    ptj = "ptJI_mvaTTH%s_coarse" % WP
       #    for src in "QCDMu", "TT": 
       #         assemble2D(outfile,"FR_wp%s_mu_%s_a_%s" % (WP,src,pt ), ptbins_c, etabins_c_mu, path+"/mu_wp"+WP+"_a_eta_%s.root", num+"_"+pt +"_"+src+"_red", etaslices_c)
       #         assemble2D(outfile,"FR_wp%s_mu_%s_a_%s" % (WP,src,ptj), ptbins_c, etabins_c_mu, path+"/mu_wp"+WP+"_a_eta_%s.root", num+"_"+ptj+"_"+src+"_red", etaslices_c)
       #    for src in "QCDEl", "TT": 
       #         assemble2D(outfile,"FR_wp%s_el_%s_a_%s" % (WP,src,pt ), ptbins_c, etabins_c_el, path+"/el_wp"+WP+"_a_eta_%s.root", num+"_"+pt +"_"+src+"_red", etaslices_c)
       #         assemble2D(outfile,"FR_wp%s_el_%s_a_%s" % (WP,src,ptj), ptbins_c, etabins_c_el, path+"/el_wp"+WP+"_a_eta_%s.root", num+"_"+ptj+"_"+src+"_red", etaslices_c)
       for WP in "06i","06ib":
           WP0 = re.sub(r"^(\d+).*",r"\1",WP)   # for binning
           WP1 = re.sub(r"^(\d+i?).*",r"\1",WP) # for numerator
           num = "mvaTTH_"+WP1
           ptj = "ptJI_mvaTTH%s_coarse" % WP0
           for ptBin in "", "_pt8", "_pt17", "_pt24":
               for src in "QCDMu", "TT": 
                    assemble2D(outfile,"FR_wp%s_mu_%s_a_%s%s" % (WP,src,ptj,ptBin), ptbins_c, etabins_c_mu, path+"/mu_wp"+WP+"_a_eta_%s"+ptBin+".root", num+"_"+ptj+"_"+src+"_red", etaslices_c)
           for ptBin in "", "_pt12", "_pt23", "_pt32":
               for src in "QCDEl", "TT": 
                    assemble2D(outfile,"FR_wp%s_el_%s_a_%s%s" % (WP,src,ptj,ptBin), ptbins_c, etabins_c_el, path+"/el_wp"+WP+"_a_eta_%s"+ptBin+".root", num+"_"+ptj+"_"+src+"_red", etaslices_c)
    if what == "iso":
       ptbins_c = [ 0,10,15,20,30,45,70,100 ]
       etabins_c_el = [0, 1.479, 2.5]
       etabins_c_mu = [0, 1.5,   2.4]
       etaslices_c  = [ (0.4,"00_15"), (1.8,"15_24") ]
       num = "relIso03_01"
       pt  = "pt_coarse"
       ptj = "ptGZ_coarse"
       for src in "QCDMu", "TT": 
            assemble2D(outfile,"FR_mu_%s_a_%s%s" % (src,ptj,ptBin), ptbins_c, etabins_c_mu, path+"/mu_a_eta_%s"+ptBin+".root", num+"_"+pt +"_"+src+"_red", etaslices_c)
            assemble2D(outfile,"FR_mu_%s_a_%s%s" % (src,ptj,ptBin), ptbins_c, etabins_c_mu, path+"/mu_a_eta_%s"+ptBin+".root", num+"_"+ptj+"_"+src+"_red", etaslices_c)
       for src in "QCDEl", "TT": 
            assemble2D(outfile,"FR_el_%s_a_%s" % (src,pt ), ptbins_c, etabins_c_el, path+"/el_a_eta_%s.root", num+"_"+pt +"_"+src+"_red", etaslices_c)
            assemble2D(outfile,"FR_el_%s_a_%s" % (src,ptj), ptbins_c, etabins_c_el, path+"/el_a_eta_%s.root", num+"_"+ptj+"_"+src+"_red", etaslices_c)

    outfile.ls()
