import ROOT
import sys
import os

color=ROOT.kMagenta
marker=20
plots    = ['xy_view','thickness_vs_r','thickness_vs_eta']
rangeMin = [   0,      0.5,      0.5]
rangeMax = [ 200,      3.5,      3.5]

def drawHeader(title=None):

    """ a lazy header for the plots """

    txt=ROOT.TLatex()
    txt.SetNDC(True)
    txt.SetTextFont(42)
    txt.SetTextSize(0.04)
    txt.SetTextAlign(ROOT.kHAlignLeft+ROOT.kVAlignCenter)
    txt.DrawLatex(0.12,0.93,'#bf{CMS} #it{preliminary}')
    txt.SetTextAlign(ROOT.kHAlignCenter+ROOT.kVAlignCenter)
    txt.SetTextSize(0.035)
    if title:
        txt.DrawLatex(0.80,0.93,title)

def makePlotsFrom(key):

    """receives a TDirectoryFile with the plots from an analyzer and saves them in png/pdf"""

    c=ROOT.TCanvas('c','c',500,500)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.12)
    c.SetBottomMargin(0.12)

    tag=key.GetName()
    tag=tag.replace('plotter','')

    for n, p in enumerate(plots):
        for d in [8,9]:

            #c.SetRightMargin(0.03)
            #c.SetTopMargin(0.3)
            layers=range(-28,29) if d==8 else range(-22,23)
            for l in layers:
                if l == 0: continue
                pname='d%d_layer%d_%s'%(d,l,p)
                h=key.Get(pname)
                h.SetLineColor(color)
                h.SetMarkerColor(color)
                h.SetMarkerStyle(marker)
                h.Draw('COLZ')

                drawHeader(h.GetTitle())
                c.Modified()
                c.Update()
                for ext in ['png','pdf']:
                    c.SaveAs(outName+'/%s_%s_layer_%d.%s'%(p,d,l,ext))


ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gROOT.SetBatch(True)

url='geom_output.root'
if len(sys.argv)>1 :
    url=sys.argv[1]
    outName=str(sys.argv[1])[:-5]
    if not os.path.isdir(outName):
        os.mkdir(outName)

fIn=ROOT.TFile.Open(url)
for key in fIn.GetListOfKeys(): makePlotsFrom(key.ReadObj())
