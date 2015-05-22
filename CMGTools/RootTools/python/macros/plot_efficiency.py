#!/bin/env python

import sys
import copy
from CMGTools.RootTools.RootInit import *
from CMGTools.RootTools.Style import sBlack, sBlue, styleSet

from ROOT import TGraphAsymmErrors

args = sys.argv[1:]
# fileName = args[0]


def printHist(hist):
    print hist.GetNbinsX(), hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()

class Efficiency(object):
    def __init__(self, region, file, legend='', rebin=None):
        self.dir_num = file.Get( '_'.join([region, 'Num']) ) 
        self.dir_denom = file.Get( '_'.join([region, 'Denom']) ) 
        # self.eff = self.num.Clone('eff')
        # self.eff.Divide(self.denom)
        sname = file.GetName()
        sname = sname.replace('EfficiencyAnalyzer/EfficiencyAnalyzer.root', '')
        self.desc = ','.join( [legend, region])
        def load( dir, hists, rebin):
            for key in dir.GetListOfKeys():
                keyname = key.GetName()
                histname = keyname.split('_')[-1]
                hist = dir.Get( keyname )
                if rebin is not None:
                    rebin = int(rebin)
                    hist.Rebin( rebin )
                hist.Sumw2()
                hists[histname] = hist

        self.hists_num = {}
        load(self.dir_num, self.hists_num, rebin)
        self.hists_denom = {}
        load(self.dir_denom, self.hists_denom, rebin)
        self.hists_eff = {}
        for histName, num in self.hists_num.iteritems():
            denom = self.hists_denom[histName]
            # eff = TGraphAsymmErrors( num.GetNbinsX() )
            eff = num.Clone( '_'.join([histName,'eff']) )
            #printHist(num)
            #printHist(denom)
            eff.Divide(num, denom,1,1,'b')
            self.hists_eff[histName] = eff
        self.support = {}
        self.xtitle = None
        self.ytitle = None

    def formatHistos(self, style):
        map( style.formatHisto, self.hists_eff.values() )
        map( style.formatHisto, self.hists_num.values() )
        map( style.formatHisto, self.hists_denom.values() )

    def draw(self, name, ymin=0, ymax=1.1, same=False):
        if not same:
            h = self.hists_num[name]
            sup = TH2F( name, '',
                        h.GetNbinsX(),
                        h.GetXaxis().GetXmin(),
                        h.GetXaxis().GetXmax(),
                        10, ymin, ymax)
            self.support[name] = sup
            sup.SetStats(0)
            sup.SetTitle( h.GetTitle() )
            if self.xtitle is None:
                sup.SetXTitle( h.GetXaxis().GetTitle() )
            else:
                sup.SetXTitle( self.xtitle )
            if self.ytitle is None:
                sup.SetYTitle( h.GetYaxis().GetTitle() )
            else:
                sup.SetYTitle( self.ytitle )
            sup.Draw()
        self.hists_eff[name].Draw('Psame')


import sys
from optparse import OptionParser

parser = OptionParser()
parser.usage = '%prog <region> <var> <dir1> [dir2 ..]'

parser.add_option("-r", "--rebin",
                  dest="rebin",
                  default=None,help='rebin factor for your histograms')
parser.add_option("-m", "--min",
                  dest="ymin",
                  default=0.0,help='y min')
parser.add_option("-M", "--max",
                  dest="ymax",
                  default=1.1,help='y max')

options, args = parser.parse_args()
if len(args)<3:
    print 'provide at least 3 arguments: <region> <var> <input files>'
    sys.exit(1)

options.ymin = float(options.ymin)
options.ymax = float(options.ymax)
region = args[0]
var = args[1]
files = args[2:]



def setMVAStyle():
    mitStyles = map(copy.deepcopy, [sBlack]*3)
    danStyles = map(copy.deepcopy, [sBlue]*3)
    def setMarkStyle(styles, start):
        for style, mark in zip(styles, range(start,start+3)):
            style.markerStyle = mark
    setMarkStyle(mitStyles, 20)
    setMarkStyle(danStyles, 24)
    styles = mitStyles
    styles.extend(danStyles)
    return styles

## def setMVAEffs( effs ):
##     pattern = re.compile('Analyzer_(.*)/.*') 
##     for eff in effs:
##         m = pattern.match( eff.desc )
##         if m is not None:
##             eff.desc = m.group(1)


# styles = setMVAStyle()
styles = [sBlue, sBlackSquares]
styles[1].markerStyle = 25

keeper = []

def setup( fileName, index ):
    print 'setup', fileName
    ffileName = '/'.join( [fileName, 'EfficiencyAnalyzer.root'] )
    file = TFile( ffileName)
    legend = ''
    pattern = re.compile('.*Analyzer_(.*)$')
    m = pattern.match( fileName )
    if m is not None:
        legend = m.group(1)
        print legend
    eff = Efficiency( region, file, legend, options.rebin)
    eff.formatHistos( styles[index] )
    eff.ytitle='Efficiency'
    keeper.extend( [file] )
    return eff

effs = []
for index, file in enumerate(files):
    effs.append( setup( file, index ) )

legDX = 0.25
legDY = 0.20
legX1 = 0.65
legY1 = 0.15
legX2 = legX1 + legDX
legY2 = legY1 + legDY

legend = None

    
def draw(name, ymin=options.ymin, ymax=options.ymax):
    same = False
    global legend
    if legend is None:
        legend = TLegend(legX1, legY1, legX2, legY2)
    else:
        legend.Clear()
    keeper.append( legend )
    for eff in effs:
        eff.draw( name, ymin, ymax, same)
        if not same:
            same = True
        legend.AddEntry(eff.hists_eff[name], eff.desc, 'lp' )
    legend.Draw('same')

# draw('pu', 0.5, 1.05)

def drawHist(name, nord='num', norm=False, same=''):
    same = same
    legend = TLegend(0.6, 0.15, 0.89, 0.35)
    keeper.append( legend )
    for eff in effs:
        hists = getattr( eff, '_'.join(['hists',nord]))
        if norm:
            hists[name].DrawNormalized( same )
        else:
            hists[name].Draw( same )
        if same=='':
            same = 'same'
        legend.AddEntry(eff.hists_eff[name], eff.desc, 'lp' )
    legend.Draw('same')
        
    
draw(var)


