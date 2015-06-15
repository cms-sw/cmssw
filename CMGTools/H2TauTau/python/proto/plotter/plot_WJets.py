import sys
import os
from uuid import uuid4

from CMGTools.RootTools.PyRoot import *
from CMGTools.RootTools.Style import *

from CMGTools.H2TauTau.proto.plotter.categories_TauMu import *

file = TFile(sys.argv[1])
tree = file.Get('H2TauTauTreeProducerTauMu')
# tree.Draw('svfitMass')

hists = []

def draw(var, cutN, cutD, cut=None, hist=None):
    id = uuid4()
    if cut:
        cutN = ' && '.join([cutN, cut])
        cutD = ' && '.join([cutD, cut])
    if hist is None:
        tree.Draw(var, cutN, '')
        hN = tree.GetHistogram().Clone('hN_{id}'.format(id=id))
        tree.Draw(var, cutD, 'same')
        hD = tree.GetHistogram().Clone('hD_{id}'.format(id=id))
    else:
        hN = hist.Clone('hN_{id}'.format(id=id))
        hD = hist.Clone('hD_{id}'.format(id=id))
        tree.Project(hN.GetName(), var, cutN)
        tree.Project(hD.GetName(), var, cutD)        
    sBlue.formatHisto( hN )
    hN.Sumw2()
    hD.Sumw2()
    hN.Scale(1/hN.GetEntries())
    hD.Scale(1/hD.GetEntries())
    ratio = hN.Clone('ratio_{id}'.format(id=id))
    ratio.Divide(hD)
    hN.Draw()
    hD.Draw('same')
    gPad.Update()
    hists.extend([hN, hD, ratio])
    return hN, hD, ratio

h = TH1F('h','', 50, 0, 500)
num, den, rat = draw('svfitMass', '1', cat_Inc, '1', h)

h = TH1F('h','', 10, 0, 300)
num, den, rat = draw('svfitMass', cat_Inc + ' && ' + cat_VBF_Rel, cat_VBF_Rel, '1', h)
num, den, rat = draw('svfitMass', cat_Inc + ' && ' + cat_J2, cat_J2, '1', h)
num, den, rat = draw('svfitMass', cat_Inc + ' && ' + cat_J1, cat_J1, '1', h)
