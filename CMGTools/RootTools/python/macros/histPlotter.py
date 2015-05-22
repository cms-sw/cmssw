import fnmatch
from CMGTools.RootTools.PyRoot import *

import ROOT

histTypes = [ ROOT.TH1F,
              ROOT.TH1D,
              ROOT.TH2F,
              ROOT.TH2D,
              ROOT.TProfile ]

fileName = sys.argv[1]
file = TFile(fileName)

hists = []
for key in file.GetListOfKeys():
    obj = file.Get(key.GetName())
    if type(obj) in histTypes:
        hists.append(obj)
        print type(obj), key.GetName()
        locals()[key.GetName()] = obj


canvases = []
cx = 500
cy = 500
def draw(pattern):
    import pdb; pdb.set_trace()
    for hist in hists:
        name = hist.GetName()
        if fnmatch.fnmatch(name, pattern):
            can =  TCanvas(hist, hist, cy, cy)
            canvases.append( can )
            can.Draw()
            hist.Draw()
        

        
