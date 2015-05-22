import os, sys

import CMGTools.RootTools.StartUp 

from CMGTools.RootTools.RootTools import *

from ROOT import gROOT, TFile, TCanvas, gPad, TBrowser, TH2F, TH1F, TH1D , TProfile, TLegend

gROOT.Macro( os.path.expanduser( '~/rootlogon.C' ) )

# adding current directory in PYTHONPATH
sys.path.append('.')

if __name__ == '__main__':
    events = Chain('Events', sys.argv[1])
    # lumis = Chain('LuminosityBlocks', sys.argv[1])
