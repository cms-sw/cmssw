from ROOT import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)


#_______________________________________________________________________________
def drawEtaLabel(minEta, maxEta, x=0.17, y=0.35, font_size=0.):
    label = minEta + " < |#eta| < " + maxEta
    tex = TLatex(x, y,label)
    if font_size > 0.:
      tex.SetFontSize(font_size)
      tex.SetTextSize(0.05)
      tex.SetNDC()
      tex.Draw()
      return tex
