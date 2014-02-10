import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def simTrackProperties(plotter):
  draw_1D(plotter.targetDir, "track_pt", plotter.ext, plotter.treeTracks, "Track p_{T};Track p_{T} [GeV];Entries", "h_", "(100,0,200)", "pt", "")
  draw_1D(plotter.targetDir, "track_eta", plotter.ext, plotter.treeTracks, "Track |#eta|;Track |#eta|;Entries", "h_", "(100,1.45,2.5)", "eta", "")
  draw_1D(plotter.targetDir, "track_phi", plotter.ext, plotter.treeTracks, "Track #phi;Track #phi [rad];Entries", "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", "")
