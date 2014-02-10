import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

from BaseValidation import *
from GEMDigiValidation import *



if __name__ == "__main__":  

  ## Style
  gStyle.SetStatStyle(0);
  plotter = DigiPlotter()
  
  gemGEMDigiOccupancyXY(plotter)
  gemGEMDigiOccupancyStripPhi(plotter)
  gemGEMDigiOccupancyStrip(plotter)
  gemGEMDigiBX(plotter)
  gemGEMDigiOccupancyRZ(plotter)
  
  gemGEMPadOccupancyXY(plotter)
  gemGEMPadOccupancyPadPhi(plotter)
  gemGEMPadOccupancyPad(plotter)
  gemGEMPadBX(plotter)
  gemGEMPadOccupancyRZ(plotter)

  gemGEMCoPadOccupancyXY(plotter)
  gemGEMCoPadOccupancyCoPadPhi(plotter)
  gemGEMCoPadOccupancyCoPad(plotter)
  gemGEMCoPadBX(plotter)
  gemGEMCoPadOccupancyRZ(plotter)
  
  simTrackDigiMatchingEta(plotter)
  simTrackDigiMatchingPhi(plotter)
  simTrackDigiMatchingLX(plotter)
  simTrackDigiMatchingLY(plotter)
  simTrackPadMatchingEta(plotter)
  simTrackPadMatchingPhi(plotter)
  simTrackPadMatchingLX(plotter)
  simTrackPadMatchingLY(plotter)
  simTrackCoPadMatchingEta(plotter)
  simTrackCoPadMatchingPhi(plotter)
  simTrackCoPadMatchingLX(plotter)
  simTrackCoPadMatchingLY(plotter)
