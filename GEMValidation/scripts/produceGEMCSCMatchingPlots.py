
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
from GEMCSCValidation import *


if __name__ == "__main__":  

  plotter = GEMCSCStubPlotter()
  for i in range(len(plotter.stationsToUse)):
    st = plotter.stationsToUse[i]
    print "Processing station ", plotter.stations.reverse_mapping[st]
    simTrackToCscSimHitMatching(plotter,st)
    simTrackToCscStripsWiresMatching(plotter,st)
    simTrackToCscStripsWiresMatching_2(plotter,st)
    simTrackToCscAlctClctMatching(plotter,st)
    simTrackToCscAlctClctMatching_2(plotter,st)
    simTrackToCscLctMatching(plotter,st)
#    simTrackToCscMpLctMatching(plotter,st)
