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

  ## Style
  gStyle.SetStatStyle(0);
  plotter = GEMCSCStubPlotter()
#  print plotter.stations.size()
  """
  for st in plotter.stationsToUse:
    cscMatchingEfficiencyToStripsAndWires(plotter,st)
    cscMatchingEfficiencyToStripsAndWires_2(plotter,st)
    cscMatchingEfficiencyToAlctClct(plotter,st)
    cscMatchingEfficiencyToAlctClct_2(plotter,st)
    cscMatchingEfficiencyToLct(plotter,st)
  """
