import sys

from ROOT import *
xo
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
  
  cscMatchingEfficiencyToStripsAndWires(plotter)
  cscMatchingEfficiencyToStripsAndWires_2(plotter)
  cscMatchingEfficiencyToAlctClct(plotter)
  cscMatchingEfficiencyToAlctClct_2(plotter)
  cscMatchingEfficiencyToLct(plotter)
