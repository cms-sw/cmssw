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
  """  
  cscMatchingEfficiencyToStripsAndWires(plotter)
  cscMatchingEfficiencyToStripsAndWires_2(plotter)
  cscMatchingEfficiencyToAlctClct(plotter)
  cscMatchingEfficiencyToAlctClct_2(plotter)
  cscMatchingEfficiencyToLct(plotter)

  cscMatchingEfficiencyToStripsAndWires(plotter,2)
  cscMatchingEfficiencyToStripsAndWires_2(plotter,2)
  cscMatchingEfficiencyToAlctClct(plotter,2)
  cscMatchingEfficiencyToAlctClct_2(plotter,2)
  cscMatchingEfficiencyToLct(plotter,2)

  cscMatchingEfficiencyToStripsAndWires(plotter,3)
  cscMatchingEfficiencyToStripsAndWires_2(plotter,3)
  cscMatchingEfficiencyToAlctClct(plotter,3)
  cscMatchingEfficiencyToAlctClct_2(plotter,3)
  cscMatchingEfficiencyToLct(plotter,3)

  cscMatchingEfficiencyToStripsAndWires(plotter,4)
  cscMatchingEfficiencyToStripsAndWires_2(plotter,4)
  cscMatchingEfficiencyToAlctClct(plotter,4)
  cscMatchingEfficiencyToAlctClct_2(plotter,4)
  cscMatchingEfficiencyToLct(plotter,4)
  """
