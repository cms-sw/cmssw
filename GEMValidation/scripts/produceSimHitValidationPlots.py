import sys

from ROOT import *

from cuts import *
from drawPlots import *
from BaseValidation import *
from TrackValidation import *
from GEMSimHitValidation import *
from ME0SimHitValidation import *
from RPCSimHitValidation import *
from CSCSimHitValidation import *
from gemChamberNumbering import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
if __name__ == "__main__":

  ## Style
  gStyle.SetStatStyle(0);

  plotter = SimHitPlotter()
  simTrackProperties(plotter)
  gemSimTrackToSimHitMatchingLX(plotter) 
  gemSimTrackToSimHitMatchingLY(plotter) 
  gemSimTrackToSimHitMatchingEta(plotter) 
  gemSimTrackToSimHitMatchingPhi(plotter)
  me0SimTrackToSimHitMatchingLX(plotter) 
  me0SimTrackToSimHitMatchingLY(plotter) 
  me0SimTrackToSimHitMatchingEta(plotter) 
  me0SimTrackToSimHitMatchingPhi(plotter)
  
  for i in range(len(plotter.sel)):

    gemSimHitOccupancyXY(plotter,i)
    gemSimHitOccupancyRZ(plotter,i)
    gemSimHitTOF(plotter,i)
    #   gemSimhitMomentum(plotter,i) 
    #   GEMSimValidation.SimhitMomentum(plotter,i)

    """      
    rpcSimHitOccupancyXY(plotter,i)
    rpcSimHitOccupancyRZ(plotter,i)
    rpcSimHitTOF(plotter,i)
    
    cscSimHitOccupancyXY(plotter,i)
    cscSimHitOccupancyRZ(plotter,i)
    cscSimHitTOF(plotter,i)
    
    me0SimHitOccupancyXY(plotter,i)
    me0SimHitOccupancyRZ(plotter,i)
    me0SimHitTOF(plotter,i)
    """
    
  gemChamberNumbering(plotter)
