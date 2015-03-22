import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.DigiToRawDM_cff import *

DigiToRaw.remove(siPixelRawData)
DigiToRaw.remove(SiStripDigiToRaw)
# note: no need to remove castorRawData: it is not there
