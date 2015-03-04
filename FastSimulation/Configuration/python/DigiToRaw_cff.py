import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.DigiToRaw_cff import *

DigiToRaw.remove(siPixelRawData)
DigiToRaw.remove(SiStripDigiToRaw)
DigiToRaw.remove(castorRawData)
