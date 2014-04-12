# specialize the AlCa sequences for MC
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.AlCaRecoStreams_cff import *

hcalDigiAlCaMB.InputLabel = 'rawDataCollector'

