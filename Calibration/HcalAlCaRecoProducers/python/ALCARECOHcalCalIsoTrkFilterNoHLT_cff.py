import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.alcaIsoTracksFilter_cff import *

seqALCARECOHcalCalIsoTrkFilterNoHLT = cms.Sequence(alcaIsoTracksFilter)


