import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.Tree_RECO_cff import *

#local reconstruction
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
recolocal = cms.Sequence( siPixelDigis*siPixelClusters*
                          siStripDigis*siStripZeroSuppression*siStripClusters)
siPixelDigis.InputLabel = 'rawDataCollector'

#Schedule
recolocal_step  = cms.Path( recolocal )
schedule = cms.Schedule( recolocal_step, recotrack_step, ntuple_step )
