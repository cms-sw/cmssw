# $Id: qcdLowPtDQM_cfi.py,v 1.7 2009/11/23 10:06:49 loizides Exp $

import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi          import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi         import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff    import *

siPixelDigis.InputLabel = cms.InputTag("source")

myRecoSeq1 = cms.Sequence(
    siPixelDigis    *
    siPixelClusters
)

myRecoSeq2 = cms.Sequence(
    siPixelRecHits
)

myRecoSeq = cms.Sequence(myRecoSeq1 * myRecoSeq2)

QcdLowPtDQM = cms.EDAnalyzer("QcdLowPtDQM",
    hltTrgNames  = cms.untracked.vstring(
    'HLT_MinBiasBSC',
    'HLT_L1_BSC',
    'HLT_MinBiasBSC_OR',
    'HLT_MinBiasEcal',
    'HLT_MinBiasHcal',
    'HLT_ZeroBiasPixel_SingleTrack',
    'HLT_HighMultiplicity_BSC',
#    'HLT_HFThreshold3',
#    'HLT_Activity_Ecal',
#    'HLT_Activity_PixelClusters',
    ),
                             
    verbose = cms.untracked.int32(3)
)
