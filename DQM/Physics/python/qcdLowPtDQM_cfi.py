# $Id: qcdLowPtDQM_cfi.py,v 1.11 2010/01/18 16:02:25 ameyer Exp $

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

qcdLowPtDQM = cms.EDAnalyzer("QcdLowPtDQM",
    hltTrgNames  = cms.untracked.vstring(
    'HLT_MinBiasBSC',
    'HLT_MinBiasBSC_OR',
    'HLT_MinBiasEcal',
    'HLT_MinBiasHcal',
    'HLT_MinBiasPixel_SingleTrack',
    'HLT_ZeroBiasPixel_SingleTrack',
    'HLT_L1Tech_HCAL_HF_coincidence_PM',
    'HLT_L1_BPTX',
#    'HLT_HFThreshold3',
#    'HLT_Activity_Ecal',
#    'HLT_Activity_PixelClusters',
    ),
                             
    verbose = cms.untracked.int32(3)
)
