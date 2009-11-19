# $Id: qcdLowPtDQM_cfi.py,v 1.5 2009/11/17 06:45:14 loizides Exp $

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
    'HLT_Activity_EcalREM',
    'HLT_Activity_L1A',
    'HLT_Activity_PixelClusters',
    'HLT_BackwardBSC',
    'HLT_CSCBeamHalo',
    'HLT_ForwardBSC',
    'HLT_FwdJet20U',
    'HLT_HFThreshold',
    'HLT_HighMultiplicityBSC',
    'HLT_L1SingleEG2',
    'HLT_L1SingleEG2_NoBPTX',
    'HLT_L1SingleEG5',
    'HLT_L1SingleForJet',
    'HLT_L1_BPTX',
    'HLT_L1_BPTX_MinusOnly',
    'HLT_L1_BPTX_PlusOnly',
    'HLT_L1_BSC',
    'HLT_L1_HFtech',
    'HLT_MinBias',
    'HLT_MinBiasBSC',
    'HLT_MinBiasBSC_OR',
    'HLT_MinBiasEcal',
    'HLT_MinBiasHcal',
    'HLT_MinBiasPixel_DoubleIsoTrack5',
    'HLT_MinBiasPixel_DoubleTrack',
    'HLT_MinBiasPixel_SingleTrack',
    'HLT_Mu3',
    'HLT_Random',
    'HLT_ZeroBias'
    ),
                             
    verbose = cms.untracked.int32(3)
)
