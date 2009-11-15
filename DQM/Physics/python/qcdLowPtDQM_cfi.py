# $Id: qcdLowPtDQM_cfi.py,v 1.3 2009/11/13 09:59:18 loizides Exp $

import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi          import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi         import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff    import *

clusterVertices = cms.EDFilter("PixelVertexProducerClusters") 

siPixelDigis.InputLabel = cms.InputTag("source")

myRecoSeq1 = cms.Sequence(
    siPixelDigis    *
    siPixelClusters
)

myRecoSeq2 = cms.Sequence(
    siPixelRecHits  *
    clusterVertices
)

myRecoSeq = cms.Sequence(myRecoSeq1 * myRecoSeq2)

QcdLowPtDQM = cms.EDAnalyzer("QcdLowPtDQM",
    hltTrgNames  = cms.untracked.vstring('HLT_MinBiasHcal',
                                         'HLT_MinBiasEcal',
                                         'HLT_MinBiasPixel'),
    verbose      = cms.untracked.int32(3),                                     
)
