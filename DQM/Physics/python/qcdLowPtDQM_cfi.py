# $Id: qcdLowPtDQM_cfi.py,v 1.2 2009/11/11 16:01:00 loizides Exp $

import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
#from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
#from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
#from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
#from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
#from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *


from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *

#import RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi
#process.load('RecoPixelVertexing.PixelVertexFinding.PixelVertexes')
#pixel3Vertices = RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi.pixelVertices.clone();
#pixel3Vertices.TrackCollection = 'pixel3ProtoTracks'
#pixel3Vertices.UseError = True
#pixel3Vertices.WtAverage = True
#pixel3Vertices.ZOffset = 5.
#pixel3Vertices.ZSeparation = 0.3
#pixel3Vertices.NTrkMin = 3
#pixel3Vertices.PtMin = 0.15

#import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
#pixel3ProtoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()

#pixelVertexFromClusters = cms.EDProducer('PixelVertexProducerClusters')

#dump = cms.EDAnalyzer('EventContentAnalyzer')

#from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *

siPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
#    CPE = cms.string('PixelCPETemplateReco'),
    VerboseLevel = cms.untracked.int32(0),
)

myRecoSequence = cms.Sequence(
#                              siPixelDigis*
#                              siPixelClusters*
                              siPixelRecHits)
#                            pixel3ProtoTracks *
#                            pixel3Vertices *
#                            pixelVertexFromClusters)

siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")

QcdLowPtDQM = cms.EDAnalyzer("QcdLowPtDQM",
#    hltTrgNames  = cms.untracked.vstring('HLT_MinBiasHcal',
#                                         'HLT_MinBiasEcal',
#                                         'HLT_MinBiasPixel'),
    verbose      = cms.untracked.int32(0),                                     
)

#process.p = cms.Path(myRecoSequence*process.QcdLowPtDQM+process.dqmSaver)
