import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#  $Id: simplePhotonAnalyzer.cfi,v 1.6 2008/03/16 23:13:04 nancy Exp $ 
#
simplePhotonAnalyzer = cms.EDAnalyzer("SimplePhotonAnalyzer",
    mcProducer = cms.string('source'),
    photonCollection = cms.string(''),
    phoProducer = cms.string('photons'),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesFromCTFTracks')
)


