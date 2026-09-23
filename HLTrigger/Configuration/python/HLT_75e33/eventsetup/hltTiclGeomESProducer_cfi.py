import FWCore.ParameterSet.Config as cms

# HGCal TICLGeom SoA cells for the HLT, under the HLT-owned data label so it
# never collides with the offline reco instance when HLT and RECO run together
hltTiclGeomESProducer = cms.ESProducer('TICLGeomESProducer@alpaka',
    detectors = cms.vstring('HGCal'),
    appendToDataLabel = cms.string('')
)
