import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *

## regional seeded unpacking for specialized HLT paths
siPixelDigisRegional = siPixelDigis.clone()
siPixelDigisRegional.Regions = cms.PSet(
    inputs = cms.VInputTag( "hltL2EtCutDoublePFIsoTau45Trk5" ),
    deltaPhi = cms.vdouble( 0.5 ),
    maxZ = cms.vdouble( 24. ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
)

