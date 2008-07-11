import FWCore.ParameterSet.Config as cms

# Magnetic Field
# Geometries
from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# Old stand alone muon seed producer used priod to 2-X-X
#include "RecoMuon/MuonSeedGenerator/data/standAloneMuonSeeds.cfi"
# New standalone muon producer to be used in 2-X-X
from RecoMuon.MuonSeedGenerator.standAloneMuonSeedProducer_cfi import *
# Old field map
#include "RecoMuon/MuonSeedGenerator/data/ptSeedParameterization_40T_851.cfi"
# New map at 3.8 T
from RecoMuon.MuonSeedGenerator.ptSeedParameterization_38T_cfi import *


