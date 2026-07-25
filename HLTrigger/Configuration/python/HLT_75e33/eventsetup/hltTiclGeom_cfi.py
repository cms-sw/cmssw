import FWCore.ParameterSet.Config as cms

# TICLGeom SoA geometry EventSetup producers (RecHitTools replacement),
# needed by the HGCal layer cluster, TICL and HGCal egamma HLT modules.
# The withBarrel instances feed the ticl_barrel EB/HB layer clustering;
# like every ES product they are only built when a scheduled module
# consumes them, so importing them unconditionally is free otherwise.
from RecoHGCal.TICL.TICLGeom_cff import (
    ticlGeomESProducer,
    ticlGeomLookupESProducer,
    ticlGeomLayersESProducer,
    ticlGeomWithBarrelESProducer,
    ticlGeomWithBarrelLookupESProducer,
)
