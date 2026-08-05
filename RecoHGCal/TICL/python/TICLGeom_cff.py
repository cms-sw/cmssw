import FWCore.ParameterSet.Config as cms

# TICL geometry SoAs are built per subdetector. HGCal is the default
# (data label "") since TICL consumers only use HGCal; the barrel
# calorimeters are available under their own labels and, like every
# EventSetup product, are only built when actually consumed. Each geometry
# comes with its rawDetId to dense id lookup table, usable in kernels via
# ticlgeom::denseIdOf.

ticlGeomESProducer = cms.ESProducer('TICLGeomESProducer@alpaka',
    detectors = cms.vstring('HGCal'),
    appendToDataLabel = cms.string(''),
)

ticlGeomLookupESProducer = cms.ESProducer('TICLGeomLookupESProducer@alpaka',
    src = cms.ESInputTag('', ''),
    appendToDataLabel = cms.string(''),
)

# per-layer positions (getPositionLayer replacement); geometry wide, one
# instance is enough for every detector selection
ticlGeomLayersESProducer = cms.ESProducer('TICLGeomLayersESProducer@alpaka',
    appendToDataLabel = cms.string(''),
)

ticlGeomECALESProducer = ticlGeomESProducer.clone(
    detectors = ['ECAL'],
    appendToDataLabel = 'ECAL',
)

ticlGeomECALLookupESProducer = ticlGeomLookupESProducer.clone(
    src = cms.ESInputTag('', 'ECAL'),
    appendToDataLabel = 'ECAL',
)

ticlGeomHCALESProducer = ticlGeomESProducer.clone(
    detectors = ['HCAL'],
    appendToDataLabel = 'HCAL',
)

ticlGeomHCALLookupESProducer = ticlGeomLookupESProducer.clone(
    src = cms.ESInputTag('', 'HCAL'),
    appendToDataLabel = 'HCAL',
)

# inclusive instance for modules whose detids can also be barrel cells
ticlGeomWithBarrelESProducer = ticlGeomESProducer.clone(
    detectors = ['ECAL', 'HCAL', 'HGCal'],
    appendToDataLabel = 'withBarrel',
)

ticlGeomWithBarrelLookupESProducer = ticlGeomLookupESProducer.clone(
    src = cms.ESInputTag('', 'withBarrel'),
    appendToDataLabel = 'withBarrel',
)

# HFNose eras include the nose cells in the default collection, so the
# HFNose TICL iterations find their cells under the default label
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toModify(ticlGeomESProducer, detectors = ['HGCal', 'HFNose'])

# These are @alpaka EventSetup producers, so they can only be scheduled in a
# process that has the Alpaka backends loaded. Attach them only under the
# phase2_hgcal era (which always sets up Alpaka), so they never leak into
# FastSim, Run 2 or Run 3 processes that load the HGCal local reco cff.
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal


def _addTICLGeomToReco(process):
    process.ticlGeomESProducer = ticlGeomESProducer
    process.ticlGeomLookupESProducer = ticlGeomLookupESProducer
    process.ticlGeomLayersESProducer = ticlGeomLayersESProducer
    process.ticlGeomWithBarrelESProducer = ticlGeomWithBarrelESProducer
    process.ticlGeomWithBarrelLookupESProducer = ticlGeomWithBarrelLookupESProducer


addTICLGeomToReco = phase2_hgcal.makeProcessModifier(_addTICLGeomToReco)
