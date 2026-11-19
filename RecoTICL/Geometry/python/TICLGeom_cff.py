import FWCore.ParameterSet.Config as cms

# TICL geometry SoAs are built per subdetector. HGCal is the default
# (data label "") since TICL consumers only use HGCal; the barrel
# calorimeters are available under their own labels and, like every
# EventSetup product, are only built when actually consumed. Each product
# holds the cells, the detid lookup table and the per-layer positions;
# kernels read it through its ConstView.

ticlGeomESProducer = cms.ESProducer('TICLGeomESProducer@alpaka',
    detectors = cms.vstring('HGCal'),
    appendToDataLabel = cms.string(''),
)

ticlGeomECALESProducer = ticlGeomESProducer.clone(
    detectors = ['ECAL'],
    appendToDataLabel = 'ECAL',
)

ticlGeomHCALESProducer = ticlGeomESProducer.clone(
    detectors = ['HCAL'],
    appendToDataLabel = 'HCAL',
)

# inclusive instance for modules whose detids can also be barrel cells
ticlGeomWithBarrelESProducer = ticlGeomESProducer.clone(
    detectors = ['ECAL', 'HCAL', 'HGCal'],
    appendToDataLabel = 'withBarrel',
)

# HFNose eras include the nose cells in the default collection, so the
# HFNose TICL iterations find their cells under the default label
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toModify(ticlGeomESProducer, detectors = ['HGCal', 'HFNose'])

# These are @alpaka EventSetup producers, so they must only reach a process that
# loads the Alpaka accelerator. RecoLocalCalo/Configuration/hgcalLocalReco_cff
# adds them to the HGCal reco Task, so their EventSetup modules are instantiated
# only when that Task is scheduled, i.e. when HGCal local reco actually runs
# (which is where the accelerator is present). Processes that only load the reco
# sequence without scheduling it, such as FastSim or the tracker geometry dumps,
# never instantiate them.
