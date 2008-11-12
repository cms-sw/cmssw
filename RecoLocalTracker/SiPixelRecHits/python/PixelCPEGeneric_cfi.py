import FWCore.ParameterSet.Config as cms

PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",

    ComponentName = cms.string('PixelCPEGeneric'),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    PixelErrorParametrization = cms.string('NOTcmsim'),

    # Allows cuts to be optimized
    eff_charge_cut_lowX = cms.double(0.0),
    eff_charge_cut_lowY = cms.double(0.0),
    eff_charge_cut_highX = cms.double(1.0),
    eff_charge_cut_highY = cms.double(1.0),
    size_cutX = cms.double(3.0),
    size_cutY = cms.double(3.0),

    # ggiurgiu@jhu.edu
    inflate_errors = cms.bool(False),
    inflate_all_errors_no_trk_angle = cms.bool(False),

    # petar, for clusterProbability() from TTRHs
    ClusterProbComputationFlag = cms.int32(0)         
)


