import FWCore.ParameterSet.Config as cms

PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    # dfehling
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    ComponentName = cms.string('PixelCPEGeneric'),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    PixelErrorParametrization = cms.string('NOTcmsim')
)


