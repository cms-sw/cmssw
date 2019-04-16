import FWCore.ParameterSet.Config as cms

ctppsLHCInfoLabel = cms.string("ctpps_minimal")

# minimal LHCInfo for 2016 data
ctppsLHCInfoESSource_2016 = cms.ESSource("CTPPSLHCInfoESSource",
  label = ctppsLHCInfoLabel,
  validityRange = cms.EventRange("270293:min - 290872:max"),
  beamEnergy = cms.double(6500),  # GeV
  xangle = cms.double(185)  # murad
)
