import FWCore.ParameterSet.Config as cms

# minimal LHCInfo for 2016 data
ctppsLHCInfoESSource = cms.ESSource("CTPPSLHCInfoESSource",
  label = cms.string("ctpps_minimal"),
  validityRange = cms.EventRange("270293:min - 290872:max"),
  beamEnergy = cms.double(6500),  # GeV
  xangle = cms.double(185)  # murad
)
