import FWCore.ParameterSet.Config as cms

# by default, LHCInfo is now loaded from CondDB using a GT
ctppsLHCInfoLabel = cms.string("")

## minimal LHCInfo for 2016 data
#ctppsLHCInfoLabel = cms.string("ctpps_minimal")
#ctppsLHCInfoESSource_2016 = cms.ESSource("CTPPSLHCInfoESSource",
#  label = ctppsLHCInfoLabel,
#  validityRange = cms.EventRange("270293:min - 290872:max"),
#  beamEnergy = cms.double(6500),  # GeV
#  xangle = cms.double(185)  # murad
#)
