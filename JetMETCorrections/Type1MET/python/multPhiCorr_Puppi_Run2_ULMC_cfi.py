import FWCore.ParameterSet.Config as cms

# UL 2018 MC corrections
multPhiCorr_Puppi_ULMC2018 = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2018"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0214557,0.969428),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0167134,0.199296),
    ),
)

# UL 2017 MC corrections
multPhiCorr_Puppi_ULMC2017 = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2017"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0102265,-0.446416),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0198663,0.243182),
    ),
)

# UL 2016preVFP MC corrections
multPhiCorr_Puppi_ULMC2016preVFP = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2016preVFP"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0058341,-0.395049),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.00971595,-0.101288),
    ),
)

# UL 2016postVFP MC corrections
multPhiCorr_Puppi_ULMC2016postVFP = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2016postVFP"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0060447,-0.4183),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.008331,-0.0990046),
    ),
)
