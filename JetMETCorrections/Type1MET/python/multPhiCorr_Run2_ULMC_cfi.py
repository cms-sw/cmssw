import FWCore.ParameterSet.Config as cms

# UL 2018 MC corrections
multPhiCorr_ULMC2018 = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2018"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.183518,0.546754),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.192263,-0.42121),
    ),
)

# UL 2017 MC corrections
multPhiCorr_ULMC2017 = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2017"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.300155,1.90608),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.300213,-2.02232),
    ),
)

# UL 2016preVFP MC corrections
multPhiCorr_ULMC2016preVFP = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2016preVFP"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.188743,0.136539),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0127927,0.117747),
    ),
)

# UL 2016postVFP MC corrections
multPhiCorr_ULMC2016postVFP = cms.VPSet(
    cms.PSet(
      name=cms.string("ULMC2016postVFP"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.153497,-0.231751),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.00731978,0.243323),
    ),
)
