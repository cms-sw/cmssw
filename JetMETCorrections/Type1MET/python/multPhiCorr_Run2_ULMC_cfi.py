import FWCore.ParameterSet.Config as cms

# UL 2018 MC corrections
multPhiCorr_UL2018MC = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2018MC"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.183518,0.546754),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.192263,-0.42121),
    ),
)

# UL 2017 MC corrections
multPhiCorr_UL2017MC = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2017MC"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.300155,1.90608),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.300213,-2.02232),
    ),
)

# UL 2016preVFP MC corrections
multPhiCorr_UL2016preVFPMC = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016preVFPMC"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.188743,0.136539),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0127927,0.117747),
    ),
)

# UL 2016postVFP MC corrections
multPhiCorr_UL2016postVFPMC = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016postVFPMC"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.153497,-0.231751),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.00731978,0.243323),
    ),
)
