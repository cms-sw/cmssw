import FWCore.ParameterSet.Config as cms

# EOY 2018 MC corrections
multPhiCorr_EOYMC2018 = cms.VPSet(
    cms.PSet(
      name=cms.string("EOYMC2018"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.296713,-0.141506),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.115685,0.0128193),
    ),
)

# EOY 2017 MC corrections without METv2 fix
multPhiCorr_EOYMC2017 = cms.VPSet(
    cms.PSet(
      name=cms.string("EOYMC2017"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.217714,0.493361),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.177058,-0.336648),
    ),
)

# EOY 2017 MC corrections with METv2 fix
multPhiCorr_EOYMC2017_METv2 = cms.VPSet(
    cms.PSet(
      name=cms.string("EOYMC2017_METv2"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.182569,0.276542),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.155652,-0.417633),
    ),
)

# EOY 2016 MC corrections
multPhiCorr_EOYMC2016 = cms.VPSet(
    cms.PSet(
      name=cms.string("EOYMC2016"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.195191,-0.170948),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.0311891,0.787627),
    ),
)
