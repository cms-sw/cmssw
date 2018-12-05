import FWCore.ParameterSet.Config as cms

ctppsOpticalFunctionsESSource = cms.ESSource("CTPPSOpticalFunctionsESSource",
    xangle1 = cms.double(185),
    fileName1 = cms.FileInPath("CondFormats/CTPPSReadoutObjects/data/year_2016/optical_functions.root"),

    xangle2 = cms.double(185),
    fileName2 = cms.FileInPath("CondFormats/CTPPSReadoutObjects/data/year_2016/optical_functions.root"),

    scoringPlanes = cms.VPSet(
      cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-203.826) ),  # RP 002
      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-212.551) ),  # RP 003
      cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+203.826) ),  # RP 102
      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+212.551) ),  # RP 103
    )
)
