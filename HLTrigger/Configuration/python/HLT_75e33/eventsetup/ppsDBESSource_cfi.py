import FWCore.ParameterSet.Config as cms

ppsDBESSource = cms.ESSource("PoolDBESSource",
    DumpStat = cms.untracked.bool(False),
    beamEnergy = cms.double(6500),
    label = cms.string(''),
    timetype = cms.untracked.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
            record = cms.string('LHCInfoRcd'),
            tag = cms.string('LHCInfoEndFill_prompt_v2')
        ),
        cms.PSet(
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
            record = cms.string('CTPPSOpticsRcd'),
            tag = cms.string('PPSOpticalFunctions_offline_v6')
        )
    ),
    validityRange = cms.EventRange(0, 0, 1, 999999, 0, 0),
    xangle = cms.double(-1)
)
