import FWCore.ParameterSet.Config as cms

def customise(process):

    # From python/simEmtfDigis_cfi.py
    if hasattr(process, 'simEmtfDigis'):
        process.simEmtfDigis.spPCParams16.ZoneBoundaries = [0,36,54,96,127]
        process.simEmtfDigis.spPCParams16.UseNewZones    = True
        process.simEmtfDigis.RPCEnable                   = True
        process.simEmtfDigis.GEMEnable                   = True
        process.simEmtfDigis.IRPCEnable                  = True
        process.simEmtfDigis.ME0Enable                   = True
        process.simEmtfDigis.TTEnable                    = False
        process.simEmtfDigis.ME0Input                    = cms.InputTag('fakeSimMuonME0PadDigis')
        process.simEmtfDigis.Era                         = cms.string('Phase2C2')
        process.simEmtfDigis.spPAParams16.PtLUTVersion   = cms.int32(7)

    # From python/fakeEmtfParams_cff.py
    if hasattr(process, 'emtfParams'):
        process.emtfParams.PtAssignVersion = cms.int32(7)

    if hasattr(process, 'emtfForestsDB'):
        process.emtfForestsDB = cms.ESSource(
            "EmptyESSource",
            recordName = cms.string('L1TMuonEndCapForestRcd'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
            )

        process.emtfForests = cms.ESProducer(
            "L1TMuonEndCapForestESProducer",
            PtAssignVersion = cms.int32(7),
            bdtXMLDir = cms.string("2017_v7")
            )

    return process

