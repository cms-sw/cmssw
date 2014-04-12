# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

#                                       { string record = "EcalIntercalibConstantsRcd"
#                                         string tag = "EcalIntercalibConstants_trivial" },

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('miscalib0.00')
    ), 
        cms.PSet(
            record = cms.string('EcalPedestalsRcd'),
            tag = cms.string('EcalPedestals_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_trivial')
        )),
    messagelevel = cms.untracked.uint32(2),
    catalog = cms.untracked.string('relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL'), ##cms_orcoff_int2r/CMS_COND_GENERAL"

    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_ECAL'), ##cms_orcoff_int2r/CMS_COND_ECAL"

    authenticationMethod = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(1),
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        data = cms.vstring('EcalIntercalibConstants')
    ), 
        cms.PSet(
            record = cms.string('EcalPedestalsRcd'),
            data = cms.vstring('EcalPedestals')
        ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            data = cms.vstring('EcalADCToGeVConstant')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            data = cms.vstring('EcalGainRatios')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            data = cms.vstring('EcalWeightXtalGroups')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            data = cms.vstring('EcalTBWeights')
        )),
    verbose = cms.untracked.bool(True)
)

process.printer = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.get)
process.ep = cms.EndPath(process.printer)

