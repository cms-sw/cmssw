import FWCore.ParameterSet.Config as cms

src1 = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals')
    ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string('EcalIntercalibConstants')
        ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups')
        )),
    messagelevel = cms.untracked.uint32(1),
    catalog = cms.untracked.string('file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2004/v2/PoolFileCatalog.xml'),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2004/v2/ecal2004condDB.db'),
    authenticationMethod = cms.untracked.uint32(0)
)

getCond = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        data = cms.vstring('EcalPedestals')
    ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            data = cms.vstring('EcalTBWeights')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            data = cms.vstring('EcalGainRatios')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            data = cms.vstring('EcalIntercalibConstants')
        ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            data = cms.vstring('EcalADCToGeVConstant')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            data = cms.vstring('EcalWeightXtalGroups')
        )),
    verbose = cms.untracked.bool(True)
)


