import FWCore.ParameterSet.Config as cms

src1 = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('EcalIntercalibConstants')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant')
        )),
    messagelevel = cms.untracked.uint32(1),
    catalog = cms.untracked.string('file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2004/v2/PoolFileCatalog.xml'),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2004/v2/ecal2004condDB.db'),
    authenticationMethod = cms.untracked.uint32(0)
)

getCalibrations = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        data = cms.vstring('EcalIntercalibConstants')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            data = cms.vstring('EcalADCToGeVConstant')
        )),
    verbose = cms.untracked.bool(True)
)


