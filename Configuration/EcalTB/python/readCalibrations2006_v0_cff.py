import FWCore.ParameterSet.Config as cms

src1 = cms.ESSource("PoolDBESSource",
    messagelevel = cms.untracked.uint32(1),
    loadAll = cms.bool(True),
    #                               string connect = "sqlite_file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2004/v2/ecal2004condDB.db"
    #		               untracked string catalog = "file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2004/v2/PoolFileCatalog.xml"	
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('EcalIntercalibConstants')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant')
        )),
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


