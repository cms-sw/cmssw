import FWCore.ParameterSet.Config as cms

# This config file can be used to retrieve constants from the Condition Data Base. 
# Ricky Egeland 18/08/06 Modif Alex Zabi: change into cff file
PoolDBESSource = cms.ESSource("PoolDBESSource",
    siteLocalConfig = cms.untracked.bool(False),
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_from_online')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant_prelim_V0')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios_lab_prelim_V0')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string('EcalIntercalibConstants_cosmic_1xtal_V0')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_standard_V0')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_standard_V0')
        )),
    messagelevel = cms.untracked.uint32(0),
    catalog = cms.untracked.string('xmlcatalog_file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/config_files/frontierCatalog.xml'),
    timetype = cms.string('runnumber'),
    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_ECAL_H4_COND'), ##cmsfrontier.cern.ch:8000/Frontier/CMS_ECAL_H4_COND"

    authenticationMethod = cms.untracked.uint32(0)
)

getCond = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
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
            record = cms.string('EcalIntercalibConstantsRcd'),
            data = cms.vstring('EcalIntercalibConstants')
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


