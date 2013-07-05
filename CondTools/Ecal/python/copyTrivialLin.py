import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_mc')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalChannelStatusRcd'),
            tag = cms.string('EcalChannelStatus_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string('EcalIntercalibConstants_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibErrorsRcd'),
            tag = cms.string('EcalIntercalibErrors_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsMCRcd'),
            tag = cms.string('EcalIntercalibConstantsMC_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalClusterCrackCorrParametersRcd'),
            tag = cms.string('EcalClusterCrackCorrParameters_mc')
        ),
        cms.PSet(
            record = cms.string('EcalClusterEnergyUncertaintyParametersRcd'),
            tag = cms.string('EcalClusterEnergyUncertaintyParameters_mc')
        ),
        cms.PSet(
            record = cms.string('EcalClusterEnergyCorrectionParametersRcd'),
            tag = cms.string('EcalClusterEnergyCorrectionParameters_mc')
        ),
        cms.PSet(
            record = cms.string('EcalClusterEnergyCorrectionObjectSpecificParametersRcd'),
            tag = cms.string('EcalClusterEnergyCorrectionObjectSpecificParameters_mc')
        ),
        cms.PSet(
             record = cms.string('EcalTimeCalibConstantsRcd'),
             tag = cms.string('EcalTimeCalibConstants_mc')
        ),
        cms.PSet(
            record = cms.string('EcalClusterLocalContCorrParametersRcd'),
            tag = cms.string('EcalClusterLocalContCorrParameters_mc')
        ))
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        container = cms.string('EcalPedestals')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            container = cms.string('EcalADCToGeVConstant')
        ), 
        cms.PSet(
            record = cms.string('EcalChannelStatusRcd'),
            container = cms.string('EcalChannelStatus')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            container = cms.string('EcalIntercalibConstants')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibErrorsRcd'),
            container = cms.string('EcalIntercalibErrors')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsMCRcd'),
            container = cms.string('EcalIntercalibConstantsMC')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            container = cms.string('EcalGainRatios')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            container = cms.string('EcalWeightXtalGroups')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            container = cms.string('EcalTBWeights')
        ), 
        cms.PSet(
            record = cms.string('EcalClusterEnergyUncertaintyParametersRcd'),
            container = cms.string('EcalClusterEnergyUncertaintyParameters')
        ),
        cms.PSet(
            record = cms.string('EcalClusterEnergyCorrectionParametersRcd'),
            container = cms.string('EcalClusterEnergyCorrectionParameters')
        ),
        cms.PSet(
            record = cms.string('EcalClusterEnergyCorrectionObjectSpecificParametersRcd'),
            container = cms.string('EcalClusterEnergyCorrectionObjectSpecificParameters')
        ),
        cms.PSet(
            record = cms.string('EcalClusterLocalContCorrParametersRcd'),
            container = cms.string('EcalClusterLocalContCorrParameters')
        ),
        cms.PSet(
            record = cms.string('EcalTimeCalibConstantsRcd'),
            container = cms.string('EcalTimeCalibConstants')
        ))
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)

