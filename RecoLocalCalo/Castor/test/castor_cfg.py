
import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorProducts")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.CastorDbProducer = cms.ESProducer("CastorDbProducer")
process.es_pool = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
   authenticationMethod = cms.untracked.uint32(0),
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('CastorPedestalsRcd'),
           tag = cms.string('CastorPedestals_v2.0_offline')
           ),
       cms.PSet(
           record = cms.string('CastorPedestalWidthsRcd'),
           tag = cms.string('CastorPedestalWidths_v2.0_offline')
           ),
       cms.PSet(
           record = cms.string('CastorGainsRcd'),
           tag = cms.string('CastorGains_v2.0_offline')
           ),
       cms.PSet(
           record = cms.string('CastorGainWidthsRcd'),
 	   tag = cms.string('CastorGainWidths_v2.0_offline')
           ),
       cms.PSet(
           record = cms.string('CastorQIEDataRcd'),
 	   tag = cms.string('CastorQIEData_v2.0_offline')
           ),
       cms.PSet(
           record = cms.string('CastorChannelQualityRcd'),
           tag = cms.string('CastorChannelQuality_v2.0_offline')
           ),
       cms.PSet(
           record = cms.string('CastorElectronicsMapRcd'),
     	   tag = cms.string('CastorElectronicsMap_v2.0_offline')
           )
   )
)

#process.es_pool2 = cms.ESSource("PoolDBESSource",
#     process.CondDBSetup,
#     timetype = cms.string('runnumber'),
#    connect = cms.string('sqlite_file:testExample.db'),
#	authenticationMethod = cms.untracked.uint32(0),
#	toGet = cms.VPSet(
#		cms.PSet(
#			record = cms.string('CastorChannelQualityRcd'),
#			tag = cms.string('castor_channelstatus_v1.0_test')
#		)
#	)
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = 
cms.untracked.vstring('file:../../../data_RAW2DIGI_L1Reco_RECO.root')
)

process.load('RecoLocalCalo.Castor.Castor_cff')

process.rechitcorrector = cms.EDProducer("RecHitCorrector",
	rechitLabel = cms.InputTag("castorreco","","RERECO"),
	revertFactor = cms.double(62.5)
)

process.MyOutputModule = cms.OutputModule("PoolOutputModule",
    #outputCommands = cms.untracked.vstring('keep recoGenParticles*_*_*_*', 'keep *_castorreco_*_*', 'keep *_Castor*Reco*_*_CastorProducts'),
    #outputCommands = cms.untracked.vstring('keep *_Castor*Reco*_*_CastorProducts','drop *_CastorFastjetReco*_*_CastorProducts','drop *_CastorTowerCandidateReco*_*_CastorProducts'),
    fileName = cms.untracked.string('rechitcorrector_output.root')
)

process.producer = cms.Path(process.rechitcorrector*process.CastorFullReco)
process.end = cms.EndPath(process.MyOutputModule)

