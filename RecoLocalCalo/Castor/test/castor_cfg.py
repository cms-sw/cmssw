
import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorProducts")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# specify the correct database tags which contain the updated gains and channelquality flags
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
# end of Db configuration

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
    	'file:../../../../../CMSSW_4_2_0_pre5/src/data_RAW2DIGI_L1Reco_RECO.root' # choose your input file here
	)
)

# load CASTOR default reco chain (from towers on)
process.load('RecoLocalCalo.Castor.Castor_cff')

# construct the module which executes the RechitCorrector
process.rechitcorrector = cms.EDProducer("RecHitCorrector",
	rechitLabel = cms.InputTag("castorreco","","RECO"), # choose the original RecHit collection
	revertFactor = cms.double(62.5) # this is the factor to go back to the original fC: 1/0.016
)

# tell to the CastorTower reconstruction that he should use the new corrected rechits
process.CastorTowerReco.inputprocess = "rechitcorrector"

process.MyOutputModule = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('rechitcorrector_output.root') # choose your output file
)

# execute the rechitcorrector and afterwards do the reco chain again (towers -> jets)
process.producer = cms.Path(process.rechitcorrector*process.CastorFullReco)
process.end = cms.EndPath(process.MyOutputModule)

