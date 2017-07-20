import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('LUT')

process.source = cms.Source("EmptySource")
process.source.firstRun = cms.untracked.uint32( __RUN__ )
process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1)
)

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.Geometry.GeometryExtended2017Plan1_cff")
process.load("Configuration.Geometry.GeometryExtended2017Plan1Reco_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = '__GlobalTag__'   

process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

process.CaloTPGTranscoder.HFTPScaleShift.NCT=2;

process.HcalTPGCoderULUT.DumpL1TriggerObjects = cms.bool(True)
process.HcalTPGCoderULUT.TagName = cms.string('__LUTtag__')

CONDDIR="__CONDDIR__"

process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')
process.es_ascii = cms.ESSource('HcalTextCalibrations',
   input = cms.VPSet(
      cms.PSet(
         object = cms.string('ChannelQuality'),
         file   = cms.FileInPath(CONDDIR+'/ChannelQuality/DumpChannelQuality_Run__RUN__.txt')
      ),
      cms.PSet(
         object = cms.string('Pedestals'),
	 file   = cms.FileInPath(CONDDIR+'/Pedestals/DumpPedestals_Run__RUN__.txt')
      ),	
      cms.PSet(
         object = cms.string('Gains'),
	 file   = cms.FileInPath(CONDDIR+'/Gains/DumpGains_Run__RUN__.txt')
      ),
      cms.PSet(
         object = cms.string('RespCorrs'),
	 file   = cms.FileInPath(CONDDIR+'/RespCorrs/DumpRespCorrs_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('ElectronicsMap'),
	file   = cms.FileInPath(CONDDIR+'/ElectronicsMap/DumpElectronicsMap_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('TPParameters'),
	file   = cms.FileInPath(CONDDIR+'/TPParameters/DumpTPParameters_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('TPChannelParameters'),
	file   = cms.FileInPath(CONDDIR+'/TPChannelParameters/DumpTPChannelParameters_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('LUTCorrs'),
	file   = cms.FileInPath(CONDDIR+'/LUTCorrs/DumpLUTCorrs_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('QIEData'),
	file   = cms.FileInPath(CONDDIR+'/QIEData/DumpQIEData_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('QIETypes'),
	file   = cms.FileInPath(CONDDIR+'/QIETypes/DumpQIETypes_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('LutMetadata'),
	file   = cms.FileInPath(CONDDIR+'/LutMetadata/DumpLutMetadata_Run__RUN__.txt')
      ),
   )
)

process.generateLuts = cms.EDAnalyzer("HcalLutGenerator",
   tag = cms.string('__LUTtag__'),
   HO_master_file = cms.string('__HO_master_file__'),
   status_word_to_mask = cms.uint32(0x8000)
)

process.writeL1TriggerObjectsXml = cms.EDAnalyzer('WriteL1TriggerObjetsXml',
   TagName = cms.string('L1TriggerObjects-__LUTtag__')
)

process.writeL1TriggerObjectsTxt = cms.EDAnalyzer('WriteL1TriggerObjectsTxt',
   TagName = cms.string('__LUTtag__'),
)

process.p = cms.Path(
   process.generateLuts + process.writeL1TriggerObjectsTxt
)
