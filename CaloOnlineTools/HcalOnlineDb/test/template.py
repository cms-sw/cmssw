import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process("TEST", eras.Run2_2018)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('LUT')

process.source = cms.Source("EmptySource")
process.source.firstRun = cms.untracked.uint32( __RUN__ )
process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1)
)

process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = '__GlobalTag__'   

process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

process.HcalTPGCoderULUT.DumpL1TriggerObjects = cms.bool(True)
process.HcalTPGCoderULUT.TagName = cms.string('__LUTtag__')

CONDDIR="__CONDDIR__"

process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')
process.es_ascii = cms.ESSource('HcalTextCalibrations',
   input = cms.VPSet(
      cms.PSet(
         object = cms.string('ChannelQuality'),
         file   = cms.FileInPath(CONDDIR+'/ChannelQuality/ChannelQuality_Run__RUN__.txt')
      ),
      cms.PSet(
         object = cms.string('Pedestals'),
	 file   = cms.FileInPath(CONDDIR+'/Pedestals/Pedestals_Run__RUN__.txt')
      ),	
      cms.PSet(
         object = cms.string('Gains'),
	 file   = cms.FileInPath(CONDDIR+'/Gains/Gains_Run__RUN__.txt')
      ),
      cms.PSet(
         object = cms.string('RespCorrs'),
	 file   = cms.FileInPath(CONDDIR+'/RespCorrs/RespCorrs_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('ElectronicsMap'),
	file   = cms.FileInPath(CONDDIR+'/ElectronicsMap/ElectronicsMap_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('TPParameters'),
	file   = cms.FileInPath(CONDDIR+'/TPParameters/TPParameters_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('TPChannelParameters'),
	file   = cms.FileInPath(CONDDIR+'/TPChannelParameters/TPChannelParameters_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('LUTCorrs'),
	file   = cms.FileInPath(CONDDIR+'/LUTCorrs/LUTCorrs_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('QIEData'),
	file   = cms.FileInPath(CONDDIR+'/QIEData/QIEData_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('QIETypes'),
	file   = cms.FileInPath(CONDDIR+'/QIETypes/QIETypes_Run__RUN__.txt')
      ),
      cms.PSet(
        object = cms.string('LutMetadata'),
	file   = cms.FileInPath(CONDDIR+'/LutMetadata/LutMetadata_Run__RUN__.txt')
      ),
   )
)

process.generateLuts = cms.EDAnalyzer("HcalLutGenerator",
   tag = cms.string('__LUTtag__'),
   HO_master_file = cms.string('__HO_master_file__'),
   status_word_to_mask = cms.uint32(0x8000)
)

process.writeL1TriggerObjectsTxt = cms.EDAnalyzer('WriteL1TriggerObjectsTxt',
   TagName = cms.string('__LUTtag__'),
)

process.p = cms.Path(
   process.generateLuts + process.writeL1TriggerObjectsTxt
)
