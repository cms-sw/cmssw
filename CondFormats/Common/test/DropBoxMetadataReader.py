# documentation: https://twiki.cern.ch/twiki/bin/view/CMS/AlCaDBPCL#Drop_box_metadata_management

import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("CondCore.CondDB.CondDB_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.CondDB.connect = 'sqlite_file:DropBoxMetadata.db' 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(300000)
                            )


# given a set of .json files in the current dir, ProduceDropBoxMetadata produces a sqlite containign the payload with the prod/and/prep metadata
process.myReader = cms.EDAnalyzer("ProduceDropBoxMetadata",
                                  # set to True if you want to write out a sqlite.db translating the json's into a payload
                                  write = cms.untracked.bool(False),
                                  toWrite = cms.VPSet(),

                                  # this boolean will read the content of whichever payload is available and print its content to stoutput
                                  # set this to false if you write out a sqlite.db translating the json's into a payload
                                  read = cms.untracked.bool(True),
                                  # toRead lists of record names to be sought inside the DropBoxMetadataRcd payload avaialble to the ProduceDropBoxMetadata;
                                  # for instance, if write is True, you're reading back the metadata you've just entered in the payload from the .json files
                                  toRead = cms.untracked.vstring(
                                    'BeamSpotObjectsRcdByRun',
                                    'BeamSpotObjectsRcdByLumi',
                                    'BeamSpotObjectsRcdHPByLumi',
                                    'BeamSpotObjectsRcdHPByRun',
                                    'SiStripBadStripRcd',
                                    'SiStripBadStripFromHitEffRcd',
                                    'SiStripApvGainRcd',
                                    'TrackerAlignmentRcd',
                                    'TrackerAlignmentHGRcd',
                                    'SiStripApvGainRcdAfterAbortGap',
                                    'SiStripApvGainRcdAAG',
                                    'EcalPedestalsRcd',
                                    "LumiCorrectionsRcd",
                                    "SiPixelQualityFromDbRcd_prompt",
                                    "SiPixelQualityFromDbRcd_stuckTBM",
                                    "SiPixelQualityFromDbRcd_other",
                                    "SiPixelLorentzAngleRcd",
                                    "CTPPSRPAlignmentCorrectionsDataRcd",
                                    "PPSTimingCalibrationRcd",
                                    "PPSTimingCalibrationRcd_Sampic"
                                    ) # same strings as fType
                                  )

process.p = cms.Path(process.myReader)

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '124X_dataRun3_Express_Queue'

# Set to True if you want to read a DropBoxMetadata payload from a local sqlite
# specify the name of the sqlitefile.db and the tag name; the payload loaded will be for run 300000
readsqlite = True 
if readsqlite:
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("DropBoxMetadataRcd"),
                 tag = cms.string("DropBoxMetadata"),
                 connect = cms.string("sqlite_file:DropBoxMetadata.db")
                )
        )
