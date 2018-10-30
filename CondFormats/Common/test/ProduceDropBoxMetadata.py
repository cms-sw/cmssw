# documentation: https://twiki.cern.ch/twiki/bin/view/CMS/AlCaDBPCL#Drop_box_metadata_management

import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("CondCore.CondDB.CondDB_cfi")

process.CondDB.connect = 'sqlite_file:DropBoxMetadata.db' 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(300000)
                            )

# process.PoolDBOutputService.DBParameters.messageLevel = 3
import json

def encodeJsonInString(filename):
    """This function open the json file and encodes it in a string replacing probelamtic characters"""
    thefile = open(filename)
    thejson = json.load(thefile)
    thefile.close()
    return json.JSONEncoder().encode(thejson).replace('"',"&quot;")

# beamspot by run
BeamSpotObjectsRcdByRun_prod_str = encodeJsonInString("BeamSpotObjectsRcdByRun_prod.json")
BeamSpotObjectsRcdByRun_prep_str = encodeJsonInString("BeamSpotObjectsRcdByRun_prep.json")

# beamspot by lumi
BeamSpotObjectsRcdByLumi_prod_str = encodeJsonInString("BeamSpotObjectsRcdByLumi_prod.json")
BeamSpotObjectsRcdByLumi_prep_str = encodeJsonInString("BeamSpotObjectsRcdByLumi_prep.json")

# beamspot High Perf by lumi
BeamSpotObjectsRcdHPByLumi_prod_str = encodeJsonInString("BeamSpotObjectsRcdHPbyLumi_prod.json")
BeamSpotObjectsRcdHPByLumi_prep_str = encodeJsonInString("BeamSpotObjectsRcdHPbyLumi_prep.json")

# beamspot High Perf by run
BeamSpotObjectsRcdHPByRun_prod_str = encodeJsonInString("BeamSpotObjectsRcdHPbyRun_prod.json")
BeamSpotObjectsRcdHPByRun_prep_str = encodeJsonInString("BeamSpotObjectsRcdHPbyRun_prep.json")

#SiStripBadStripRcd
SiStripBadStripRcd_prod_str = encodeJsonInString("SiStripBadStripRcd_prod.json")
SiStripBadStripRcd_prep_str = encodeJsonInString("SiStripBadStripRcd_prep.json")

#SiStripApvGainRcd
SiStripApvGainRcd_prod_str = encodeJsonInString("SiStripApvGainRcd_prod.json")
SiStripApvGainRcd_multirun_prod_str = encodeJsonInString("SiStripApvGainRcd_multirun_prod.json") 
SiStripApvGainRcdAfterAbortGap_prod_str = encodeJsonInString("SiStripApvGainRcdAfterAbortGap_prod.json") # can be removed, once 92x deployed
SiStripApvGainRcdAAG_prod_str = encodeJsonInString("SiStripApvGainRcdAAG_prod.json") # will take over
SiStripApvGainRcdAAG_multirun_prod_str = encodeJsonInString("SiStripApvGainRcdAAG_multirun_prod.json")

SiStripApvGainRcd_prep_str = encodeJsonInString("SiStripApvGainRcd_prep.json")
SiStripApvGainRcd_multirun_prep_str = encodeJsonInString("SiStripApvGainRcd_multirun_prep.json")
SiStripApvGainRcdAfterAbortGap_prep_str = encodeJsonInString("SiStripApvGainRcdAfterAbortGap_prep.json") # can be removed, once 92x deployed
SiStripApvGainRcdAAG_prep_str = encodeJsonInString("SiStripApvGainRcdAAG_prep.json") # will take over
SiStripApvGainRcdAAG_multirun_prep_str = encodeJsonInString("SiStripApvGainRcdAAG_multirun_prep.json")

#SiPixelAli
SiPixelAliRcd_prod_str = encodeJsonInString("SiPixelAliRcd_prod.json")
SiPixelAliRcd_prep_str = encodeJsonInString("SiPixelAliRcd_prep.json")

#EcalPedestalsRcd
EcalPedestalsRcd_prod_str = encodeJsonInString("EcalPedestal_prod.json")
EcalPedestalsRcd_prep_str = encodeJsonInString("EcalPedestal_prep.json")

#LumiCorrectionsRcd
LumiCorrectionsRcd_prod_str = encodeJsonInString("LumiCorrections_prod.json")
LumiCorrectionsRcd_prep_str = encodeJsonInString("LumiCorrections_prep.json")

#PixelQuality
SiPixelQualityFromDbRcd_prompt_prod_str = encodeJsonInString("SiPixelQualityFromDbRcd_prompt_prod.json")
SiPixelQualityFromDbRcd_prompt_prep_str = encodeJsonInString("SiPixelQualityFromDbRcd_prompt_prep.json")
SiPixelQualityFromDbRcd_stuckTBM_prod_str = encodeJsonInString("SiPixelQualityFromDbRcd_stuckTBM_prod.json")
SiPixelQualityFromDbRcd_stuckTBM_prep_str = encodeJsonInString("SiPixelQualityFromDbRcd_stuckTBM_prep.json")
SiPixelQualityFromDbRcd_other_prod_str = encodeJsonInString("SiPixelQualityFromDbRcd_other_prod.json")
SiPixelQualityFromDbRcd_other_prep_str = encodeJsonInString("SiPixelQualityFromDbRcd_other_prep.json")

# given a set of .json files in the current dir, ProduceDropBoxMetadata produces a sqlite containign the payload with the prod/and/prep metadata
process.mywriter = cms.EDAnalyzer("ProduceDropBoxMetadata",
                                  # set to True if you want to write out a sqlite.db translating the json's into a payload
                                  write = cms.untracked.bool(True),

                                  # toWrite holds a list of Pset's, one for each workflow you want to produce DropBoxMetadata for; you need to have 2 .json files for each PSet
                                  toWrite = cms.VPSet(cms.PSet(record              = cms.untracked.string("BeamSpotObjectsRcdByRun"), 
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(BeamSpotObjectsRcdByRun_prod_str),
                                                               prepMetaData        = cms.untracked.string(BeamSpotObjectsRcdByRun_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdByLumi'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(BeamSpotObjectsRcdByLumi_prod_str),
                                                               prepMetaData        = cms.untracked.string(BeamSpotObjectsRcdByLumi_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdHPByLumi'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(BeamSpotObjectsRcdHPByLumi_prod_str),
                                                               prepMetaData        = cms.untracked.string(BeamSpotObjectsRcdHPByLumi_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdHPByRun'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(BeamSpotObjectsRcdHPByRun_prod_str),
                                                               prepMetaData        = cms.untracked.string(BeamSpotObjectsRcdHPByRun_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiStripBadStripRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiStripBadStripRcd_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiStripBadStripRcd_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiStripApvGainRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiStripApvGainRcd_prod_str),
                                                               prodMetaDataMultiRun = cms.untracked.string(SiStripApvGainRcd_multirun_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiStripApvGainRcd_prep_str),
                                                               prepMetaDataMultiRun = cms.untracked.string(SiStripApvGainRcd_multirun_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('TrackerAlignmentRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiPixelAliRcd_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiPixelAliRcd_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiStripApvGainRcdAfterAbortGap'), # can be removed, once 92x deployed...
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiStripApvGainRcdAfterAbortGap_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiStripApvGainRcdAfterAbortGap_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiStripApvGainRcdAAG'), # ... will take over
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiStripApvGainRcdAAG_prod_str),
                                                               prodMetaDataMultiRun = cms.untracked.string(SiStripApvGainRcdAAG_multirun_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiStripApvGainRcdAAG_prep_str),
                                                               prepMetaDataMultiRun = cms.untracked.string(SiStripApvGainRcdAAG_multirun_prep_str)
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('EcalPedestalsRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(EcalPedestalsRcd_prod_str),
                                                               prepMetaData        = cms.untracked.string(EcalPedestalsRcd_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('LumiCorrectionsRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(LumiCorrectionsRcd_prod_str),
                                                               prepMetaData        = cms.untracked.string(LumiCorrectionsRcd_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiPixelQualityFromDbRcd_prompt'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiPixelQualityFromDbRcd_prompt_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiPixelQualityFromDbRcd_prompt_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiPixelQualityFromDbRcd_stuckTBM'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiPixelQualityFromDbRcd_stuckTBM_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiPixelQualityFromDbRcd_stuckTBM_prep_str),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiPixelQualityFromDbRcd_other'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiPixelQualityFromDbRcd_other_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiPixelQualityFromDbRcd_other_prep_str),
                                                               ),
                                                      ),
                                  # this boolean will read the content of whichever payload is available and print its content to stoutput
                                  # set this to false if you write out a sqlite.db translating the json's into a payload
                                  read = cms.untracked.bool(False),

                                  # toRead lists of record naemes to be sought inside the DropBoxMetadataRcd payload avaialble to the ProduceDropBoxMetadata; for instance, if write is True, you're reading back the metadata you've just entered in the payload from the .json files
                                  toRead = cms.untracked.vstring('BeamSpotObjectsRcdByRun','BeamSpotObjectsRcdByLumi','BeamSpotObjectsRcdHPByLumi','BeamSpotObjectsRcdHPByRun','SiStripBadStripRcd','SiStripApvGainRcd','TrackerAlignmentRcd','SiStripApvGainRcdAfterAbortGap','SiStripApvGainRcdAAG','EcalPedestalsRcd',"LumiCorrectionsRcd","SiPixelQualityFromDbRcd_prompt","SiPixelQualityFromDbRcd_stuckTBM","SiPixelQualityFromDbRcd_other") # same strings as fType
                                  )

process.p = cms.Path(process.mywriter)

if process.mywriter.write:

    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = "sqlite_file:DropBoxMetadata.db"

    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                              CondDB,
                                              toPut = cms.VPSet(cms.PSet(record = cms.string('DropBoxMetadataRcd'),
                                                                         tag = cms.string('DropBoxMetadata'),
                                                                         timetype   = cms.untracked.string('runnumber')
                                                                         )
                                                                ),
                                              loadBlobStreamer = cms.untracked.bool(False),
                                              #    timetype   = cms.untracked.string('lumiid')
                                              #    timetype   = cms.untracked.string('runnumber')
                                              )
    
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '92X_dataRun2_Express_Queue'

#process.GlobalTag.connect   = 'frontier://PromptProd/CMS_CONDITIONS'
#process.GlobalTag.connect   = 'sqlite_file:/data/franzoni/92x/CMSSW_9_2_X_2017-05-16-1100_metadata/src/CondFormats/Common/test/last-iov-DropBoxMetadata_v5.1_express.db'

# Set to True if you want to read a DropBoxMetadata payload from a local sqlite
# specify the name of the sqlitefile.db and the tag name; the payload loaded will be for run 300000
readsqlite = False
if readsqlite:
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("DropBoxMetadataRcd"),
                 tag = cms.string("DropBoxMetadata"),
                 connect = cms.string("sqlite_file:DropBoxMetadata_addHPbyRun.db")
                )
        )
