import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:DropBoxMetadata.db'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(165200)
                            )

# process.PoolDBOutputService.DBParameters.messageLevel = 3
import json
# beamspot by run
BeamSpotObjectsRcdByRun_prod_file         = open("BeamSpotObjectsRcdByRun_prod.json")
BeamSpotObjectsRcdByRun_prod_json         = json.load(BeamSpotObjectsRcdByRun_prod_file)
BeamSpotObjectsRcdByRun_prod_file.close()
BeamSpotObjectsRcdByRun_prod_str         = json.JSONEncoder().encode(BeamSpotObjectsRcdByRun_prod_json).replace('"',"&quot;")



BeamSpotObjectsRcdByRun_prep_file         = open("BeamSpotObjectsRcdByRun_prep.json")
BeamSpotObjectsRcdByRun_prep_json         = json.load(BeamSpotObjectsRcdByRun_prep_file)
BeamSpotObjectsRcdByRun_prep_file.close()
BeamSpotObjectsRcdByRun_prep_str         = json.JSONEncoder().encode(BeamSpotObjectsRcdByRun_prep_json).replace('"',"&quot;")


# beamspot by lumi
BeamSpotObjectsRcdByLumi_prod_file         = open("BeamSpotObjectsRcdByLumi_prod.json")
BeamSpotObjectsRcdByLumi_prod_json         = json.load(BeamSpotObjectsRcdByLumi_prod_file)
BeamSpotObjectsRcdByLumi_prod_file.close()
BeamSpotObjectsRcdByLumi_prod_str         = json.JSONEncoder().encode(BeamSpotObjectsRcdByLumi_prod_json).replace('"',"&quot;")


BeamSpotObjectsRcdByLumi_prep_file         = open("BeamSpotObjectsRcdByLumi_prep.json")
BeamSpotObjectsRcdByLumi_prep_json         = json.load(BeamSpotObjectsRcdByLumi_prep_file)
BeamSpotObjectsRcdByLumi_prep_file.close()
BeamSpotObjectsRcdByLumi_prep_str         = json.JSONEncoder().encode(BeamSpotObjectsRcdByLumi_prep_json).replace('"',"&quot;")



#SiStripBadStripRcd
SiStripBadStripRcd_prod_file         = open("SiStripBadStripRcd_prod.json")
SiStripBadStripRcd_prod_json         = json.load(SiStripBadStripRcd_prod_file)
SiStripBadStripRcd_prod_file.close()
SiStripBadStripRcd_prod_str         = json.JSONEncoder().encode(SiStripBadStripRcd_prod_json).replace('"',"&quot;")



SiStripBadStripRcd_prep_file         = open("SiStripBadStripRcd_prep.json")
SiStripBadStripRcd_prep_json         = json.load(SiStripBadStripRcd_prep_file)
SiStripBadStripRcd_prep_file.close()
SiStripBadStripRcd_prep_str         = json.JSONEncoder().encode(SiStripBadStripRcd_prep_json).replace('"',"&quot;")




process.mywriter = cms.EDAnalyzer("ProduceDropBoxMetadata",
                                  write = cms.untracked.bool(False),
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
                                                      cms.PSet(record              = cms.untracked.string('SiStripBadStripRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               prodMetaData        = cms.untracked.string(SiStripBadStripRcd_prod_str),
                                                               prepMetaData        = cms.untracked.string(SiStripBadStripRcd_prep_str),
                                                               )
                                                      ),
                                  read = cms.untracked.bool(True),
                                  toRead = cms.untracked.vstring("BeamSpotObjectsRcdByRun",'BeamSpotObjectsRcdByLumi','SiStripBadStripRcd') # same strings as fType
                                  )


process.p = cms.Path(process.mywriter)

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:DropBoxMetadata.db"

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
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
process.GlobalTag.globaltag = 'GR_E_V31::All'
#process.GlobalTag.connect   = 'sqlite_file:/afs/cern.ch/user/c/cerminar/public/Alca/GlobalTag/GR_R_311_V2.db'

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("DropBoxMetadataRcd"),
             tag = cms.string("DropBoxMetadata"),
             connect = cms.untracked.string("sqlite_file:DropBoxMetadata.db")
            )
    )
