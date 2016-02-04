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

process.mywriter = cms.EDAnalyzer("ProduceDropBoxMetadata",
                                  write = cms.untracked.bool(True),
                                  toWrite = cms.VPSet(cms.PSet(record              = cms.untracked.string("BeamSpotObjectsRcdByRun"), 
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               destDB              = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT"),
                                                               destDBValidation    = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"),
                                                               tag                 = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_offline"),
                                                               Timetype            = cms.untracked.string("runnumber"),
                                                               IOVCheck            = cms.untracked.string("All"),
                                                               DuplicateTagHLT     = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_hlt"),
                                                               DuplicateTagEXPRESS = cms.untracked.string(""),
                                                               DuplicateTagPROMPT  = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_prompt"),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdByLumi'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               destDB              = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT"),
                                                               destDBValidation    = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"),
                                                               tag                 = cms.untracked.string("BeamSpotObjects_PCL_byLumi_v0_prompt"),
                                                               Timetype            = cms.untracked.string("lumiid"),
                                                               IOVCheck            = cms.untracked.string("prompt"),
                                                               DuplicateTagHLT     = cms.untracked.string(""),
                                                               DuplicateTagEXPRESS = cms.untracked.string(""),
                                                               DuplicateTagPROMPT  = cms.untracked.string(""),
                                                               ),
                                                      cms.PSet(record              = cms.untracked.string('SiStripBadStripRcd'),
                                                               Source              = cms.untracked.string("AlcaHarvesting"),
                                                               FileClass           = cms.untracked.string("ALCA"),
                                                               destDB              = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_STRIP"),
                                                               destDBValidation    = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_STRIP"),
                                                               tag                 = cms.untracked.string("SiStripBadChannel_PCL_v0_offline"),
                                                               Timetype            = cms.untracked.string("runnumber"),
                                                               IOVCheck            = cms.untracked.string("All"),
                                                               DuplicateTagHLT     = cms.untracked.string("SiStripBadChannel_PCL_v0_hlt"),
                                                               DuplicateTagEXPRESS = cms.untracked.string(""),
                                                               DuplicateTagPROMPT  = cms.untracked.string("SiStripBadChannel_PCL_v0_prompt"),
                                                               )
                                                      ),
                                  read = cms.untracked.bool(False),
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
process.GlobalTag.globaltag = 'GR_E_V19::All'
#process.GlobalTag.connect   = 'sqlite_file:/afs/cern.ch/user/c/cerminar/public/Alca/GlobalTag/GR_R_311_V2.db'

# process.GlobalTag.toGet = cms.VPSet(
#   cms.PSet(record = cms.string("DropBoxMetadataRcd"),
#            tag = cms.string("DropBoxMetadata_v0_express"),
#            connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_RUN_INFO")
#           )
# )
