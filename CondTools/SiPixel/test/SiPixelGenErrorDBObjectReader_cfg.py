import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("SiPixelGenErrorDBObjectReaderTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
# process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

testGlobalTag = False
if testGlobalTag :
#old DB, to be removed soon
#    process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#    process.GlobalTag.globaltag = "GR_R_72_V5::All"
#    process.GlobalTag.globaltag = "POSTLS172_V9::All"
#    process.GlobalTag.globaltag = "DESIGN72_V5::All"
#    process.GlobalTag.globaltag = "MC_72_V3::All"
#    process.GlobalTag.globaltag = "START72_V3::All"

#use GTs without ::All with the next line
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    from Configuration.AlCa.autoCond import autoCond
    #use autocond, see:
    #https://github.com/cms-sw/cmssw/blob/CMSSW_7_3_X/Configuration/AlCa/python/autoCond.py
    #process.GlobalTag.globaltag = autoCond['run2_data']
    process.GlobalTag.globaltag = autoCond['run2_mc']
    #or set GT by hand
    process.GlobalTag.globaltag = "GR_R_72_V5"

    
# for local sqlite files
else:
    process.PoolDBESSource = cms.ESSource("PoolDBESSource",
        process.CondDBSetup,
        toGet = cms.VPSet(
          cms.PSet(
            record = cms.string('SiPixelGenErrorDBObjectRcd'),
#            tag = cms.string('SiPixelGenErrorDBObject38TV10')
            tag = cms.string('SiPixelGenErrorDBObject38Tv1')
          )),
        timetype = cms.string('runnumber'),
        #connect = cms.string('sqlite_file:../../../../../DB/siPixelGenErrors38T_v1_mc.db')
        #connect = cms.string('sqlite_file:../../../../../DB/siPixelGenErrors38T_2012_IOV7_v1.db')
        connect = cms.string('sqlite_file:siPixelGenErrors38Tv1.db')
    )
    process.PoolDBESSource.DBParameters.authenticationPath='.'
    process.PoolDBESSource.DBParameters.messageLevel=0


process.reader = cms.EDAnalyzer("SiPixelGenErrorDBObjectReader",
#                     siPixelGenErrorCalibrationLocation = cms.string("./"),
                     siPixelGenErrorCalibrationLocation = cms.string(""),
#Change to True if you would like a more detailed error output
#wantDetailedOutput = False
#Change to True if you would like to output the full GenError database object
#wantFullOutput = False
                     wantDetailedGenErrorDBErrorOutput = cms.bool(True),
                     wantFullGenErrorDBOutput = cms.bool(True)
                 )

process.p = cms.Path(process.reader)






