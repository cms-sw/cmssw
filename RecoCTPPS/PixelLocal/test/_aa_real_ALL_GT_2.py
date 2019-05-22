# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --filein=file:GluGluTo2Jets_M_300_2000_13TeV_exhume_cff_py_GEN_SIM_HECTOR_CTPPS.root --conditions auto:run2_mc -s DIGI:pdigi_valid,DIGI2RAW --datatier GEN-SIM-DIGI-RAW -n 10 --era Run2_2016 --eventcontent FEVTDEBUG --no_exec
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('CTPPS2',Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
############### using only CTPPS geometry 
##process.load("Configuration.Geometry.geometry_CTPPS_2018_cfi")
##process.load("Geometry.VeryForwardGeometry.geometryRP_2017_cfi")
#process.load("CondFormats.CTPPSReadoutObjects.CTPPSPixelDAQMappingESSourceXML_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cclu_info'),
    cclu_info = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
        '/store/data/Run2017E/ZeroBias/AOD/PromptReco-v1/000/304/292/00000/00BD7574-B5AA-E711-A4C0-02163E01A24F.root'
#        'file:/afs/cern.ch/user/f/fabferro/WORKSPACE/public/geo_and_simu/test/CMSSW_10_0_0_pre1/src/00BD7574-B5AA-E711-A4C0-02163E01A24F.root'
),
duplicateCheckMode = cms.untracked.string("checkEachFile")
)



process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)



#from Geometry.VeryForwardGeometry.geometryRP_cfi import *
from RecoCTPPS.Configuration.recoCTPPS_DB_cff import *
process.load("RecoCTPPS.Configuration.recoCTPPS_DB_cff")
# Output definition

process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('drop *',
                                               'keep CTPPS*_*_*_*',
                                               'keep Totem*_*_*_*',
                                               'keep TOTEM*_*_*_*'
    
),
        fileName = cms.untracked.string('simevent_CTPPS_CLU__REC_TRA_DB_mem_ALL.root')
        )


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '101X_dataRun2_Candidate_2018_02_12_11_32_57', '')


#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '100X_mcRun2_design_Candidate_2018_02_08_16_02_53', '')
#process.GlobalTag.toGet = cms.VPSet(
#    cms.PSet(
#        record = cms.string('CTPPSPixelDAQMappingRcd'),
#        tag = cms.string("PixelDAQMapping"),
#        label = cms.untracked.string("RPix"),
#        connect = cms.string("sqlite_file:./CTPPSPixel_Mapping_Mask_Gain_Jan23rd2018.db")
#        ),
#    cms.PSet(
#        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
#        tag = cms.string("PixelAnalysisMask"),
#        label = cms.untracked.string(""),
#        connect = cms.string("sqlite_file:./CTPPSPixel_Mapping_Mask_Gain_Jan23rd2018.db")
#        ),
#    cms.PSet(
#        record = cms.string('CTPPSPixelGainCalibrationsRcd'),
#        tag = cms.string("CTPPSPixelGainCalibNew_v4"),
#        label = cms.untracked.string(""),
#        connect = cms.string("sqlite_file:./CTPPSPixel_Mapping_Mask_Gain_Jan23rd2018.db")
#        )
##    cms.PSet(record = cms.string('GeometryFileRcd'),
##             tag = cms.string('XMLFILE_Geometry_TagXX_Extended2017_mc'),
##             connect = cms.string("sqlite_file:myfile.db"),
##             )
#    )

#from CondCore.CondDB.CondDB_cfi import *
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#                                      CondDB.clone(
#        connect = cms.string('sqlite_file:CTPPSPixel_DAQMapping_Jan23rd2018_3.db')),
#                                      toGet = cms.VPSet(
#        cms.PSet(
#            record = cms.string('CTPPSPixelDAQMappingRcd'),
#            tag = cms.string("PixelDAQMapping"),
##            label = cms.untracked.string("RPix")
#            ),
#        cms.PSet(
#            record = cms.string('CTPPSPixelAnalysisMaskRcd'),
#            tag = cms.string("PixelAnalysisMask"),
##            label = cms.untracked.string("RPix")
#            ),
#               cms.PSet(
#            record = cms.string('CTPPSPixelGainCalibrationsRcd'),
#            tag = cms.string("CTPPSPixelGainCalibNew_v4"),
#            )
#        )
#                                 )


# Path and EndPath definitions

process.reco_step = cms.Path(process.recoCTPPS)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.outpath = cms.EndPath(process.o1)


# Schedule definition
process.schedule = cms.Schedule(
#process.mixedigi_step,
#process.digi2raw_step,
#process.raw2digi_step,
process.reco_step,process.endjob_step,process.outpath)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
