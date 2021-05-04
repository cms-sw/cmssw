import FWCore.ParameterSet.Config as cms

process = cms.Process("APVCyclePhaseProducerTestGR09")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.files.infos = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(10000)
    )
)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/958/C81A92C2-D788-DE11-B7BB-000423D6B42C.root', # 110958
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/FECCF15C-4872-DE11-BDB2-000423D944F8.root', # 106019
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/958/22EF7D51-CF88-DE11-AFB9-001617C3B70E.root' # 110958
#        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/616/FEF9115E-FB7F-DE11-83FA-001D09F29538.root', # 109616
#        '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/916/CCA16AD0-7588-DE11-B156-001617E30D52.root', # 110916
#        '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/417/FA8CB4E8-3395-DE11-AA3D-001D09F29321.root', # 112417
#        '/store/data/Commissioning09/Cosmics/RAW/v2/000/102/169/F6566668-4267-DE11-8354-001D09F2983F.root', # 102169
#        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/718/FAA59806-A680-DE11-A5D9-000423D99896.root', # 109718
#        '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/650/FAD443CE-F296-DE11-A25C-000423D6CA72.root' # 112650
        

    ),
                            skipBadFiles = cms.untracked.bool(True)
                            )

#---------------------------------------------------------------------
# Raw to Digi: TO BE TESTED !!
#---------------------------------------------------------------------
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V5P::All"


#-------------------------------------------------------------------------

process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_withdefault_cff")
#process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_CRAFT08_cfi")

import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1abc_GR09_cfi
process.APVPhasesFromL1ABC = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1abc_GR09_cfi.APVPhases.clone()
process.APVPhasesFromL1ABC.wantHistos = cms.untracked.bool(True)

process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")

process.apvcyclephasemonitorfroml1abc = process.apvcyclephasemonitor.clone()
process.apvcyclephasemonitorfroml1abc.apvCyclePhaseCollection = cms.InputTag("APVPhasesFromL1ABC") 

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('apvcyclephaseproducer_test_GR09.root')
                                   )

process.p0 = cms.Path(process.scalersRawToDigi +
                      process.APVPhases + process.APVPhasesFromL1ABC + 
                      process.apvcyclephasemonitor +process.apvcyclephasemonitorfroml1abc )

