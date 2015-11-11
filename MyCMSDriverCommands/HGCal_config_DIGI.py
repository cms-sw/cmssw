# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --filein dbs:/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/TP2023HGCALGS-newsplit_DES23_62_V1-v1/GEN-SIM --fileout file:JME-TP2023SHCALDR-00001_step1.root --mc --eventcontent FEVTDEBUG --pileup AVE_140_BX_25ns --pileup_input dbs:/MinBias_TuneZ2star_14TeV-pythia6/TP2023HGCALGS-DES23_62_V1-v3/GEN-SIM --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023HGCalMuon,Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-DIGI-RAW --conditions PH2_1K_FB_V6::All --step DIGI:pdigi_valid,L1,DIGI2RAW --geometry Extended2023HGCalMuon,Extended2023HGCalMuonReco --python_filename HGCal_config_DIGI.py --no_exec -n 50
import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGI2RAW')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/00080254-D2ED-E411-9B24-E03F49D6226B.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/002E5687-CFED-E411-9699-0015C5F82B46.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/00811BED-4BEE-E411-9114-1CC1DE192872.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/064AB08B-32F1-E411-BAA2-90B11C094A7E.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/081BF597-F3ED-E411-884A-FA163E055F6A.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/082E1F05-32F1-E411-B2B6-001E688650C4.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/084B60DD-32F1-E411-99F7-001E67A40451.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/0A42BAF7-9EEF-E411-8E0C-0025901D094A.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/0C10B845-D0ED-E411-BDC0-001E6724799A.root', 
        '/store/mc/TP2023HGCALGS/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM/newsplit_DES23_62_V1-v1/00000/0CF12957-74EF-E411-AE77-782BCB63EBF5.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('step1 nevts:50'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('file:JME-TP2023SHCALDR-00001_step1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    )
)

# Additional output definition

# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(140.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
# process.mix.input.fileNames = cms.untracked.vstring([
        ### Rosma ######
        # 'file:/lustre/cms/store/user/rosma/MinBias/MinBias14TeV2023HGCalMuonME0/150705_103519/MinBias.root',
        ### Amandeep ###
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/0010AE1F-6676-E411-8F16-002618943860.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/0035CDEE-5C76-E411-8214-0023AEFDEEEC.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/004B2C7D-6876-E411-ABFA-002618943949.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/006DDC01-6276-E411-9E66-00259073E4E4.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/008F6D89-5976-E411-A05E-549F35AC7DEE.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/02133DBD-6176-E411-967A-002590A8882A.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/0253431B-4F76-E411-ABFE-0025904C66F4.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/02758CA9-5F76-E411-A1D8-0015172C07E1.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/02C7F040-7176-E411-B19E-0023AEFDEE68.root', 
        # '/store/mc/TP2023HGCALGS/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v3/00000/02DE880D-5576-E411-AE26-002590200A00.root',
# ])
puFiles = cms.untracked.vstring()
process.mix.input.fileNames = puFiles
puFiles.extend(['file:/lustre/cms/store/user/rosma/MinBias/MinBias14TeV2023HGCalMuonME0/150705_103519/MinBias.root']) ### File Rosma ... just to test

process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V6::All', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.endjob_step,process.FEVTDEBUGoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023HGCalMuon 

#call to customisation function cust_2023HGCalMuon imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023HGCalMuon(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions
process.load('SimMuon.GEMDigitizer.muonME0DigisPreReco_cfi')
process.simMuonME0Digis.timeResolution = cms.double(1.0) # ns
process.simMuonME0Digis.phiResolution = cms.double(0.01) # cm
process.simMuonME0Digis.etaResolution = cms.double(1.00) # cm
