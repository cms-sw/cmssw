import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortMuonsFromDB")

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      #fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3-v1_GEN-SIM-RECO/0009/76E35258-507F-DE11-9A21-0022192311C5.root")
      fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('distortedMuons'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(1000) ),
            #threshold = cms.untracked.string('INFO')
            threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Database for scale shift if process.distortedMuons.UseDBForMomentumScale = True
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.poolDBESSource1 = cms.ESSource("PoolDBESSource",
      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
      DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(2)
      ),
      timetype = cms.untracked.string('runnumber'),
      connect = cms.string('frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS'),
      toGet = cms.VPSet(
            cms.PSet(
                  record = cms.string('MuScleFitDBobjectRcd'),
                  tag = cms.string('MuScleFit_Scale_OctoberExercise_EWK_InnerTrack'),
                  label = cms.untracked.string('')
            )
      )
)
process.poolDBESSource2 = cms.ESSource("PoolDBESSource",
      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
      DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(2)
      ),
      timetype = cms.untracked.string('runnumber'),
      connect = cms.string('frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS'),
      toGet = cms.VPSet(
            cms.PSet(
                  record = cms.string('MuScleFitDBobjectRcd'),
                  tag = cms.string('MuScleFit_Resol_OctoberExercise_EWK_InnerTrack_WithLabel'),
                  label = cms.untracked.string('MuScleFit_Resol_OctoberExercise_EWK_InnerTrack_WithLabel')
            )
      )
)
process.poolDBESSource3 = cms.ESSource("PoolDBESSource",
      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
      DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(2)
      ),
      timetype = cms.untracked.string('runnumber'),
      connect = cms.string('frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS'),
      toGet = cms.VPSet(
            cms.PSet(
                  record = cms.string('MuScleFitDBobjectRcd'),
                  tag = cms.string('MuScleFit_Resol_OctoberExercise_SherpaIdealMC_WithLabel'),
                  label = cms.untracked.string('MuScleFit_Resol_OctoberExercise_SherpaIdealMC_WithLabel')
            )
      )
)

# Create a new "distorted" Muon collection
process.distortedMuons = cms.EDFilter("DistortedMuonProducerFromDB",
      MuonTag = cms.untracked.InputTag("muons"),

      DBScaleLabel = cms.untracked.string(''),
      DBDataResolutionLabel = cms.untracked.string('MuScleFit_Resol_OctoberExercise_EWK_InnerTrack_WithLabel'),
      DBMCResolutionLabel = cms.untracked.string('MuScleFit_Resol_OctoberExercise_SherpaIdealMC_WithLabel'),
)

### NOTE: the following WMN selectors require the presence of
### the libraries and plugins fron the ElectroWeakAnalysis/WMuNu package
### So you need to process the ElectroWeakAnalysis/WMuNu package with
### some old CMSSW versions (at least <=3_1_2, <=3_3_0_pre4)
#

# WMN fast selector (use W candidates in this example)
process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")
process.corMetWMuNus.MuonTag = cms.untracked.InputTag("distortedMuons")
process.selcorMet.MuonTag = cms.untracked.InputTag("distortedMuons")

# Output
process.load("Configuration.EventContent.EventContent_cff")
process.AODSIMEventContent.outputCommands.append('keep *_distortedMuons_*_*')
process.myEventContent = process.AODSIMEventContent
process.wmnOutput = cms.OutputModule("PoolOutputModule",
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('distortMuons')
      ),
      fileName = cms.untracked.string('selectedEvents.root')
)

# Steering the process
process.distortMuons = cms.Path(
       process.distortedMuons
      *process.selectCaloMetWMuNus
)

process.end = cms.EndPath(process.wmnOutput)
