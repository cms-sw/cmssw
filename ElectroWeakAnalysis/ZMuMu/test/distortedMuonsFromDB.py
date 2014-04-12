import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortMuonsFromDB")

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(100)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      #fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3-v1_GEN-SIM-RECO/0009/76E35258-507F-DE11-9A21-0022192311C5.root")
      fileNames = cms.untracked.vstring(

"file:../../ZMuMu/test/dimuons_100.root"
                                        #rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/38980FEC-C182-DE11-A3B5-003048D4767C.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/3AF703B9-AE82-DE11-9656-0015172C0925.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/46854F8E-BC82-DE11-80AA-003048D47673.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/8025F9B0-AC82-DE11-8C28-0015172560C6.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/88DDF58E-BC82-DE11-ADD8-003048D47679.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/9A115324-BB82-DE11-9C66-001517252130.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/FC279CAC-AD82-DE11-BAAA-001517357D36.root"
)
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
            messageLevel = cms.untracked.int32(2),
            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
      ),
      timetype = cms.untracked.string('runnumber'),
      connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_PHYSICSTOOLS'),
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
            messageLevel = cms.untracked.int32(2),
            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
      ),
      timetype = cms.untracked.string('runnumber'),
      connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_PHYSICSTOOLS'),
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
            messageLevel = cms.untracked.int32(2),
            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
      ),
      timetype = cms.untracked.string('runnumber'),
      connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_PHYSICSTOOLS'),
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
#process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")
#process.corMetWMuNus.MuonTag = cms.untracked.InputTag("distortedMuons")
#process.selcorMet.MuonTag = cms.untracked.InputTag("distortedMuons")

# Output
process.load("Configuration.EventContent.EventContent_cff")
process.myEventContent = process.AODSIMEventContent
process.myEventContent.outputCommands.extend(
     cms.untracked.vstring('drop *',
                           'keep *_genParticles_*_*',
                           'keep *_muons_*_*',
                           'keep *_distortedMuons_*_*')
     )

process.Output = cms.OutputModule("PoolOutputModule",
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('distortMuons')
      ),
      fileName = cms.untracked.string('selectedEvents.root')
)





# Steering the process
process.distortMuons = cms.Path(
       process.distortedMuons
 #     *process.selectCaloMetWMuNus
)

process.end = cms.EndPath(process.Output)
