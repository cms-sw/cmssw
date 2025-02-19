import FWCore.ParameterSet.Config as cms

process = cms.Process("MCAcceptance")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)




process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring()
)
import os
dirname = "/data1/home/degruttola/CMSSW_3_3_5/src/ElectroWeakAnalysis/Utilities/test/DistMuonsv2/res/"
dirlist = os.listdir(dirname)
basenamelist = os.listdir(dirname + "/")
for basename in basenamelist:
            process.source.fileNames.append("file:" + dirname + "/" + basename)
print "Number of files to process is %s" % (len(process.source.fileNames))




## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## "file:/data1/home/degruttola/CMSSW_3_3_5/src/ElectroWeakAnalysis/Utilities/test/DistMuonsv2/res/selectedEvents_1818.root",
## "file:~/Zmumu7TeVGenSimReco/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
##     "file:~/Zmumu7TeVGenSimReco/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
## "file:~/Zmumu7TeVGenSimReco/38980FEC-C182-DE11-A3B5-003048D4767C.root",
##  "file:~/Zmumu7TeVGenSimReco/3AF703B9-AE82-DE11-9656-0015172C0925.root",
## "file:~/Zmumu7TeVGenSimReco/46854F8E-BC82-DE11-80AA-003048D47673.root",
##  "file:~/Zmumu7TeVGenSimReco/8025F9B0-AC82-DE11-8C28-0015172560C6.root",
##  "file:~/Zmumu7TeVGenSimReco/88DDF58E-BC82-DE11-ADD8-003048D47679.root",
##  "file:~/Zmumu7TeVGenSimReco/9A115324-BB82-DE11-9C66-001517252130.root",
## "file:~/Zmumu7TeVGenSimReco/FC279CAC-AD82-DE11-BAAA-001517357D36.root"
## )
## )

process.evtInfo = cms.OutputModule("AsciiOutputModule")






process.zToMuMuMC = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("genParticles"),
    cut = cms.string('pdgId = 23 & status = 3 & abs(daughter(0).pdgId) = 13')
)


process.dimuons = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('distortedMuons@+ distortedMuons@-')
)



process.load("PhysicsTools.HepMCCandAlgos.goodMuonMCMatch_cfi")
#goodMuonMCMatch.src = 'selectedLayer1Muons'
process.goodMuonMCMatch.src = 'distortedMuons'


process.dimuonsMCMatch = cms.EDFilter("MCTruthCompositeMatcherNew",
    src = cms.InputTag("dimuons"),
    #
    # comment PAT match because works only for layer-0 muons  
    #
    #  VInputTag matchMaps = { muonMatch }
    matchPDGId = cms.vint32(),
    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"))
)




process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")



process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.poolDBESSource1 = cms.ESSource("PoolDBESSource",
      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
      DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(2),
            authenticationPath = cms.untracked.string('.')
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
            messageLevel = cms.untracked.int32(2),
            authenticationPath = cms.untracked.string('.')
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
            messageLevel = cms.untracked.int32(2),
            authenticationPath = cms.untracked.string('.')
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







process.mcAcceptance = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaMin = cms.double(0.0),
    etaMax = cms.double(2.1),
    ptMin = cms.double(20.0),
# parameter for denominator
   massMinZMC = cms.double(60.0)        
)

process.mcPath = cms.Path(
    process.zToMuMuMC*
#    process.distortedMuons *
    process.goodMuonMCMatch *
    process.dimuons *
    process.dimuonsMCMatch* 
    process.mcAcceptance
    )

from Configuration.EventContent.EventContent_cff import *

process.EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_dimuons_*_*',
        'keep *_muons_*_*',
        'keep *_genParticles_*_*',
        'keep *_muonMatch_*_*', 
        'keep *_trackMuMatch_*_*', 
        'keep *_allDimuonsMCMatch_*_*',
        'keep *_distortedMuons_*_*'
#        'keep patTriggerObjects_patTrigger_*_*',
#        'keep patTriggerFilters_patTrigger_*_*',
#        'keep patTriggerPaths_patTrigger_*_*',
#        'keep patTriggerEvent_patTriggerEvent_*_*',
#        'keep patTriggerObjectsedmAssociation_patTriggerEvent_*_*'
        )
)

AODSIMDimuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMDimuonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMDimuonEventContent.outputCommands.extend(process.EventContent.outputCommands)

process.dimuonsOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMDimuonEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('acceptance'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('dimuons_testDistMuon.root')
)




process.end = cms.EndPath(process.dimuonsOutputModule)


