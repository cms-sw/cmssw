import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os 

process = cms.Process("TEST")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_38Y_V8::All'

##################################################################################

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')

# setup any defaults you want
options.output = 'test_out.root'
options.files = [
    '/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-RECO/MC_38Y_V8-v1/0013/42AFD8A5-C9A3-DF11-9F6B-001A92811706.root',
    '/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-RECO/MC_38Y_V8-v1/0013/2225585D-BFA3-DF11-8771-003048678FD6.root' ]
options.maxEvents = 1 

# get and parse the command line arguments
options.parseArguments()


##################################################################################
# Some Services

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = ['*']  
process.MessageLogger.categories = ['HeavyIonVertexing','heavyIonHLTVertexing','MuonTrackingRegionBuilder','MinBiasTracking']
process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    HeavyIonVertexing = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
	),
    heavyIonHLTVertexing = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
    MuonTrackingRegionBuilder = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    )
)
	   
process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.Timing = cms.Service("Timing")

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

##################################################################################
# Input Source
process.source = cms.Source('PoolSource',fileNames = cms.untracked.vstring(options.files))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

# GenFilter for opposite-sign status=1 muons from the embedded signal within the acceptance
process.mumugenfilter = cms.EDFilter("MCParticlePairFilter",
    moduleLabel = cms.untracked.string("hiSignal"),                               
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)

# Reconstruction			
process.load("Configuration.StandardSequences.RawToDigi_cff")		    # RawToDigi
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") # full heavy ion reconstruction
process.load("RecoHI.HiTracking.secondStep_cff")                            # pair-seeding extension

### re-run tracking only seeded by stand-alone muons
process.goodStaMuons = cms.EDFilter("TrackSelector",
                                 src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
                                 cut = cms.string("pt > 5.0"),
                                 filter = cms.bool(True)
                                 )

process.HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc="goodStaMuons"
#using modified MuonTrackingRegionBuilder.cc to pass (x,y) vertex info
process.HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion=True
process.HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaR=0.1
process.HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Dz_min=0.2
process.HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.PtMin_max=5.0
process.hiPixel3PrimTracks.RegionFactoryPSet = process.HiTrackingRegionFactoryFromSTAMuonsBlock
process.hiNewSeedFromPairs.RegionFactoryPSet = process.HiTrackingRegionFactoryFromSTAMuonsBlock

process.hiTracksWithLooseQuality.keepAllTracks=True

process.hiNewTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
process.ckfBaseTrajectoryFilter.filterPset.minimumNumberOfHits=10  # was 6

### open up trajectory builder parameters
process.MaterialPropagator.Mass = 0.105 #muon (HI default is pion)
process.OppositeMaterialPropagator.Mass = 0.105
process.ckfBaseTrajectoryFilter.filterPset.maxLostHits=1          # was 1
process.ckfBaseTrajectoryFilter.filterPset.maxConsecLostHits=1    # was 1
process.CkfTrajectoryBuilder.maxCand = 5                          # was 5

# Output EDM File
process.load("Configuration.EventContent.EventContentHeavyIons_cff")        #load keep/drop output commands
process.output = cms.OutputModule("PoolOutputModule",
                                  process.FEVTDEBUGEventContent,
                                  fileName = cms.untracked.string(options.output),
                                  SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filter_step'))
                                  )
process.output.outputCommands.extend(["keep *_*_*_TEST"])

##################################################################################
# Sequences
process.rechits = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits)
process.rerecomuons = cms.Sequence(process.goodStaMuons * process.rechits * process.heavyIonTracking)

# Paths
process.filter_step = cms.Path(process.mumugenfilter)

process.path = cms.Path(process.mumugenfilter
                        * process.rerecomuons # triplet-seeded regional step
                        * process.secondStep # pair-seeded regional step
                        )

process.save = cms.EndPath(process.output)

# Schedule
process.schedule = cms.Schedule(process.filter_step, process.path, process.save)
