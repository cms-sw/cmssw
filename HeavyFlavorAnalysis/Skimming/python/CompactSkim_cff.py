import FWCore.ParameterSet.Config as cms

def CompactSkim(process,inFileNames,outFileName,Global_Tag='auto:run2_mc',MC=True,Filter=True):

   process.load('Configuration.StandardSequences.Services_cff')
   process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
   process.load('FWCore.MessageService.MessageLogger_cfi')
   process.load('Configuration.EventContent.EventContent_cff')
   process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
   process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
   process.load('Configuration.StandardSequences.EndOfProcess_cff')
   process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

   process.MessageLogger.cerr.FwkReport.reportEvery = 100
   process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
   process.options.allowUnscheduled = cms.untracked.bool(True)
   process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(inFileNames))
   process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

   from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
   process.GlobalTag = GlobalTag(process.GlobalTag, Global_Tag, '')

   # make patCandidates, select and clean them
   process.load('PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff')
   process.load('PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff')
   process.load('PhysicsTools.PatAlgos.cleaningLayer1.cleanPatCandidates_cff')
   process.patMuons.embedTrack  = True

   process.selectedPatMuons.cut = cms.string('muonID(\"TMOneStationTight\")'
                    ' && abs(innerTrack.dxy) < 0.3'
                    ' && abs(innerTrack.dz)  < 20.'
                    ' && innerTrack.hitPattern.trackerLayersWithMeasurement > 5'
                    ' && innerTrack.hitPattern.pixelLayersWithMeasurement > 0'
                    ' && innerTrack.quality(\"highPurity\")'
                    )

   #make patTracks
   from PhysicsTools.PatAlgos.tools.trackTools import makeTrackCandidates
   makeTrackCandidates(process,
                       label        = 'TrackCands',                  # output collection
                       tracks       = cms.InputTag('generalTracks'), # input track collection
                       particleType = 'pi+',                         # particle type (for assigning a mass)
                       preselection = 'pt > 0.7',                    # preselection cut on candidates
                       selection    = 'pt > 0.7',                    # selection on PAT Layer 1 objects
                       isolation    = {},                            # isolations to use (set to {} for None)
                       isoDeposits  = [],
                       mcAs         = None                           # replicate MC match as the one used for Muons
   )
   process.patTrackCands.embedTrack = True

   # dimuon = Onia2MUMU
   process.load('HeavyFlavorAnalysis.Onia2MuMu.onia2MuMuPAT_cfi')
   process.onia2MuMuPAT.muons=cms.InputTag('cleanPatMuons')
   process.onia2MuMuPAT.primaryVertexTag=cms.InputTag('offlinePrimaryVertices')
   process.onia2MuMuPAT.beamSpotTag=cms.InputTag('offlineBeamSpot')

   process.onia2MuMuPATCounter = cms.EDFilter('CandViewCountFilter',
      src = cms.InputTag('onia2MuMuPAT'),
      minNumber = cms.uint32(1),
      filter = cms.bool(True)
   )

   # reduce MC genParticles a la miniAOD
   process.load('PhysicsTools.PatAlgos.slimming.genParticles_cff')
   process.packedGenParticles.inputVertices = cms.InputTag('offlinePrimaryVertices')

   # make photon candidate conversions for P-wave studies
   process.load('HeavyFlavorAnalysis.Onia2MuMu.OniaPhotonConversionProducer_cfi')

   # Pick branches you want to keep
   SlimmedEventContent = [
                     'keep recoVertexs_offlinePrimaryVertices_*_*',
                     'keep *_inclusiveSecondaryVertices_*_*',
                     'keep *_offlineBeamSpot_*_*',
                     'keep *_TriggerResults_*_HLT',
                     'keep *_gtDigis_*_RECO',
                     'keep *_cleanPatTrackCands_*_*',
                     'keep *_PhotonCandidates_*_*',
                     'keep *_onia2MuMuPAT_*_*',
                     'keep *_generalV0Candidates_*_*',
                     'keep PileupSummaryInfos_*_*_*'
   ]

   if not MC:
      from PhysicsTools.PatAlgos.tools.coreTools import runOnData
      runOnData( process, outputModules = [] )
   else :
      SlimmedEventContent += [
                     'keep patPackedGenParticles_packedGenParticles_*_*',
                     'keep recoGenParticles_prunedGenParticles_*_*',
                     'keep GenFilterInfo_*_*_*',
                     'keep GenEventInfoProduct_generator_*_*',
                     'keep GenRunInfoProduct_*_*_*'
      ]

   process.FilterOutput = cms.Path(process.onia2MuMuPATCounter)

   process.out = cms.OutputModule('PoolOutputModule',
      fileName = cms.untracked.string(outFileName),
      outputCommands = cms.untracked.vstring('drop *', *SlimmedEventContent),
      SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('FilterOutput')) if Filter else cms.untracked.PSet()
   )
   
   process.outpath = cms.EndPath(process.out)
