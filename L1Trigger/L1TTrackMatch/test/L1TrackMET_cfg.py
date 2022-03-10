############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import os

############################################################
# edit options here
############################################################
L1TRK_INST ="L1TrackMET" ### if not in input DIGRAW then we make them in the above step
process = cms.Process(L1TRK_INST)

ReRunTracking = True
GTTInput = True


############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

readFiles = cms.untracked.vstring(
  '/store/relval/CMSSW_11_0_0/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v2/10000/53F7C2B0-F6CF-C544-AFF4-3EAAF66DF18B.root',
)
secFiles = cms.untracked.vstring()

process.source = cms.Source ("PoolSource",
                            fileNames = readFiles,
                            secondaryFileNames = secFiles,
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            )


process.TFileService = cms.Service("TFileService", fileName = cms.string('TrackMET_Emulation.root'), closeFileFast = cms.untracked.bool(True))

if ReRunTracking:
  process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
  producerSum = process.L1HybridTracks + process.L1HybridTracksWithAssociators
else:
  producerSum = None

if GTTInput:
  process.load('L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi')
  producerSum = producerSum + process.L1GTTInputProducer



process.load("L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TrackerEtMissEmulatorProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TkMETAnalyser_cfi")

############################################################
# Primary vertex
############################################################

process.load('L1Trigger.VertexFinder.VertexProducer_cff')
process.VertexProducer.l1TracksInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")  


producerSum += process.VertexProducer

producerName = 'VertexProducer{0}'.format("fastHisto")
producerName = producerName.replace(".","p") # legalize the name
producer = process.VertexProducer.clone()
producer.VertexReconstruction.Algorithm = cms.string("fastHisto")
process.L1TrackerEtMiss.L1VertexInputTag = cms.InputTag(producerName,"l1vertices")


setattr(process, producerName, producer)
producerSum += producer
producerSum += process.L1TrackerEtMiss

process.L1TrackerEmuEtMiss.useGTTinput = GTTInput

if GTTInput:
  process.L1TrackerEmuEtMiss.L1TrackInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted")
else:
  process.L1TrackerEmuEtMiss.L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")  

EmuproducerName = 'VertexProducer{0}'.format("fastHistoEmulation")
EmuproducerName = EmuproducerName.replace(".","p") # legalize the name
Emuproducer = process.VertexProducer.clone()
Emuproducer.VertexReconstruction.Algorithm = cms.string("fastHistoEmulation")
process.L1TrackerEmuEtMiss.L1VertexInputTag = cms.InputTag(EmuproducerName,"l1verticesEmulation")

if GTTInput:
  Emuproducer.l1TracksInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted")
else:
  Emuproducer.l1TracksInputTag =  cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")  

setattr(process, EmuproducerName, Emuproducer)
producerSum += Emuproducer
producerSum += process.L1TrackerEmuEtMiss
  
process.p = cms.Path(producerSum + process.L1TkMETAnalyser)
