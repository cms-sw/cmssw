import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.modules import PFBlockProducer
particleFlowBlock = PFBlockProducer()

for imp in particleFlowBlock.elementImporters:
  if imp.importerName.value() == "SuperClusterImporter":
    _scImporter = imp

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(_scImporter,
                                minSuperClusterPt = 1.0,
                                minPTforBypass = 0.0)

#
# kill pfTICL tracks
def _findIndicesByModule(name):
   ret = []
   for i, pset in enumerate(particleFlowBlock.elementImporters):
        if pset.importerName.value() == name:
            ret.append(i)
   return ret

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
_insertTrackImportersWithVeto = {}
_trackImporters = ['GeneralTracksImporter','ConvBremTrackImporter',
                   'ConversionTrackImporter','NuclearInteractionTrackImporter']
for importer in _trackImporters:
  for idx in _findIndicesByModule(importer):
    _insertTrackImportersWithVeto[idx] = dict(
      vetoEndcap = True,
      vetoMode = cms.uint32(2), # pfTICL candidate list
      vetoSrc = cms.InputTag("pfTICL")
    )
phase2_hgcal.toModify(
    particleFlowBlock,
    elementImporters = _insertTrackImportersWithVeto
)

#
# append track-HF linkers
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_addTrackHFLinks = particleFlowBlock.linkDefinitions.copy()
_addTrackHFLinks.append(
  cms.PSet( linkerName = cms.string("TrackAndHCALLinker"),
            linkType   = cms.string("TRACK:HFEM"),
            useKDTree  = cms.bool(True),
            trajectoryLayerEntrance = cms.string("VFcalEntrance"),
            trajectoryLayerExit = cms.string(""),
            nMaxHcalLinksPerTrack = cms.int32(-1) # Keep all track-HFEM links
          )
)
_addTrackHFLinks.append(
  cms.PSet( linkerName = cms.string("TrackAndHCALLinker"),
            linkType   = cms.string("TRACK:HFHAD"),
            useKDTree  = cms.bool(True),
            trajectoryLayerEntrance = cms.string("VFcalEntrance"),
            trajectoryLayerExit = cms.string(""),
            nMaxHcalLinksPerTrack = cms.int32(-1) # Keep all track-HFHAD links for now
          )
)
phase2_tracker.toModify(
    particleFlowBlock,
    linkDefinitions = _addTrackHFLinks
)

#
# for precision timing
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
_addTiming = particleFlowBlock.elementImporters.copy()
_addTiming.append( cms.PSet( importerName = cms.string("TrackTimingImporter"),
                             useTimeQuality = cms.bool(False),
                             timeValueMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel"),
                             timeErrorMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution"),
                             timeValueMapGsf = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModel"),
                             timeErrorMapGsf = cms.InputTag("gsfTrackTimeValueMapProducer:electronGsfTracksConfigurableFlatResolutionModelResolution")
                             )
                   )

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
_addTimingLayer = particleFlowBlock.elementImporters.copy()
_addTimingLayer.append( cms.PSet( importerName = cms.string("TrackTimingImporter"),
                             timeValueMap = cms.InputTag("tofPID:t0"),
                             timeErrorMap = cms.InputTag("tofPID:sigmat0"),
                             useTimeQuality = cms.bool(True),
                             timeQualityMap = cms.InputTag("mtdTrackQualityMVA:mtdQualMVA"),
                             timeQualityThreshold = cms.double(0.5),
                             #this will cause no time to be set for gsf tracks
                             #(since this is not available for the fullsim/reconstruction yet)
                             #*TODO* update when gsf times are available
                             timeValueMapGsf = cms.InputTag("tofPID:t0"),
                             timeErrorMapGsf = cms.InputTag("tofPID:sigmat0"),
                             timeQualityMapGsf = cms.InputTag("mtdTrackQualityMVA:mtdQualMVA"),
                             )
                   )

phase2_timing.toModify(
    particleFlowBlock,
    elementImporters = _addTiming
)

phase2_timing_layer.toModify(
    particleFlowBlock,
    elementImporters = _addTimingLayer
)

#--- Use DB conditions for cuts&seeds for Run3 and phase2
from Configuration.Eras.Modifier_hcalPfCutsFromDB_cff import hcalPfCutsFromDB
hcalPfCutsFromDB.toModify( _scImporter,
                           usePFThresholdsFromDB = True)
