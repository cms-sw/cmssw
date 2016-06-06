import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_cff import *

# muons with trigger info
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff import *
selectedMuons = cms.EDFilter('PATMuonSelector',
   src = cms.InputTag('patMuonsWithTrigger'),
   cut = cms.string('muonID(\"TMOneStationTight\")'
                    ' && abs(innerTrack.dxy) < 0.3'
                    ' && abs(innerTrack.dz)  < 20.'
                    ' && innerTrack.hitPattern.trackerLayersWithMeasurement > 5'
                    ' && innerTrack.hitPattern.pixelLayersWithMeasurement > 0'
                    ' && innerTrack.quality(\"highPurity\")'
                    ' && ((abs(eta) <= 0.9 && pt > 2.5) || (0.9 < abs(eta) <= 2.4 && pt > 1.5))'
   ),
   filter = cms.bool(True)
)

# tracks
selectedTracks=cms.EDFilter("TrackSelector",
     src = cms.InputTag("generalTracks"),
     cut = cms.string('pt > 0.7 && abs(eta) <= 3.0'
                      '&& charge !=0'
                      '&& quality(\"highPurity\")')     
)

# dimuon = Onia2MUMU
from HeavyFlavorAnalysis.Onia2MuMu.onia2MuMuPAT_cfi import *
onia2MuMuPAT.muons=cms.InputTag('selectedMuons')
onia2MuMuPAT.primaryVertexTag=cms.InputTag('offlinePrimaryVertices')
onia2MuMuPAT.beamSpotTag=cms.InputTag('offlineBeamSpot')
onia2MuMuPAT.dimuonSelection=cms.string("0.2 < mass && abs(daughter('muon1').innerTrack.dz - daughter('muon2').innerTrack.dz) < 25")
onia2MuMuPAT.addMCTruth = cms.bool(False)

onia2MuMuPATCounter = cms.EDFilter('CandViewCountFilter',
      src = cms.InputTag('onia2MuMuPAT'),
      minNumber = cms.uint32(1),
      filter = cms.bool(True)
   )

# make photon candidate conversions for P-wave studies
from HeavyFlavorAnalysis.Onia2MuMu.OniaPhotonConversionProducer_cfi import *

# add v0 with tracks embed
from HeavyFlavorAnalysis.Onia2MuMu.OniaAddV0TracksProducer_cfi import *

# Pick branches you want to keep
BPHSkim_EventContent = cms.PSet(
     outputCommands = cms.untracked.vstring(
                     'drop *',
                     'keep recoVertexs_offlinePrimaryVertices_*_*',
                     'keep *_offlineBeamSpot_*_*',
                     'keep *_TriggerResults_*_HLT',
                     'keep *_gtDigis_*_RECO',
                     'keep *_selectedTracks_*_*',
                     'keep *_PhotonCandidates_*_*',
                     'keep *_onia2MuMuPAT_*_*',
                     'keep *_oniaV0Tracks_*_*',
                     'keep PileupSummaryInfos_*_*_*'
     )
)

BPHSkimSequence = cms.Sequence(
            patMuonsWithTriggerSequence *
	    selectedMuons *
            onia2MuMuPAT *
	    onia2MuMuPATCounter *
	    PhotonCandidates *
	    oniaV0Tracks *
	    selectedTracks
)
