import FWCore.ParameterSet.Config as cms

hltMtdTrackQualityMVA = cms.EDProducer('MTDTrackQualityMVAProducer',
    tracksSrc = cms.InputTag('hltGeneralTracks'),
    btlMatchChi2Src = cms.InputTag('hltTrackExtenderWithMTD', 'btlMatchChi2'),
    btlMatchTimeChi2Src = cms.InputTag('hltTrackExtenderWithMTD', 'btlMatchTimeChi2'),
    etlMatchChi2Src = cms.InputTag('hltTrackExtenderWithMTD', 'etlMatchChi2'),
    etlMatchTimeChi2Src = cms.InputTag('hltTrackExtenderWithMTD', 'etlMatchTimeChi2'),
    mtdTimeSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracktmtd'),
    sigmamtdTimeSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracksigmatmtd'),
    pathLengthSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackPathLength'),
    npixBarrelSrc = cms.InputTag('hltTrackExtenderWithMTD', 'npixBarrel'),
    npixEndcapSrc = cms.InputTag('hltTrackExtenderWithMTD', 'npixEndcap'),
    outermostHitPositionSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackOutermostHitPosition'),
    offlineBS = cms.InputTag('hltOnlineBeamSpot'),
    qualityBDT_weights_file = cms.FileInPath('RecoMTD/TimingIDTools/data/BDT_nvars_17_d7.xml'),
    mightGet = cms.optional.untracked.vstring
  )
