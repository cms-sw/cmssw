import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol04CfgV2

# supposed to be calculated from AOD (see https://github.com/cms-sw/cmssw/pull/42007)
# but since we didn't create the new PAT version of SC
# trk iso values are calculated on-the-fly from miniAOD
# parameters are inherited from HEEP trk iso

# instead, we use fairly loose inner cone & strip veto from EgammaHLTTrackIsolation
# to avoid strong bias due to the tight trk isolation
# e.g. see hltEgammaHollowTrackIsoL1Seeded filter

scTrkIso04 = cms.PSet(
  barrelCuts = trkIsol04CfgV2.barrelCuts.clone(
    minDR = 0.06,
    minDEta = 0.03
  ),
  endcapCuts = trkIsol04CfgV2.endcapCuts.clone(
    minDR = 0.06,
    minDEta = 0.03
  )
)

superclusValueMaps = cms.EDProducer("SuperclusValueMapProducer",
  srcBs = cms.InputTag("offlineBeamSpot"),
  srcPv = cms.InputTag("offlineSlimmedPrimaryVertices"),
  srcSc = cms.InputTag("reducedEgamma:reducedSuperClusters"),
  cands = cms.VInputTag("packedPFCandidates",
                        "lostTracks"), # do not count electron tracks in the trk iso
  candVetos = cms.vstring("ELES","NONE"),
  trkIsoConfig = scTrkIso04,
)
