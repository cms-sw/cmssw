import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALlayerclusters as _postProcessorHGCALlayerclusters
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALsimclusters as _postProcessorHGCALsimclusters
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALTracksters as _postProcessorHGCALTracksters
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALCandidates as _postProcessorHGCALCandidates 

hltPostProcessorHGCALlayerclusters = _postProcessorHGCALlayerclusters.clone(
    subDirs = cms.untracked.vstring('HLT/HGCAL/HGCalValidator/hgcalMergeLayerClusters/LCToCP_association')
)
hltPostProcessorHGCALsimclusters = _postProcessorHGCALsimclusters.clone(
    subDirs = cms.untracked.vstring(
        'HLT/HGCAL/HGCalValidator/SimClusters/hltTiclTrackstersCLUE3DHigh/',
        'HLT/HGCAL/HGCalValidator/SimClusters/hltTiclSimTracksters/',
        'HLT/HGCAL/HGCalValidator/SimClusters/hltTiclSimTracksters_fromCPs/'
    )
)

hltPostProcessorHGCALTracksters = _postProcessorHGCALTracksters.clone(
    subDirs = cms.untracked.vstring(
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters/TSbyHits',
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters/TSbyHits_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters/TSbyLCs',
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters/TSbyLCs_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters_fromCPs/TSbyHits'
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters_fromCPs/TSbyHits_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters_fromCPs/TSbyLCs',
        'HLT/HGCAL/HGCalValidator/hltTiclSimTracksters_fromCPs/TSbyLCs_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersCLUE3DHigh/TSbyHits',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersCLUE3DHigh/TSbyLCs',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersCLUE3DHigh/TSbyLCs_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersCLUE3DHigh/TSbyHits_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersMerge/TSbyHits',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersMerge/TSbyHits_CP',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersMerge/TSbyLCs',
        'HLT/HGCAL/HGCalValidator/hltTiclTrackstersMerge/TSbyLCs_CP',
    ),
)
hltPostProcessorHGCALCandidates = _postProcessorHGCALCandidates.clone(
    subDirs = cms.untracked.vstring(
        'HLT/HGCAL/HGCalValidator/ticlCandidates/photons',
        'HLT/HGCAL/HGCalValidator/ticlCandidates/neutral_pions',
        'HLT/HGCAL/HGCalValidator/ticlCandidates/neutral_hadrons',
        'HLT/HGCAL/HGCalValidator/ticlCandidates/electrons',
        'HLT/HGCAL/HGCalValidator/ticlCandidates/muons',
        'HLT/HGCAL/HGCalValidator/ticlCandidates/charged_hadrons'
    )
)

hltHcalValidatorPostProcessor = cms.Sequence(
    hltPostProcessorHGCALlayerclusters+
    hltPostProcessorHGCALsimclusters+
    hltPostProcessorHGCALTracksters+
    hltPostProcessorHGCALCandidates        
)
