import FWCore.ParameterSet.Config as cms

hltPFTauPairLeadTrackDzMatchFilter = cms.EDFilter(
    "HLTPFTauPairLeadTrackDzMatchFilter",

    # Tau collection
    tauSrc = cms.InputTag('hltPFTaus'),

    # Main cut
    # max dZ distance at PCA between leading tracks of two taus 
    tauLeadTrackMaxDZ = cms.double(0.2),
    
    ## Tau preselection
    # min Pt of Tau
    tauMinPt = cms.double(0.0),
    # max eta of Tau
    tauMaxEta = cms.double(100.0),
    # min dR distance between Taus (to avoid overlap)
    tauMinDR = cms.double(0.1), # 0.1 is the minimal value hardcoded in .cc  

    # To save collection of filtered taus
    saveTags = cms.bool(False)

)
