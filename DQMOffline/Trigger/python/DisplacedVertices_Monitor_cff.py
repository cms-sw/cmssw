import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

PFHT330_TripleBTag_bjet = hltTOPmonitoring.clone(
    FolderName   = 'HLT/EXO/DisplacedVertices/FullyHadronic/TripleBTag/',
    enable2DPlots = False,
    # Selections
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>45 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 4,
    bjetSelection    = 'pt>45 & abs(eta)<2.4',
    btagAlgos        = ["pfParticleNetAK4DiscriminatorsJetTagsForRECO:BvsAll"],
    workingpoint     = 0.0359, # Loose (According to: https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer23BPix/)
    # Binning
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                 jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
                 HTBinning    = [0,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers 
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_4p3_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*']),
    #requireValidHLTPaths = False, #For debugging
)

PFHT330_TripleBTag_bjet_backup = hltTOPmonitoring.clone(
    FolderName   = 'HLT/EXO/DisplacedVertices/FullyHadronic/TripleBTagBackup/',
    enable2DPlots = False,
    # Selections                                                                                                                                                                                                   
    leptJetDeltaRmin = 0.0,
    njets            = 4,
    jetSelection     = 'pt>45 & abs(eta)<2.4',
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 500,
    nbjets           = 4,
    bjetSelection    = 'pt>45 & abs(eta)<2.4',
    btagAlgos        = ["pfParticleNetAK4DiscriminatorsJetTagsForRECO:BvsAll"],
    workingpoint     = 0.0359, #Loose
    # Binning                                                                                                                                                  
    histoPSet = dict(htPSet = dict(nbins= 50, xmin= 0.0, xmax= 1000),
                 jetPtBinning = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400],
                 HTBinning    = [0,460,480,500,520,540,560,580,600,650,700,750,800,850,900]),
    # Triggers                                                                                                                                                                                                     
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_2p0_v*']),
    denGenericTriggerEventPSet = dict(hltPaths = ['HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v*']),
    #requireValidHLTPaths = False, #For debugging
)

exoHLTDisplacedVerticesmonitoring = cms.Sequence(
    PFHT330_TripleBTag_bjet
    + PFHT330_TripleBTag_bjet_backup
)
