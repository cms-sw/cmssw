import FWCore.ParameterSet.Config as cms

allLayer1MHTs = cms.EDProducer(
    "PATMHTProducer",
    # General configurables
    verbose = cms.double(0.),

    jetTag       = cms.untracked.InputTag("selectedLayer1Jets"),
    electronTag  = cms.untracked.InputTag("selectedLayer1Electrons"),
    muonTag      = cms.untracked.InputTag("selectedLayer1Muons"),
    tauTag       = cms.untracked.InputTag("selectedLayer1Taus"),
    photonTag    = cms.untracked.InputTag("selectedLayer1Photons"),
    
    # Selection configurables
    
    jetPtMin      = cms.double( 20.),
    jetEtaMax     = cms.double( 5. ),
    jetEMfracMax  = cms.double( 0.9),
    elePtMin      = cms.double( 10.),
    eleEtaMax     = cms.double( 3. ),
    muonPtMin     = cms.double( 10.),
    muonEtaMax    = cms.double( 2.5),
    
    # Resolution configurables

    uncertaintyScaleFactor = cms.double(1.0 ),
    
    controlledUncertainty = cms.bool(True), #use controlled uncertainty parameters.
    
    # -------------------------------------------
    #  Jet Uncertainties
    # -------------------------------------------
    # //-- values from PTDR 1, ch 11.4 --//
    jetEtUncertaintyParameter0 = cms.double(5.6 ), 
    jetEtUncertaintyParameter1 = cms.double(1.25),
    jetEtUncertaintyParameter2 = cms.double(0.033),

    # // values from :
    # http://indico.cern.ch/getFile.py/access?contribId=9&sessionId=0&resId=0&materialId=slides&confId=46394
    jetPhiUncertaintyParameter0 =cms.double(4.75  ),
    jetPhiUncertaintyParameter1 =cms.double(-0.426),
    jetPhiUncertaintyParameter2 =cms.double(0.023 ), 
    
    # -------------------------------------------
    #  Electron Uncertainties
    # -------------------------------------------
    eleEtUncertaintyParameter0  =cms.double ( 0.01),   
    elePhiUncertaintyParameter0 =cms.double (0.01 ),  
    
    # -------------------------------------------
    #  Muon Uncertainties
    # -------------------------------------------

    muonEtUncertaintyParameter0  = cms.double(0.01),   
    muonPhiUncertaintyParameter0 = cms.double(0.01),   
 
    # -------------------------------------------
    #  For MET Significance 
    # -------------------------------------------
   
    #CaloTowerTag    = cms.InputTag("caloTowers"),
    CaloTowerTag    = cms.InputTag("towerMaker"),
    noHF            = cms.bool(False),
)


