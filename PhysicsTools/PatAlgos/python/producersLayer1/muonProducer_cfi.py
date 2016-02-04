import FWCore.ParameterSet.Config as cms

patMuons = cms.EDProducer("PATMuonProducer",
    # input
    muonSource      = cms.InputTag("muons"),

    # use particle flow instead of std reco                                
    useParticleFlow =  cms.bool( False ),
    pfMuonSource    = cms.InputTag("pfIsolatedMuons"),          
    linkToPFSource  = cms.InputTag(""),

    # add TeV refit tracks
    addTeVRefits    = cms.bool(True),
    pickySrc        = cms.InputTag("tevMuons", "picky"),
    tpfmsSrc        = cms.InputTag("tevMuons", "firstHit"),

    # add user data
    userData = cms.PSet(
      # add custom classes here
      userClasses = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add doubles here
      userFloats = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add ints here
      userInts = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add candidate ptrs here
      userCands = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),

    # embedding objects
    embedTrack          = cms.bool(False), ## embed in AOD externally stored tracker track
    embedCombinedMuon   = cms.bool(True),  ## embed in AOD externally stored combined muon track
    embedStandAloneMuon = cms.bool(True),  ## embed in AOD externally stored standalone muon track
    embedPickyMuon      = cms.bool(True),  ## embed in AOD externally stored TeV-refit picky muon track
    embedTpfmsMuon      = cms.bool(True),  ## embed in AOD externally stored TeV-refit TPFMS muon track
    embedPFCandidate    = cms.bool(True),  ## embed in AOD externally stored particle flow candidate

    # embedding of muon MET corrections for caloMET
    embedCaloMETMuonCorrs = cms.bool(True),
    caloMETMuonCorrs = cms.InputTag("muonMETValueMapProducer"  , "muCorrData"),
    # embedding of muon MET corrections for tcMET
    embedTcMETMuonCorrs   = cms.bool(True),
    tcMETMuonCorrs   = cms.InputTag("muonTCMETValueMapProducer", "muCorrData"),
                          
    # embed IsoDeposits
    isoDeposits = cms.PSet(
        #user    = cms.VInputTag(
        #             cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"), 
        #             cms.InputTag("muIsoDepositJets")
        #          ),
    ),
    
    # user defined isolation variables the variables defined here will be accessible
    # via pat::Muon::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = cms.PSet(
        #user = cms.VPSet(cms.PSet(
        #    src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
        #    deltaR = cms.double(0.3)
        #    ), 
        #    cms.PSet(
        #        src = cms.InputTag("muIsoDepositJets"),
        #        deltaR = cms.double(0.3)
        #    )),
    ),

    # mc matching
    addGenMatch   = cms.bool(True),
    embedGenMatch = cms.bool(True),
    genParticleMatch = cms.InputTag("muonMatch"), ## particles source to be used for the matching

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution configurables
    addResolutions  = cms.bool(False),
    resolutions      = cms.PSet(),

    # high level selections
    embedHighLevelSelection = cms.bool(True),
    usePV                   = cms.bool(True),                          
    beamLineSrc             = cms.InputTag("offlineBeamSpot"),
    pvSrc                   = cms.InputTag("offlinePrimaryVertices")
)






















