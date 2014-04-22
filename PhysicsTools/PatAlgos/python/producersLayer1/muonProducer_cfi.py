import FWCore.ParameterSet.Config as cms

patMuons = cms.EDProducer("PATMuonProducer",
    # input
    muonSource      = cms.InputTag("muons"),

    # use particle flow instead of std reco
    useParticleFlow =  cms.bool( False ),
    pfMuonSource    = cms.InputTag("particleFlow"),

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
    embedMuonBestTrack      = cms.bool(True),  ## embed in AOD externally stored muon best track from global pflow
    embedTunePMuonBestTrack = cms.bool(True),  ## embed in AOD externally stored muon best track from muon only
    forceBestTrackEmbedding = cms.bool(False), ## force embedding separately the best tracks even if they're already embedded e.g. as tracker or global tracks
    embedTrack          = cms.bool(False), ## embed in AOD externally stored tracker track
    embedCombinedMuon   = cms.bool(True),  ## embed in AOD externally stored combined muon track
    embedStandAloneMuon = cms.bool(True),  ## embed in AOD externally stored standalone muon track
    embedPickyMuon      = cms.bool(True),  ## embed in AOD externally stored TeV-refit picky muon track
    embedTpfmsMuon      = cms.bool(True),  ## embed in AOD externally stored TeV-refit TPFMS muon track
    embedDytMuon        = cms.bool(True),  ## embed in AOD externally stored TeV-refit DYT muon track
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






















