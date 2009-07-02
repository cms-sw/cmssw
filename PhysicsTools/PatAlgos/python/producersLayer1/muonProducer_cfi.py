import FWCore.ParameterSet.Config as cms

allLayer1Muons = cms.EDProducer("PATMuonProducer",
    # input
    muonSource      = cms.InputTag("muons"),

    # use particle flow instead of std reco                                
    useParticleFlow =  cms.bool( False ),
    pfMuonSource    = cms.InputTag("pfIsolatedMuons"),          

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
    embedPFCandidate    = cms.bool(False), ## embed in AOD externally stored particle flow candidate

    # define IsoDeposits to recompute isolation values on the fly in the producer.
    # not used in the case of PF2PAT
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
            deltaR = cms.double(0.3)
        ),
        tracker = cms.PSet(
            src = cms.InputTag("muIsoDepositTk"),
            deltaR = cms.double(0.3)
        ),
        user = cms.VPSet(cms.PSet(
            src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
            deltaR = cms.double(0.3)
            ), 
            cms.PSet(
                src = cms.InputTag("muIsoDepositJets"),
                deltaR = cms.double(0.3)
            )),
        ecal = cms.PSet(
            src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
            deltaR = cms.double(0.3)
        )
    ),

    
    # embed IsoDeposits
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("muIsoDepositTk"),
        ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
        hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
        user    = cms.VInputTag(
                     cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"), 
                     cms.InputTag("muIsoDepositJets")
                  ),
    ),
    

    # mc matching
    addGenMatch   = cms.bool(True),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("muonMatch"), ## particles source to be used for the matching

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution configurables
    addResolutions  = cms.bool(False),
    resolutions      = cms.PSet(),
)






















