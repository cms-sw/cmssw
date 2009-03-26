import FWCore.ParameterSet.Config as cms

allLayer1Muons = cms.EDProducer("PATMuonProducer",

    # General configurables
    muonSource = cms.InputTag("muons"),
    pfMuonSource = cms.InputTag("pfMuons"),
    useParticleFlow =  cms.bool( False ),

    # user data to add
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
      # add "inline" functions here
      userFunctions = cms.vstring(""),
      userFunctionLabels = cms.vstring("")
    ),
                                
    embedTrack          = cms.bool(False), ## whether to embed in AOD externally stored tracker track
    embedCombinedMuon   = cms.bool(True), ## whether to embed in AOD externally stored combined muon track
    embedStandAloneMuon = cms.bool(True), ## whether to embed in AOD externally stored standalone muon track
    embedPickyMuon      = cms.bool(True), ## whether to embed in AOD externally stored TeV-refit picky muon track
    embedTpfmsMuon      = cms.bool(True), ## whether to embed in AOD externally stored TeV-refit TPFMS muon track
    embedPFCandidate = cms.bool(False),

    # isolation configurables
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
    # embed IsoDeposits to recompute isolation easily
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("muIsoDepositTk"),
        ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
        hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
        user    = cms.VInputTag(
                     cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"), 
                     cms.InputTag("muIsoDepositJets")
                  ),
    ),

    # Resolution configurables
    addResolutions = cms.bool(False),

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("muonTrigMatchHLT1MuonNonIso"), cms.InputTag("muonTrigMatchHLT1MET65")),

    # MC matching configurables
    addGenMatch = cms.bool(True),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("muonMatch"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # TeV refit tracks
    addTeVRefits = cms.bool(True),
    pickySrc = cms.InputTag("tevMuons", "picky"),
    tpfmsSrc = cms.InputTag("tevMuons", "firstHit"),
)






















