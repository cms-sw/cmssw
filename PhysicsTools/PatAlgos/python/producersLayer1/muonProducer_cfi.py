import FWCore.ParameterSet.Config as cms

allLayer1Muons = cms.EDProducer("PATMuonProducer",

    # General configurables
    muonSource = cms.InputTag("allLayer0Muons"),
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
    embedPFCandidate = cms.bool(False),

    # isolation configurables
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
            deltaR = cms.double(0.3)
        ),
        tracker = cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositTk"),
            deltaR = cms.double(0.3)
        ),
        user = cms.VPSet(cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersho"),
            deltaR = cms.double(0.3)
        ), 
            cms.PSet(
                src = cms.InputTag("layer0MuonIsolations","muIsoDepositJets"),
                deltaR = cms.double(0.3)
            )),
        ecal = cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
            deltaR = cms.double(0.3)
        )
    ),
    # embed IsoDeposits to recompute isolation easily
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("layer0MuonIsolations","muIsoDepositTk"),
        ecal    = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
        hcal    = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
        user    = cms.VInputTag(
                     cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersho"), 
                     cms.InputTag("layer0MuonIsolations","muIsoDepositJets")
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

)


