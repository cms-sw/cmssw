trigPrimTask = dict(
    runOnEmul = True,
    expectedTiming = 3,
    HLTCaloPath = 'HLT_SingleJet*',
    HLTMuonPath = 'HLT_Mu5_v*'
)

trigPrimTaskPaths = dict(
    EtReal = "TriggerPrimitives/Et/TPTask TP Et 1D",
    EtEmul = "TriggerPrimitives/Et/TPTask emul Et 1D",
    EtMaxEmul = "TriggerPrimitives/Et/TPTask emul max Et 1D",
    EtRealMap = "TriggerPrimitives/Et/TPTask TP Et",
    EtEmulMap = "TriggerPrimitives/Et/Emulation/TPTask emul Et",
    MatchedIndex = "TriggerPrimitives/EmulMatching/MatchIndex/TPTask emul index matching real",
    EmulMaxIndex = "TriggerPrimitives/EmulMatching/TPTask emul max Et index",
    TimingError = "TriggerPrimitives/EmulMatching/Errors/Timing/",
    EtVsBx = "TriggerPrimitives/Et/TPTask TP Et vs BX",
    OccVsBx = "TriggerPrimitives/TPTask TP number vs BX",
    HighIntMap     = "TriggerPrimitives/TPTask tower high interest occupancy",
    MedIntMap      = "TriggerPrimitives/TPTask tower med interest occupancy",
    LowIntMap      = "TriggerPrimitives/TPTask tower low interest occupancy",
    TTFlags          = "TriggerPrimitives/TPTask TT flags",
    TTFMismatch      = "TriggerPrimitives/TPTask TT flag mismatch",
    TimingCalo = "TriggerPrimitives/TPTask TP timing calo triggers",
    TimingMuon = "TriggerPrimitives/TPTask TP timing muon triggers",
    EtEmulError = "TriggerPrimitives/EmulMatching/Errors/Et/",
    FGEmulError = "TriggerPrimitives/EmulMatching/Errors/FineGrainBit/"
)
