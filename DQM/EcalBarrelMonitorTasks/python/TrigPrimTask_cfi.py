etAxis = {'nbins': 128, 'low': 0., 'high': 256., 'title': 'TP Et'}
indexAxis = {'nbins': 6, 'low': 0., 'high': 6., 'title': 'TP index'}
bxAxis = {'nbins': 15, 'low': 0., 'high': 15., 'title': 'bunch crossing'}

ecalTrigPrimTask = dict(
    runOnEmul = True,
    HLTCaloPath = 'HLT_SingleJet*',
    HLTMuonPath = 'HLT_Mu5_v*',
    MEs = dict(
        EtReal = dict(path = "TriggerPrimitives/Et/TPTask TP Et 1D", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = etAxis),
        EtMaxEmul = dict(path = "TriggerPrimitives/Emulation/TPTask emul max Et 1D", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = etAxis),
        EtRealMap = dict(path = "TriggerPrimitives/Et/TPTask TP Et", otype = 'SM', btype = 'TriggerTower', kind = 'TProfile2D', zaxis = etAxis),
        EtSummary = dict(path = "TriggerPrimitives/Et/TPTask TP Et", otype = 'Ecal2P', btype = 'TriggerTower', kind = 'TProfile2D', zaxis = etAxis),
        MatchedIndex = dict(path = "TriggerPrimitives/Emulation/Timing/TPTask emul index matching real", otype = 'SM', btype = 'TriggerTower', kind = 'TH2F', yaxis = indexAxis),
        EmulMaxIndex = dict(path = "TriggerPrimitives/Emulation/Timing/TPTask emul max Et index", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = indexAxis),
        EtVsBx = dict(path = "TriggerPrimitives/Et/TPTask TP Et vs BX", otype = 'Ecal2P', btype = 'User', kind = 'TProfile', xaxis = bxAxis, yaxis = {'title': 'TP Et'}),
        OccVsBx = dict(path = "TriggerPrimitives/TPTask TP number vs BX", otype = 'Ecal', btype = 'User', kind = 'TProfile', xaxis = bxAxis),
        HighIntMap = dict(path = "TriggerPrimitives/TPTask tower high interest occupancy", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        MedIntMap = dict(path = "TriggerPrimitives/TPTask tower med interest occupancy", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        LowIntMap = dict(path = "TriggerPrimitives/TPTask tower low interest occupancy", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        TTFlags = dict(path = "TriggerPrimitives/TPTask TT flags", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'nbins': 8, 'low': 0., 'high': 8., 'title': 'TT flag'}),
        TTFMismatch = dict(path = "TriggerPrimitives/Errors/TTFlagMismatch/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        EtEmulError = dict(path = "TriggerPrimitives/Emulation/Errors/Et/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        FGEmulError = dict(path = "TriggerPrimitives/Emulation/Errors/FineGrainBit/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F')
    )
)
