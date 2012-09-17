etAxis = {'nbins': 128, 'low': 0., 'high': 256., 'title': 'TP Et'}
indexAxis = {'nbins': 6, 'low': 0., 'high': 6., 'title': 'TP index'}
bxAxis = {'nbins': 15, 'low': 0., 'high': 15., 'title': 'bunch crossing'}

ecalTrigPrimTask = dict(
    runOnEmul = True,
    HLTCaloPath = 'HLT_SingleJet*',
    HLTMuonPath = 'HLT_Mu5_v*',
    MEs = dict(
        EtReal = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et spectrum Real Digis%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = etAxis),
        EtMaxEmul = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/Emulated/%(prefix)sTTT Et spectrum Emulated Digis max%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = etAxis),
        EtRealMap = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et map Real Digis %(sm)s", otype = 'SM', btype = 'TriggerTower', kind = 'TProfile2D', zaxis = etAxis),
        EtSummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Et trigger tower summary", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TProfile2D', zaxis = etAxis),
        MatchedIndex = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulMatch %(sm)s", otype = 'SM', btype = 'TriggerTower', kind = 'TH2F', yaxis = indexAxis),
        EmulMaxIndex = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT max TP matching index%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = indexAxis),
        EtVsBx = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et vs bx Real Digis%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TProfile', xaxis = bxAxis, yaxis = {'title': 'TP Et'}),
        OccVsBx = dict(path = "%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TP occupancy vs bx Real Digis%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TProfile', xaxis = bxAxis),
        HighIntMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower high interest counter%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        MedIntMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower med interest counter%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        LowIntMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F'),
        TTFlags = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT Flags%(suffix)s", otype = 'Ecal3P', btype = 'DCC', kind = 'TH2F', yaxis = {'nbins': 8, 'low': 0., 'high': 8., 'title': 'TT flag'}),
        TTFMismatch = dict(path = "Ecal/Errors/TriggerPrimitives/FlagMismatch/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        EtEmulError = dict(path = "Ecal/Errors/TriggerPrimitives/EtEmulation/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F'),
        FGEmulError = dict(path = "Ecal/Errors/TriggerPrimitives/FGBEmulation/", otype = 'Channel', btype = 'TriggerTower', kind = 'TH1F')
    )
)
