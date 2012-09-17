ecalIntegrityTask = dict(
    MEs = dict(
        ByLumi = dict(path = '%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Total = dict(path = '%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT integrity quality errors summary', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Gain = dict(path = 'Ecal/Errors/Integrity/Gain/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        ChId = dict(path = 'Ecal/Errors/Integrity/ChId/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        GainSwitch = dict(path = 'Ecal/Errors/Integrity/GainSwitch/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        BlockSize = dict(path = 'Ecal/Errors/Integrity/BlockSize/', otype = 'Channel', btype = 'SuperCrystal', kind = 'TH1F'),
        TowerId = dict(path = 'Ecal/Errors/Integrity/TowerId/', otype = 'Channel', btype = 'SuperCrystal', kind = 'TH1F'),
        FEDNonFatal = dict(path = '%(subdet)s/FEDIntegrity/FEDNonFatal', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        TrendNErrors = dict(path = 'Ecal/Trends/IntegrityTask number of integrity errors', otype = 'Ecal', btype = 'Trend', kind = 'TH1F')
    )
)

