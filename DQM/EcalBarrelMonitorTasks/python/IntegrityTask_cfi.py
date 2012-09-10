integrityTask = dict(
    MEs = dict(
        ByLumi = dict(path = 'Integrity/IntegrityTask errors by lumi', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Total = dict(path = 'Integrity/IntegrityTask errors total', otype = 'Ecal2P', btype = 'DCC', kind = 'TH1F'),
        Gain = dict(path = 'Integrity/Gain/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        ChId = dict(path = 'Integrity/ChId/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        GainSwitch = dict(path = 'Integrity/GainSwitch/', otype = 'Channel', btype = 'Crystal', kind = 'TH1F'),
        BlockSize = dict(path = 'Integrity/BlockSize/', otype = 'Channel', btype = 'SuperCrystal', kind = 'TH1F'),
        TowerId = dict(path = 'Integrity/TowerId/', otype = 'Channel', btype = 'SuperCrystal', kind = 'TH1F'),
        FEDNonFatal = dict(path = '%(hlttask)s/FEDNonFatal', otype = 'Ecal', btype = 'DCC', kind = 'TH1F'),
        TrendNErrors = dict(path = 'Trend/IntegrityTask number of integrity errors', otype = 'Ecal', btype = 'Trend', kind = 'TH1F')
    )
)

