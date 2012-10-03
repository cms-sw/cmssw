import FWCore.ParameterSet.Config as cms

ecalIntegrityTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        GainSwitch = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/GainSwitch/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        BlockSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/BlockSize/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('')
        ),
        FEDNonFatal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/FEDIntegrity/FEDNonFatal'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Total number of integrity errors for each FED.')
        ),
        ByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.')
        ),
        Gain = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/Gain/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        Total = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT integrity quality errors summary'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Total number of integrity errors for each FED.')
        ),
        TrendNErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/IntegrityTask number of integrity errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of integrity errors.')
        ),
        ChId = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/ChId/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        TowerId = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/TowerId/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('')
        )
    )
)

