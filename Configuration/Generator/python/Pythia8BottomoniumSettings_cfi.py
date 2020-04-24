import FWCore.ParameterSet.Config as cms

pythia8BottomoniumSettingsBlock = cms.PSet(
    pythia8BottomoniumSettings = cms.vstring(
            'Bottomonium:O(3S1)[3S1(1)] = 9.28',
            'Bottomonium:O(3S1)[3S1(8)] = 0.15',
            'Bottomonium:O(3S1)[1S0(8)] = 0.02',
            'Bottomonium:O(3S1)[3P0(8)] = 0.02',
            'Bottomonium:gg2bbbar(3S1)[3S1(1)]g = on',
            'Bottomonium:gg2bbbar(3S1)[3S1(1)]gm = on',
            'Bottomonium:gg2bbbar(3S1)[3S1(8)]g = on',
            'Bottomonium:qg2bbbar(3S1)[3S1(8)]q = on',
            'Bottomonium:qqbar2bbbar(3S1)[3S1(8)]g = on',
            'Bottomonium:gg2bbbar(3S1)[1S0(8)]g = on',
            'Bottomonium:qg2bbbar(3S1)[1S0(8)]q = on',
            'Bottomonium:qqbar2bbbar(3S1)[1S0(8)]g = on',
            'Bottomonium:gg2bbbar(3S1)[3PJ(8)]g = on',
            'Bottomonium:qg2bbbar(3S1)[3PJ(8)]q = on',
            'Bottomonium:qqbar2bbbar(3S1)[3PJ(8)]g = on',
    )
)
