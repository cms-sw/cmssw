import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Onia central skim'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Skimming/test/CSmaker_Onia_PDMu_1e29_reprocess361p3_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_R_36X_V11A::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F85204EE-EB40-DF11-8F71-001A64789D1C.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F80836A2-FF40-DF11-A43E-00E08178C067.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F65A94F7-4141-DF11-9F4E-003048D47A80.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F2118BE5-FF40-DF11-B2F0-00E081791865.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/E80DA1CD-0041-DF11-8CE3-0025B3E063A8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/E6261B1F-EC40-DF11-89F7-00E08178C045.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/E2F3F819-F040-DF11-9B20-00E08178C0F5.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/DA86BDB2-E940-DF11-A3A8-0025B3E063A8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/D6DB0FEC-FD40-DF11-93FC-003048D46004.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/D4C61300-E940-DF11-A659-003048673EA4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/D27E9F23-FC40-DF11-92F3-00E08178C067.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/CEDBFF0C-E640-DF11-931F-003048D45FA2.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/CA0525F2-EC40-DF11-948D-0025B31E3C0A.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/C63BB055-EA40-DF11-B401-003048673F8A.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/C4DE0B83-ED40-DF11-BDC6-003048D46110.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/C4A9EF9C-F240-DF11-A5F3-002481E150FC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/BC865204-EF40-DF11-A1FF-00E0817917E7.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/B84C00C3-ED40-DF11-AE86-00E0817917B9.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/B6FC5BA2-EB40-DF11-AC16-003048D45FEA.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/B62BF913-FE40-DF11-95C0-0015170AE328.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/B0889A04-F240-DF11-BA37-00E081B08BC9.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/ACCAE48C-EB40-DF11-B88F-0025B3E05CE4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/AC1AA07C-4041-DF11-8F7E-00E081791813.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/A65FBFDC-E640-DF11-9BC5-003048D45FEA.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/A4AF10B5-0041-DF11-B6A0-0025B3E0650E.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/A41D9B67-EC40-DF11-A121-003048D45FA8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/A2CE75FE-EE40-DF11-A061-003048D4600C.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/A286F3B9-EE40-DF11-8F19-003048D47796.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/A26890AB-E740-DF11-97DC-003048D45FA2.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/9CE0781A-EC40-DF11-A3BD-003048D45FA2.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/9CAF81A9-E940-DF11-A84C-003048D4600C.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/9C70947C-ED40-DF11-825C-003048D47A84.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/9AB2191A-EF40-DF11-B3D4-001A64789DDC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/98842725-FC40-DF11-BF7E-001A64789DEC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/98098625-FF40-DF11-8DBA-003048D47A1A.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/94A4A967-E940-DF11-8008-003048D47A1A.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/944D50F8-EF40-DF11-918F-002481E14FCA.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/8A446821-EF40-DF11-9FC1-002481E150FC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/88F8CF52-EA40-DF11-8D78-003048673F74.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/86871F39-FF40-DF11-BC3C-003048D46004.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/860DED05-3E41-DF11-B4E2-00E08178C181.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/80673EAD-E740-DF11-AE30-0025B3E05CE4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/709CE2A3-EB40-DF11-9B71-003048D45FD4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/702A765E-F140-DF11-A66F-002481E14D76.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/700A76F0-FC40-DF11-9BB3-0015170ACA88.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/6A5EB2B2-E740-DF11-A543-0015170ACA88.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/66D8AECE-EF40-DF11-BFB8-00E08178C05F.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/669FC07F-ED40-DF11-9532-001A64789458.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/667FDFA7-E940-DF11-AF69-0025B3E05CE4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/5EFD390B-ED40-DF11-89D1-0025B3E06698.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/5E800FB2-F240-DF11-AC55-0025B3E05CDA.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/5E53FAC0-E440-DF11-A2CF-003048D4774E.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/5A90DBB0-F340-DF11-A32E-001A64789DDC.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/4E3D0F6F-ED40-DF11-91D2-0025B3E05CE4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/4CA21995-EB40-DF11-A5E0-003048D4600C.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/489A0CF6-E940-DF11-89A6-003048D46004.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/46DB27A6-E940-DF11-A620-003048D45FA2.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/42E38BB5-EF40-DF11-819D-003048D47A46.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/4299620D-F240-DF11-80E6-003048635E12.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/3CC87CA8-3B41-DF11-958D-003048D460D4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/3680D462-EB40-DF11-B36F-003048D45FC8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/283724A6-FA40-DF11-B8D2-001A64789DF8.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/24F9EA4B-F340-DF11-9FFA-001A64789D1C.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/22D65FC8-EF40-DF11-B2FB-003048D4624A.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/1E29E006-4141-DF11-9FF1-002481E15000.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/1E127F4A-EE40-DF11-AAE6-003048635E12.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/1C1A8D9C-F140-DF11-8A96-003048D47A46.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/1AF719A8-F140-DF11-AB66-001A64789458.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/16F51B00-E940-DF11-B3FE-00E0817917E7.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/162DD150-FB40-DF11-AD4A-003048673EA4.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/0E1FFB86-0141-DF11-805E-003048D46028.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/0813F6D1-FF40-DF11-A473-003048D47774.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/02071949-FA40-DF11-9990-001A64789DEC.root'
        )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

import HLTrigger.HLTfilters.hltHighLevelDev_cfi

### Onia skim CS
process.goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isGlobalMuon || (isTrackerMuon && numberOfMatches('SegmentAndTrackArbitration')>0)"),
)
process.diMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodMuons goodMuons"),
    checkCharge = cms.bool(False),
    cut         = cms.string("mass > 2"),
)
process.diMuonFilter = cms.EDFilter("CandViewCountFilter",
    src       = cms.InputTag("diMuons"),
    minNumber = cms.uint32(1),
)
process.Skim_diMuons = cms.Path(
    process.goodMuons    *
    process.diMuons      *
    process.diMuonFilter
)



process.outputCsOnia = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('CS_Onia')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('Skim_diMuons')),                                        
                                        outputCommands = process.RECOEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Onia_1e29.root')
                                        )


process.this_is_the_end = cms.EndPath(process.outputCsOnia)
