import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

### Miscellanous ###

# Logging #
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool( True )
)
process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'INFO' )
    )
)

# # Profiling #
# process.ProfilerService = cms.Service( "ProfilerService",
#     paths = cms.untracked.vstring(
#         'FullEvent'
#     )
# )

# Memory check #
process.SimpleMemoryCheck = cms.Service( "SimpleMemoryCheck",
#     oncePerEventMode = cms.untracked.bool( True ),
    ignoreTotal      = cms.untracked.int32( 0 )
)

### Import ###

# Magnetic fiels #
# process.load( "Configuration.StandardSequences.MagneticField_0T_cff" )
process.load( "Configuration.StandardSequences.MagneticField_38T_cff" )
# Geometry #
process.load( "Configuration.StandardSequences.Geometry_cff" )
# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
# process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'CRAFT_30X::All'
process.es_prefer_GlobalTag = cms.ESPrefer(
    'PoolDBESSource',
    'GlobalTag'
)

### SiStrip DQM ###

process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

### Input ###

# Source #
process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        # run 70036 (B=3.8T), re-reco
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/32281C3A-68C1-DD11-BB66-001D0967D5DF.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/324DB530-5EC1-DD11-887F-001D0967CE69.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/326FDF8B-61C1-DD11-B0AD-001D0967DE63.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/5407EF07-5DC1-DD11-B15B-001D0967D689.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/C05C4637-8DC1-DD11-8435-001D0968F36E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/CE818FC0-55C1-DD11-A75F-0019B9E48C7C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/F034524B-8DC1-DD11-A87E-001D0967DF6C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/386666DE-02C2-DD11-A037-001D0969096E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/3E6E3B12-A1C1-DD11-B1AF-001D0967DF67.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/448A7FC2-02C2-DD11-81DB-001D0967C149.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/60C3A4F1-B8C1-DD11-8F78-0019B9E4F89A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/646C8506-1BC2-DD11-85F2-001D096B0F80.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/D41BDF7E-ACC1-DD11-8DC3-001D0967D625.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/E8E505B6-10C2-DD11-B3FC-001D0967D1DE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0002/8E1606F7-2FC2-DD11-B2E2-001D0966E23E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0002/E42D292C-3DC2-DD11-9BF1-001D0967CEF5.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0003/2276FCED-67C2-DD11-8754-0019B9E48897.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0003/2869DEEA-67C2-DD11-A2A9-001D0967DAE4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0003/C82E08ED-67C2-DD11-BB66-0019B9E4FD98.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0004/92AA896C-9EC2-DD11-BCEF-001D0967CCE3.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0004/AE77852A-BAC2-DD11-9F9F-001D0967D25B.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0004/B2E2E533-AAC2-DD11-A155-0019B9E4B146.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0004/FEDB35D3-93C2-DD11-A052-001D0967CF86.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0005/3ED14E48-DAC2-DD11-9670-0019B9E4FCD0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0005/CCCFCF14-05C3-DD11-B9F1-001D09690089.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0005/E8DC56CB-11C3-DD11-97A0-001D0967CFA9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/0EF177C7-91C3-DD11-A0CB-001D0967DFFD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/16CF7C61-62C3-DD11-AC5F-0019B9E4F3AE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/2CAE236F-CCC3-DD11-83EE-001D096763C7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/422B48FF-66C3-DD11-86E1-001D0967DE90.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/44433A2B-C5C3-DD11-BF89-001D0967DEEA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/B0382A91-6DC3-DD11-940B-001D0967DAE4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/B222B130-BFC3-DD11-93C5-001D0967DA49.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/C8C8105B-6FC3-DD11-9E92-001D0967D2DD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/225FEEE3-6CC4-DD11-8A3E-0019B9E71465.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/28C03E8B-FFC3-DD11-BC06-001D0967D319.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/8A66AB75-40C4-DD11-8952-001D0967DF17.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/92AF9228-F1C3-DD11-8ADF-001D0967D49F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/C6B1CB39-25C4-DD11-A5AD-001D0967D7DD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/D2502DB4-70C4-DD11-8CBA-0019B9E48FC5.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/F808AE02-37C4-DD11-835C-0019B9E480D6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0008/EE56BEBB-A6C4-DD11-95A4-001D0967D517.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0010/AED84C02-EAC4-DD11-A95B-0019B9E4AC46.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0010/D8FF0892-F7C4-DD11-978D-001D0967D580.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0013/0C9B6B89-41C5-DD11-A99F-0019B9E4FB72.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0013/8A8E72A2-41C5-DD11-B102-0019B9E4B010.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0013/8C9D6579-3EC5-DD11-984B-001D0967DBFC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0013/94FB5B92-41C5-DD11-B085-001D0967DFB7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0014/0E479A66-55C5-DD11-B4FF-001D0967CFC7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0014/72565845-4AC5-DD11-9AC7-001D0967D625.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0014/8880657B-58C5-DD11-AC2F-001D0968F765.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0018/447E07CF-B2C5-DD11-8D8B-001D0967D2A1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0019/32C9B75E-DFC5-DD11-8F00-001D0967DBE8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0019/A4F5C7AA-CBC5-DD11-AD10-001D0967D6E3.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0019/D8F0AB42-CBC5-DD11-91FB-001D0967DAB7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0020/D6FDFF25-F2C5-DD11-B021-001D0967DAC1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0021/D82ADC04-F9C5-DD11-BEFC-001D0967D60C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0021/E4A82F10-15C6-DD11-A9FF-0019B9E50090.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0022/D6EDE13E-20C6-DD11-9F08-001D0967D085.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0023/FA50E153-34C6-DD11-B3C2-0019B9E489B4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0024/12AC84BF-57C6-DD11-87AD-0019B9E4FE06.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0025/68FC2A04-6CC6-DD11-8E1E-001D0967DC83.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0028/02E0F0A7-D9C6-DD11-93D7-001D0967CFCC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0030/50A40E86-02C7-DD11-8301-001D0967DAAD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0030/5E11785B-02C7-DD11-A6CB-0019B9E7B929.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0033/4E0A146A-37C7-DD11-AC43-001D0967DCAB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0033/CEA54531-37C7-DD11-A7D3-0019B9E4FD57.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0035/AA444A37-D4C7-DD11-9915-00145EDD7A0D.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0044/2468ABAF-57C8-DD11-A2B7-001D0967DC6F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0045/0494FEA3-86C8-DD11-AEA7-001D0968EEE6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0045/121BEE35-A7C8-DD11-A0DF-001D0967D337.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0045/CEB2CD59-A5C8-DD11-86E7-0019B9E7140B.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/280D9428-24C9-DD11-8BE5-001D0967DFF3.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/30CA978E-F5C8-DD11-A481-001D0967DDEB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/5209EFBC-27C9-DD11-8FDC-001D0967D9B8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/5447EB68-C2C8-DD11-AA9A-001D0967C1E9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/94FE4D8B-F5C8-DD11-AA95-0019B9E48B41.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/A0AF3F6B-25C9-DD11-8661-001D096B0F08.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/243C330C-FEC8-DD11-BCED-001D0967DB7F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/2C11B0C7-56C9-DD11-ABEF-001D0967D337.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/D843AAE7-21C9-DD11-B196-001D0967D63E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/E279F407-0EC9-DD11-8F64-001D0969086F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/EC2B2AE5-52C9-DD11-BADA-001D0967DB98.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/18319BB5-FDC9-DD11-984A-001D0967D97C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/6045BDB3-EAC9-DD11-A074-0019B9E50108.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/60CB5FCD-F3C9-DD11-B2A1-001D0967DF12.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/B2E95FDB-E8C9-DD11-9A2D-001D0967D11B.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/C27F0D69-0CCA-DD11-BEC9-001D0967D26A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/DC408954-EDC9-DD11-959A-001D0967C03F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/E4AEBB0A-EEC9-DD11-85D0-001D0967D19D.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/1CC334C4-E6CA-DD11-88AD-0019B9E487D7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/4C9A8B6C-E0CA-DD11-8F0A-001D0967DE0E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/86829BFF-EDCA-DD11-BF4B-001D0967DC83.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/9C65BE44-F1CA-DD11-B127-001D0967DE4A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/AE2C0933-DACA-DD11-AC05-001D0967DB7F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/E09DC1A5-D4CA-DD11-997B-001D0967D341.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/E441119C-D7CA-DD11-AB5C-001D0967DFA8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/1E2228F7-D6CB-DD11-AB3C-001D0967DE54.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/28BCCDD0-D1CB-DD11-A83C-001D0967DE95.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/72A61905-B9CB-DD11-9135-001D0967DCB5.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/8C1CDB42-B1CB-DD11-82B4-001D0967DF12.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/A0014FFC-B2CB-DD11-9DF1-001D0966E23E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/A45C66D1-D0CB-DD11-B098-001D0968F26A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0050/BC858104-BACB-DD11-84AD-001D0968F2F6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0051/10CC97DA-00CC-DD11-B114-0019B9E48730.root'
#         # run 70421 (B=0T), re-reco
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/288D8130-66C1-DD11-B33F-001D0967DD0A.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/0A5D0DE2-02C2-DD11-BE89-001D0967DB98.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0003/D62B2FED-67C2-DD11-84D3-0019B9E4FCA8.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/28B4A51C-9BC3-DD11-8E8E-001D0967D986.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/7A91BFCA-C6C3-DD11-B6EC-001D0967B82E.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/A023DB09-AAC3-DD11-B66B-001D0967D21A.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0006/A43AEAFC-82C3-DD11-A095-001D09707CB5.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/80524102-0AC4-DD11-9131-00145EDD75D9.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/BEC814FA-11C4-DD11-B0CA-0019B9E4B1F0.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0032/6A51D241-2AC7-DD11-B276-001D0967DA3A.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0032/6C9F67D1-28C7-DD11-A26F-001D0967CFCC.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0032/BCC75AD9-28C7-DD11-80E4-0019B9E8B5DD.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0033/20D4B573-2CC7-DD11-A39A-001D0967C130.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0034/B063298B-4FC7-DD11-BC2C-001D0967D5DF.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0034/BE2F1907-7DC7-DD11-88F4-001D0967D9B8.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0035/2C9C03DE-E5C7-DD11-9577-0019B9E714F1.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0045/C8ACC971-A7C8-DD11-A1DC-0019B9E4FA2B.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/049F0EB8-C5C8-DD11-ABF9-001D0967CF1D.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/2C147139-D1C8-DD11-BA80-001D0967D5F8.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/38278ED1-C5C8-DD11-A430-0019B9E7DE7D.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/3A51D68C-F5C8-DD11-977C-001D0967DB11.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/C28BE3B5-C5C8-DD11-B164-001D0967D37D.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0046/CAF06B36-2CC9-DD11-AFA3-0019B9E489B4.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/4409EED0-FFC8-DD11-B897-001D0967D341.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/8C7A8480-FBC8-DD11-AF77-001D0967D4EF.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0047/D4BE6D81-4DC9-DD11-B569-001D0967D670.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/14E04029-02CA-DD11-9042-001D0967C130.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0048/94623D98-F8C9-DD11-9184-001D0967D247.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0049/EE80A5CD-D7CA-DD11-910F-001D0967DE13.root'
    )
)
# Input steering #
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10000 )
)

# HLT Filter #
process.hltFilter = cms.EDFilter( "HLTHighLevel",
    HLTPaths           = cms.vstring(
        'HLT_WhatEverFunnyFilter',
        'HLT_TrackerCosmics',
        'HLT_TrackerCosmics_CoTF',
        'HLT_TrackerCosmics_RS'  ,
        'HLT_TrackerCosmics_CTF'
    ),
    eventSetupPathsKey = cms.string( '' ),
    andOr              = cms.bool( True ),
    throw              = cms.bool( False ),
    # use this according to https://hypernews.cern.ch/HyperNews/CMS/get/global-runs/537.html
    TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'HLT' )
#     TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'FU' )    
)

### Scheduling ###

process.p = cms.Path(
    process.hltFilter                            * # comment this out to switch off the HLT pre-selection
#     process.SiStripDQMRecoFromRaw                * # comment this out when running from RECO or with full reconstruction
#     process.SiStripDQMSourceGlobalRunCAF_fromRAW * # comment this out when running from RECO or with full reconstruction
#     process.SiStripDQMRecoGlobalRunCAF           *
#     process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripMonitorClusterCAF             *
#     process.SiStripOfflineDQMClient              *
#     process.qTester                              *
    process.dqmSaver
#     process.MEtoEDMConverter
)

### Output ###

# DQM Saver path
process.dqmSaver.dirName = '.'

# PoolOutput #
process.out = cms.OutputModule( "PoolOutputModule",
    fileName       = cms.untracked.string( './SiStripDQMOfflineGlobalRunCAF.root' ),
    SelectEvents   = cms.untracked.PSet(
        SelectEvents = cms.vstring( 'p' )
    ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_SiStripDQMOfflineGlobalRunCAF'
    )
)

# process.outpath = cms.EndPath(
#     process.out
# )
