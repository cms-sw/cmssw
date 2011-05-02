import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('HSCP Secondary Dataset'),
    name = cms.untracked.string('$Source: /cvs/CMSSW/CMSSW/Configuration/Skimming/test/,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_P_V16::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/FE4EB995-B657-E011-B8E0-001D09F23A20.root',
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/F8A4A397-B657-E011-8191-003048F11942.root',
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/F24CEADE-AE57-E011-80A2-0030487C7828.root',
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/ECC17F09-4358-E011-A0AC-003048F024F6.root',
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/EC5DDE29-B557-E011-901E-003048F117B6.root',
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/E070342F-BC57-E011-9463-001617DBD472.root',
        '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/DEA19F4B-B057-E011-BC6C-0030487A18D8.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/DE313B2A-B557-E011-9223-003048F1182E.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/D0957E27-B557-E011-9A30-003048F118D4.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/C87775F4-B757-E011-8585-00304879EDEA.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/C6483693-B657-E011-B31D-003048F110BE.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/C468E11D-B357-E011-BDD3-00304879BAB2.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/C02B6B9F-B157-E011-8C7F-003048F118E0.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/BC52B2EF-B057-E011-9F4E-0030487A1884.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/BAFE3149-B057-E011-9C88-0030487CD6E6.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/B6889F5B-B957-E011-92E0-00304879FA4A.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/B215CCA9-C957-E011-BF45-001D09F34488.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/A8C158E0-CD57-E011-8348-001D09F34488.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/A65C332B-B557-E011-BB69-003048CFB40C.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/A48E2515-B857-E011-BB12-0030487CBD0A.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/A235ACBB-B357-E011-BFD4-0030487CD7E0.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/A2162F47-B057-E011-B319-0030487CAEAC.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/A06D5519-B357-E011-83A4-000423D9A2AE.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/9AEDA561-B957-E011-A021-001617C3B77C.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/9A3867D9-BC57-E011-924D-001D09F252F3.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/8AC8A5E9-CF57-E011-BFE6-001D09F25208.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/841D3AC3-B357-E011-BA60-0030487A322E.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/807480A9-B157-E011-AC41-0030487CBD0A.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/7E215956-CC57-E011-996B-001D09F24E39.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/7A964C1D-BA57-E011-89D2-000423D9A212.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/760578F8-C557-E011-9F6A-001D09F251CC.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/6EAC6755-B757-E011-8F89-0030487CD840.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/6A7011A1-B157-E011-B4F5-0030487D0D3A.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/669A885B-B957-E011-8888-0030487C90EE.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/62671D18-B357-E011-A176-0030487CAEAC.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/54252488-AF57-E011-A711-003048F117EA.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/507D33EB-CA57-E011-ADCF-001D09F295FB.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/4ECB911B-9B57-E011-A73E-001617DBCF90.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/4A500FE8-CA57-E011-A9B1-001D09F24664.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/48DE0E77-CE57-E011-9C23-001D09F2A49C.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/46D5CEBE-B357-E011-B382-0030487A17B8.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/3CAC2613-B357-E011-9D1A-0030487C2B86.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/360D6C48-B057-E011-A134-000423D33970.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/349B41B6-0D58-E011-B053-003048F024FA.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/2EDA2F79-CE57-E011-ADDD-0019B9F70468.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/2E8D3C9F-B157-E011-BD9C-003048F024F6.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/1EECDD4A-CA57-E011-A89D-001D09F28F1B.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/1A774D92-D757-E011-8C9F-001D09F29849.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/1627F346-D157-E011-A9FC-001617C3B6DE.root',
       '/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/161/312/02C8AF13-D457-E011-8516-003048F117B4.root'
        ),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/FEE65985-EF55-E011-A137-001617E30F50.root',
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/F4E4E71A-E755-E011-B7BA-001617E30CC8.root',
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/F442D1A6-D755-E011-A93F-003048F01E88.root',
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/F2C06ADC-CD55-E011-88CB-001D09F24FBA.root',
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/E209873D-D655-E011-A0F5-0030487D1BCC.root',
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/E0520B94-DC55-E011-BA0C-003048F1183E.root',
        '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/D8CB81E4-0356-E011-B89B-001617DBCF90.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/D6C94968-E455-E011-B05F-0030487A3232.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/D4C8B603-F355-E011-8367-0030487C2B86.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/D40C2268-E455-E011-BC88-00304879EDEA.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/D2F4E357-F255-E011-ADBF-001617E30D00.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/D2211F89-F455-E011-A90E-001617C3B69C.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/C836B363-E455-E011-B31E-0030487C8E02.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/B6A79013-CB55-E011-A620-001D09F24303.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/AE9079C2-E355-E011-A652-0016177CA778.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/A26C3800-E555-E011-8614-003048F1C424.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/9AEE034B-EB55-E011-831B-001617DC1F70.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/940CC345-DD55-E011-AA77-0030487C2B86.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/90D41E00-E555-E011-A36D-003048F1C832.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/8C6A8656-D155-E011-AD72-003048CFB40C.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/88CC6B3D-F755-E011-A389-0030487CD6E6.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/7E4AA245-FE55-E011-A689-003048F024FE.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/7C981A17-CD55-E011-B46F-001D09F24763.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/7AAFF971-D855-E011-A3D6-0030487C608C.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/6E8B2BEF-D655-E011-B513-001D09F2AD4D.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/649DA768-ED55-E011-ADB7-001617C3B778.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/5A631DC7-CB55-E011-87DE-003048F117EA.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/544B1CD5-F555-E011-84E3-000423D987E0.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/52659776-DA55-E011-A083-003048F118AC.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/4ED10418-FA55-E011-AB39-0030487CD716.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/4A623026-0656-E011-A74C-001617E30F48.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/4018F779-F955-E011-B3D3-001D09F27067.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/3CEE59F1-CF55-E011-A64F-001D09F25438.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/34EB3A3E-E955-E011-B33E-0030487CD178.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/32701DFD-E255-E011-BB14-000423D9997E.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/320EA73D-E955-E011-BEFB-003048F11C5C.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/30A8ECE9-FA55-E011-B13F-003048F118C6.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/2CBE6417-CB55-E011-8A6C-0019B9F72F97.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/2C9B7209-CB55-E011-8988-001D09F25438.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/2ACE3E21-D455-E011-AD7D-003048F1C58C.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/28BB24FC-EB55-E011-B658-000423D996C8.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/20C340C5-E355-E011-960D-001D09F23174.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/20AD0469-D355-E011-AA04-003048F1110E.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/203D4D49-0156-E011-B4A3-001617DBD316.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/1EA86632-F055-E011-BF63-003048F118E0.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/1A7CF9D7-E755-E011-AD6A-003048F1BF66.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/146DBF2B-FC55-E011-9896-001617C3B69C.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/0223EA24-DA55-E011-8F9D-000423D94E70.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/00AFA2B2-FF55-E011-A9E5-001617C3B6E2.root',
       '/store/data/Run2011A/MinimumBias/RAW/v1/000/161/312/00592F8A-CE55-E011-8FE3-000423D98950.root'
        )
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("Configuration.Skimming.PDWG_HSCP_SD_cff")
process.hscpFilter = cms.Path(process.HSCPSD)

process.outputHSCP = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('SD_HSCP')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('hscpFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('SD_HSCP_2011.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputHSCP
)
