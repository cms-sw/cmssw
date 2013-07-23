import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('Tau central skim'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Skimming/test/test_tauSkim.py,v $')
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
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/E86FFC9E-D757-E011-BE46-001D09F2915A.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/D62438A0-E357-E011-A372-001D09F24664.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/D43A6E46-DF57-E011-931C-001D09F231B0.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/D0B5FB32-E457-E011-BD36-001D09F251FE.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/C8E9E69C-E557-E011-9593-001D09F28F1B.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/C86402C8-E257-E011-8E23-003048F1182E.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/C840505C-9258-E011-AAEF-0030487C7392.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/C60C95AE-FA57-E011-B848-003048F117EA.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/C2CAAC81-F657-E011-B654-003048F024FE.root',
        '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/C2B9F934-E457-E011-BF40-001D09F26509.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/AE27C23C-0558-E011-A023-003048F024E0.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/8ABFC9A1-E357-E011-9D79-001D09F2426D.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/8084D411-E057-E011-99F6-0030487CD6D2.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/7E3522F0-F057-E011-BFFE-0030487A18F2.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/7AD5E6E1-E957-E011-9309-00304879FBB2.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/7A3C5880-DC57-E011-B98D-001617C3B654.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/74CB59FA-EB57-E011-A988-003048F118C4.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/70DF7171-E857-E011-A6F3-001D09F2516D.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/706ED316-EE57-E011-A3F1-000423D996C8.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/685D1571-E857-E011-9B63-001D09F241F0.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/6040A49F-D757-E011-B350-001D09F23174.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/5AD8CB86-DC57-E011-89C0-003048D2C1C4.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/52E1F813-E057-E011-84F4-001617C3B79A.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/525E0BA4-DE57-E011-A5E1-0030487C90C2.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/4CF9F2C9-F557-E011-912F-003048F11C5C.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/4675C9A9-EC57-E011-ACF1-001D09F252DA.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/2AF8E934-E457-E011-B335-001D09F253D4.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/18EE38AE-EC57-E011-8723-001D09F25109.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/12E7CAAE-F357-E011-9086-003048F11C5C.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/0CD60D2B-DD57-E011-9B38-0030487A322E.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/06A08BFC-D857-E011-980D-003048D2BED6.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/040B519A-D057-E011-848A-0030487CD77E.root',
      '/store/data/Run2011A/SingleMu/RECO/PromptReco-v1/000/161/312/00DC8253-E657-E011-8805-0019DB2F3F9A.root'


#        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/D00ECD06-FA57-E011-B670-003048F1BF66.root',
#        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/B4AA6B6D-E857-E011-A081-001617C3B76A.root',
#        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/9E43C5BB-0858-E011-9400-0030487C6A66.root',
#        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/9483279C-E557-E011-B828-001D09F291D2.root',
#        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/5A98F600-FA57-E011-88EE-003048F11C5C.root',
#        '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v1/000/161/312/286081CA-E257-E011-8E2D-001D09F253D4.root'

        ),
                            secondaryFileNames = cms.untracked.vstring(

        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/FADB8906-FD55-E011-996D-000423D996C8.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/ECF250B9-E355-E011-934E-001617C3B654.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/E2F8F27A-E855-E011-99D6-0030487CD6D8.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/D86F1258-0656-E011-B0E8-001617E30D0A.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/D4D1E804-CB55-E011-8FA0-001D09F24489.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/CE271E3F-F755-E011-8EEB-003048F11DE2.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/C4C77CB7-D055-E011-9D00-001617C3B6E2.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/C240EDFD-F255-E011-9DFE-0030487CBD0A.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/B299B93D-DA55-E011-AC86-003048F1BF68.root',
        '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/A671BFE4-D455-E011-BCD0-001617E30CC2.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/A6401605-CB55-E011-9EFF-001D09F2AD7F.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/A281BC93-EF55-E011-BFD7-003048F024F6.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/9E56EEA3-D755-E011-BD75-003048F118E0.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/9A39C322-CD55-E011-A409-001D09F24DA8.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/907AFC68-E455-E011-A803-0030487A3C92.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/8A2B78F4-F055-E011-92B6-0030487CD6F2.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/84965A70-D355-E011-B726-001617E30D52.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/7C320A5D-E455-E011-8C24-001617C3B654.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/72A96101-E555-E011-958D-0030487C90C2.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/6E1A14D6-CD55-E011-AC69-001D09F2960F.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/6ACF2D85-F955-E011-8380-003048F1110E.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/64A61B76-F455-E011-8B99-003048F024FA.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/5EC134C4-DC55-E011-832F-000423D98B6C.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/5AAF5E00-FB55-E011-B0A5-001617DC1F70.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/5A25CEB2-E555-E011-9F12-003048F118C2.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/48E78C36-E955-E011-B70C-0030487C6062.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/4029CFAD-FF55-E011-8CBA-003048F117B4.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/383E6175-DA55-E011-A7F0-0030487CD162.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/2A98DBF5-CF55-E011-AF01-001617E30CC2.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/246606D5-CB55-E011-BC5A-0016177CA778.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/1226EE4C-EB55-E011-914F-001617DBD5AC.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/0CB4B104-E355-E011-A201-0030487C60AE.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/08824169-ED55-E011-9813-001617C3B65A.root',
      '/store/data/Run2011A/SingleMu/RAW/v1/000/161/312/06FEE2F0-D655-E011-9C44-001D09F2B30B.root'


#        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/B49B80DE-FA55-E011-89A5-001D09F2527B.root',
#        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/9CAED4DC-EE55-E011-92E0-0030487CD6E6.root',
#        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/98352641-0656-E011-A1F2-001617DBD472.root',
#        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/86618229-DA55-E011-A93E-003048F118DE.root',
#        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/48B3C7E4-CF55-E011-8551-000423D9A2AE.root',
#        '/store/data/Run2011A/SingleElectron/RAW/v1/000/161/312/30FA984D-E455-E011-920E-003048F11C5C.root'
        )
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("Configuration.Skimming.PDWG_TauSkim_cff")
process.tauFilter = cms.Path(process.tauSkimSequence)

process.outputCsTau = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('CS_Tau')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('tauFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Tau_2011.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputCsTau
)
