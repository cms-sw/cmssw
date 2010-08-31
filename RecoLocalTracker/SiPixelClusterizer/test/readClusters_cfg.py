#
# Last update: new version for python
#
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("cluTest")


import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.hltfilter = hlt.hltHighLevel.clone(
# Min-Bias
# HLT_L1_BscMinBiasOR_BptxPlusORMinus, HLT_L1Tech_BSC_minBias, 	HLT_L1Tech_BSC_halo_forPhysicsBackground	
#    HLTPaths = ['HLT_L1Tech_BSC_minBias'],
#    HLTPaths = ['HLT_L1Tech_BSC_minBias_OR'],
#    HLTPaths = ['HLT_L1Tech_BSC_halo_forPhysicsBackground'],
#    HLTPaths = ['HLT_L1Tech_BSC_HighMultiplicity'],
#    HLTPaths = ['HLT_L1_BPTX'],
#    HLTPaths = ['HLT_ZeroBias'],
#    HLTPaths = ['HLT_L1_BPTX_MinusOnly','HLT_L1_BPTX_PlusOnly'],
# Commissioning: HLT_L1_BptxXOR_BscMinBiasOR
    HLTPaths = ['HLT_L1_BptxXOR_BscMinBiasOR'],
# Zero-Bias : HLT_L1_BPTX, HLT_L1_BPTX_PlusOnly, HLT_L1_BPTX_MinusOnly, HLT_ZeroBias
#    HLTPaths = ['HLT_L1_BPTX','HLT_ZeroBias','HLT_L1_BPTX_MinusOnly','HLT_L1_BPTX_PlusOnly'],
#    HLTPaths = ['p*'],
#    HLTPaths = ['path_?'],
    andOr = True,  # False = and, True=or
    throw = False
    )

# to select PhysicsBit
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelClusters'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(    
#    'file:/tmp/dkotlins/digis.root'
#
# 136100
# "/store/data/Run2010A/MinimumBias/RECO/v2/000/136/100/0422C984-FC67-DF11-AE5B-000423D98634.root",
# "/store/data/Run2010A/MinimumBias/RECO/v2/000/136/100/1494E118-DD67-DF11-A2AA-001D09F24DDA.root",
# "/store/data/Run2010A/MinimumBias/RECO/v2/000/136/100/160083A3-CA67-DF11-BB2B-001D09F2A690.root",
# "/store/data/Run2010A/MinimumBias/RECO/v2/000/136/100/826E7966-FA67-DF11-BED1-0030487C6A66.root",
# "/store/data/Run2010A/MinimumBias/RECO/v2/000/136/100/EE6076DC-F667-DF11-83BD-000423D98BC4.root"

# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/3AAD4598-A081-DF11-A559-0030487CD812.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/4AF13298-A081-DF11-8A59-0030487CBD0A.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/5436520B-A681-DF11-BACF-003048F117B6.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/56F6EEAA-A781-DF11-A11D-003048F117EA.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/74716012-A481-DF11-AF88-001617C3B710.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/7A22A170-A381-DF11-B6B4-001D09F24682.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/88B4B585-AC81-DF11-8020-001D09F2305C.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/A8B9BBD8-A681-DF11-A0D7-001617C3B6DC.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/AED219A3-A081-DF11-BF25-003048F117B6.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/B6178E70-A881-DF11-B781-0019B9F70607.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/C6F8A2A9-9B81-DF11-817C-0030487C60AE.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/C8AF4630-9F81-DF11-8B3B-001D09F2514F.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/CC7BAE97-A081-DF11-98A6-003048F1BF66.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/DC11FDBC-A981-DF11-81CD-001D09F253FC.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/E26582B7-A281-DF11-A7F8-0019DB2F3F9A.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/138/747/F818964A-A181-DF11-B781-000423D9A212.root"

# mb
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/139/239/64172C3E-2686-DF11-AE6A-000423D94494.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/139/239/CCAA4C3B-2686-DF11-8452-000423D98B6C.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/139/239/2649F9B7-2286-DF11-BB4E-001617E30D40.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/139/239/16501807-2286-DF11-8215-000423D98950.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/139/239/B6B6BED6-1D86-DF11-876F-000423D996C8.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/139/239/5E48DC09-1B86-DF11-BE5C-0030487C90D4.root"

# zb
# "/store/data/Run2010A/ZeroBias/RECO/v4/000/139/239/42F3662E-0A86-DF11-9B7D-0016177CA778.root",
# "/store/data/Run2010A/ZeroBias/RECO/v4/000/139/239/441A4944-1186-DF11-984F-000423D9A2AE.root",
# "/store/data/Run2010A/ZeroBias/RECO/v4/000/139/239/6674E18B-1E86-DF11-8241-003048F1C58C.root",
# "/store/data/Run2010A/ZeroBias/RECO/v4/000/139/239/9ABFCCEF-1886-DF11-91E7-000423D9A2AE.root"

# comissioning
# "/store/data/Run2010A/Commissioning/RECO/v4/000/139/239/086F1609-1B86-DF11-AFE0-0030487CD77E.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/139/239/6A53E2CD-1686-DF11-AD2B-000423D9A2AE.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/139/239/88FDF981-4686-DF11-A4B5-000423D94E70.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/139/239/923E6108-2286-DF11-8B35-001617E30D12.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/139/239/BEEFA09E-1986-DF11-B54B-000423D9A212.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/139/239/EE7C0308-2286-DF11-A986-000423D9A212.root"

# r142187
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/187/10DE15E4-C49E-DF11-B4CB-000423D98B6C.root"
# "/store/data/Run2010A/EG/RECO/v4/000/142/187/160DF95A-C79E-DF11-A742-001D09F2841C.root"
# "/store/data/Run2010A/JetMET/RECO/v4/000/142/187/1E464E9C-DB9E-DF11-8FDC-001617DBD556.root"
# "/store/data/Run2010A/Mu/RECO/v4/000/142/187/0C19A1AC-C69E-DF11-ACC1-001D09F24934.root"

# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/187/64FF1C23-BE9E-DF11-8C96-0030487C90C4.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/187/02A0AEE3-BE9E-DF11-AB90-001D09F29597.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/187/6E896A0A-C39E-DF11-A685-001D09F2438A.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/187/B2E8120A-C39E-DF11-A2C2-003048D2C0F0.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/187/12620AB0-C49E-DF11-95FD-003048F024FE.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/187/3EACD4B1-C89E-DF11-AB4D-003048F118D4.root"

# run 142663  
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/663/201CB31A-ACA4-DF11-A0C4-0030487D05B0.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/663/265CF3CA-84A4-DF11-9407-001617C3B654.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/663/927A182E-86A4-DF11-9639-00304879FC6C.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/663/98899FCA-89A4-DF11-9DB1-00304879EDEA.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/663/A810AAC9-89A4-DF11-9A71-003048F117EA.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/663/C662E6AB-82A4-DF11-A41D-003048F024DC.root"

# run 142933
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/933/CEF1C984-72A7-DF11-9179-0030487CAF5E.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/933/86D85052-75A7-DF11-AB02-001D09F290BF.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/933/D084C472-77A7-DF11-B79B-0030487C8CB6.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/933/3C4471DF-78A7-DF11-A113-001D09F2424A.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/933/F4B33291-79A7-DF11-A581-0030487C6062.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/142/933/B2444C92-79A7-DF11-B18D-0030487C912E.root"

# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/FABB1503-76A7-DF11-8828-00304879FC6C.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/B8B73E0C-76A7-DF11-A417-0030487C7392.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/4650F001-76A7-DF11-AF73-003048D3756A.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/44869E73-77A7-DF11-B134-0030487C8CBE.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/36FEDF51-75A7-DF11-AF76-001D09F2516D.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/6CCF7603-76A7-DF11-B41B-003048D2C01A.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/C4754567-7CA7-DF11-81AF-000423D9A2AE.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/A0222FEE-78A7-DF11-A396-003048F11C28.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/9813AE80-7EA7-DF11-A78D-003048F1182E.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/142/933/824A8004-7BA7-DF11-AD28-003048F1C836.root"

# run 143657, comm
# "/store/data/Run2010A/Commissioning/RECO/v4/000/143/657/3016BDA7-6AAE-DF11-A15F-001617E30E28.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/143/657/486E3EAC-6AAE-DF11-A45D-003048D2BB90.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/143/657/4E97F5B4-6CAE-DF11-884D-001617C3B654.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/143/657/66409FB3-6CAE-DF11-9219-001D09F232B9.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/143/657/CC9CE620-6EAE-DF11-BC95-001D09F251CC.root",
# "/store/data/Run2010A/Commissioning/RECO/v4/000/143/657/EC773EDF-6BAE-DF11-AB62-001617C3B76E.root"

# cosmics
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/657/20CF5788-7DAE-DF11-932C-003048D3756A.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/657/CA92481C-81AE-DF11-8A36-001D09F24489.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/657/EE5EB6C7-81AE-DF11-A078-001D09F28F25.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/657/FA898C87-7DAE-DF11-BEDD-000423D94494.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/657/FC3AFA2E-83AE-DF11-93BB-001D09F24600.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/657/EC3E5A7A-89AE-DF11-9428-003048D2BC42.root"

# mb
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/265FFC5B-74AE-DF11-84F3-001617E30CD4.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/401792C0-75AE-DF11-9FAD-00304879FA4C.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/5C185B5E-74AE-DF11-9E16-000423D996C8.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/703C315F-74AE-DF11-B4D5-003048D374F2.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/763F6E7B-76AE-DF11-A227-00304879FC6C.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/7881DA5E-74AE-DF11-A763-0030486730C6.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/8299E25C-74AE-DF11-8C54-001617E30E2C.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/86E84169-74AE-DF11-B8B9-000423D987E0.root",
# "/store/data/Run2010A/MinimumBias/RECO/v4/000/143/657/C82420A8-74AE-DF11-AE8B-001617DBD556.root"

# "/store/data/Run2010A/EG/RECO/v4/000/143/657/082AC006-8BAE-DF11-A7BA-003048F1C424.root"
# "/store/data/Run2010A/JetMET/RECO/v4/000/143/657/8EBC6072-93AE-DF11-A835-0019DB29C5FC.root" 
 "/store/data/Run2010A/Mu/RECO/v4/000/143/657/EAFB311D-81AE-DF11-8121-00304879EE3E.root"

# cosmics 143749
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/749/CE9D2CA8-7EAF-DF11-AE27-003048F118D4.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/749/CCEE1EA8-7EAF-DF11-B490-003048F118DE.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/749/7EDABAC1-80AF-DF11-87A3-003048F117EC.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/749/6AE3B4C3-80AF-DF11-96B9-003048F0258C.root",
# "/store/data/Run2010A/Cosmics/RECO/v4/000/143/749/0EDC172C-82AF-DF11-9DBF-003048F118D4.root"

 

  )

)

# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124230:26-124230:9999','124030:2-124030:9999')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('133450:1-133450:657')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('139239:160-139239:213')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('142187:207-142187:9999')

process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('143657:211-143657:9999')


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
# process.GlobalTag.globaltag = 'GR10_P_V5::All'
process.GlobalTag.globaltag = 'GR10_P_V4::All'
# OK for 2009 LHC data
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

process.analysis = cms.EDAnalyzer("ReadPixClusters",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("siPixelClusters"),
)

#process.p = cms.Path(process.hltPhysicsDeclared*process.hltfilter*process.analysis)
process.p = cms.Path(process.hltPhysicsDeclared*process.analysis)
#process.p = cms.Path(process.analysis)


# define an EndPath to analyze all other path results
#process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
#    HLTriggerResults = cms.InputTag( 'TriggerResults','','' )
#)
#process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
