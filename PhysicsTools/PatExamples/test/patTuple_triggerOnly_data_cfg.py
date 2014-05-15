## --
## Start with pre-defined skeleton process
## --
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## ... and modify it according to the needs:
process.source.fileNames = [ '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/0609B981-BFB8-E111-9C19-5404A63886EC.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/0C06CE57-C2B8-E111-BA54-001D09F242EF.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/0C97DC6D-CBB8-E111-B39F-00237DDC5BBC.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/1414DF29-B0B8-E111-995D-003048F1C836.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/149AECC7-DDB8-E111-B64C-BCAEC5329713.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/14E56528-BEB8-E111-B538-E0CB4E4408E3.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/169CC587-D7B8-E111-B931-BCAEC532971F.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/1A1965B4-DFB8-E111-9AF1-003048D2C108.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/1A769685-D0B8-E111-A494-BCAEC5329713.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/1CF16C1C-FAB8-E111-8D9E-0019B9F70468.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/1E3B479C-E5B8-E111-BD66-001D09F25041.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/22024FE9-C5B8-E111-AF49-003048F117EC.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/281CC457-C2B8-E111-BA01-003048D37456.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/28CF4557-C0B8-E111-9D2C-001D09F2305C.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/2C720413-D6B8-E111-9B5F-001D09F24D67.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/2CA0AF11-DBB8-E111-B2CC-001D09F25041.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/3007DC96-B4B8-E111-B97E-5404A638869E.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/30E73932-CCB8-E111-B4CA-5404A63886B7.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/34A1AB2E-BEB8-E111-802A-BCAEC5329708.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/38E12248-DAB8-E111-B6A0-5404A63886A8.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/3AD35B23-C3B8-E111-95C1-001D09F291D2.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/3C795B12-B9B8-E111-84DD-001D09F2527B.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/3C866812-D6B8-E111-85F5-003048D2BB58.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/3EA1CF8D-A9B8-E111-A97D-BCAEC5329717.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/4ADA49C7-E2B8-E111-8B80-5404A63886B6.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/4C3F6AC9-D6B8-E111-AB16-5404A63886BD.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/4C823F44-B7B8-E111-BA62-BCAEC532970F.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/4E14C168-AFB8-E111-B536-BCAEC518FF52.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/54C4B067-B5B8-E111-B9DA-BCAEC518FF30.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/60205292-C2B8-E111-A735-001D09F291D2.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/6066780D-DBB8-E111-BDDA-003048F024F6.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/60757E04-ACB8-E111-B337-001D09F24DA8.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/607B4D00-E0B8-E111-8C47-001D09F2A49C.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/6639F43D-CEB8-E111-8EA7-0015C5FDE067.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/6A36C40F-B3B8-E111-961F-5404A640A643.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/6ACC51E9-D3B8-E111-A9C8-BCAEC518FF6E.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/6C659AC9-E2B8-E111-A824-003048D37560.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/6C9094E2-CCB8-E111-B179-003048D375AA.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/70060AF9-DFB8-E111-84DA-E0CB4E55367F.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/7027A1F7-C9B8-E111-9C76-E0CB4E4408C4.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/7075EC36-D3B8-E111-B01B-001D09F24303.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/72634C88-D7B8-E111-A414-5404A63886EB.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/780F56E6-AEB8-E111-8AEF-003048D373F6.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/781EA596-B4B8-E111-A073-0025901D5DB2.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/804E595A-E1B8-E111-B42F-003048D2C0F0.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/820C5E88-D0B8-E111-8FFF-0025901D6272.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/826AA5B9-DBB8-E111-B321-003048D3751E.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/86F0406D-CBB8-E111-A7A3-00237DDC5C24.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/8818E2C8-D6B8-E111-89BB-BCAEC518FF5F.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/8A940770-DCB8-E111-A662-003048D373AE.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/8ADB95DC-CCB8-E111-8EC8-003048D2BDD8.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/8AEE1E55-C2B8-E111-BD20-003048F024FA.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/90E7342A-B0B8-E111-A1B2-003048F118D2.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/92A296B0-A6B8-E111-B117-485B3977172C.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/92C4A7A4-D9B8-E111-B265-5404A6388694.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/92CF834D-AEB8-E111-AF07-001D09F25460.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/9E72BEFB-BBB8-E111-846D-003048D2BBF0.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/9EFAD8BC-BCB8-E111-A0CB-003048F1C420.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/A2527DCB-A3B8-E111-BEC3-E0CB4E4408E7.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/A4D6AE2A-B0B8-E111-81D2-003048F1BF66.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/A6D1F0A3-D9B8-E111-9F62-0025901D5DEE.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/A8FFFDCF-C0B8-E111-89A7-5404A63886CB.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/AE932793-C6B8-E111-BB78-0030486780AC.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/B0624C2B-C5B8-E111-BB6C-003048F118D4.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/B49BCB59-D5B8-E111-8F3F-5404A640A639.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/BE5931D5-C7B8-E111-87C0-001D09F295A1.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/C09B5259-C0B8-E111-902B-003048D2BEAA.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/C8916851-E6B8-E111-BAC8-001D09F2305C.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/D4C0DB30-CCB8-E111-8A1F-5404A63886D6.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/D8100D99-E3B8-E111-8E03-5404A640A63D.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/DA0729DD-E9B8-E111-987E-001D09F2525D.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/DA1E9C37-D3B8-E111-9068-001D09F242EF.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/DE50ADD5-CAB8-E111-B2E1-001D09F2AD4D.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/E036F03D-ABB8-E111-BF04-0025901D627C.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/E2B0FEDC-CCB8-E111-937C-001D09F28D54.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/E44C4832-CCB8-E111-84D3-BCAEC518FF76.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/E4970EEB-C5B8-E111-8443-0025901D624A.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/E6DD24C5-DDB8-E111-8C0D-BCAEC5329709.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/EA410BE6-C8B8-E111-B1F2-5404A63886EF.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/EC6EC768-B5B8-E111-9F2D-5404A63886CE.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/EEC2A387-D7B8-E111-8D67-BCAEC518FF63.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/F0583E36-CEB8-E111-9E26-001D09F29114.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/F2B867D6-CAB8-E111-8014-001D09F2A49C.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/F2E649EA-D3B8-E111-B7FA-5404A63886EC.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/F8C5AFFD-BBB8-E111-AEBC-002481E0D958.root'
                           , '/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/196/364/FABD8E12-E7B8-E111-B9DE-BCAEC518FF44.root'
                           ]
process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange( '196364:1-196364:93'
                                                                   , '196364:96-196364:136'
                                                                   , '196364:139-196364:365'
                                                                   , '196364:368-196364:380'
                                                                   , '196364:382-196364:601'
                                                                   , '196364:603-196364:795'
                                                                   , '196364:798-196364:884'
                                                                   , '196364:887-196364:1196'
                                                                   , '196364:1199-196364:1200'
                                                                   , '196364:1203-196364:1302'
                                                                   )
# use the correct conditions
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:com10_7E33v2' ) # 2012B
# use a sufficient number of events
process.maxEvents.input = 25000
# have a proper output file name
process.out.fileName = 'patTuple_triggerOnly_data.root'

## ---
## Define the path as empty skeleton
## ---
process.p = cms.Path(
)

### ===========
### PAT trigger
### ===========

## --
## Switch on
## --
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger( process, sequence = 'p' ) # overwrite sequence default "patDefaultSequence", since it is not used in any path

## --
## Modify configuration according to your needs
## --
# add L1 algorithms' collection
process.patTrigger.addL1Algos     = cms.bool( True ) # default: 'False'
# add L1 objects collection-wise (latest collection)
process.patTrigger.l1ExtraMu      = cms.InputTag( 'l1extraParticles', ''            )
process.patTrigger.l1ExtraNoIsoEG = cms.InputTag( 'l1extraParticles', 'NonIsolated' )
process.patTrigger.l1ExtraIsoEG   = cms.InputTag( 'l1extraParticles', 'Isolated'    )
process.patTrigger.l1ExtraCenJet  = cms.InputTag( 'l1extraParticles', 'Central'     )
process.patTrigger.l1ExtraForJet  = cms.InputTag( 'l1extraParticles', 'Forward'     )
process.patTrigger.l1ExtraTauJet  = cms.InputTag( 'l1extraParticles', 'Tau'         )
process.patTrigger.l1ExtraETM     = cms.InputTag( 'l1extraParticles', 'MET'         )
process.patTrigger.l1ExtraHTM     = cms.InputTag( 'l1extraParticles', 'MHT'         )
# save references to original L1 objects
process.patTrigger.saveL1Refs = cms.bool( True ) # default: 'False'
# exclude HLT copies of L1 objects
process.patTrigger.exludeCollections = cms.vstring( 'hltL1extraParticles*' )
# update event content to save
switchOnTrigger( process, sequence = 'p' )       # called once more to update the event content according to the changed parameters!!!
