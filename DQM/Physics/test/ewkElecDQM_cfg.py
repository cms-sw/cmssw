import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkElecDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Physics/EWK/Elec')

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(-1)
    input = cms.untracked.int32(5000)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
# RelVal Zee 315
    '/store/relval/CMSSW_3_1_5/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0001/D484792A-D8D1-DE11-AD79-00261894388B.root',
    '/store/relval/CMSSW_3_1_5/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0001/CAC4B678-49D1-DE11-8F6D-002618943981.root',
    '/store/relval/CMSSW_3_1_5/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0001/56101012-48D1-DE11-B885-00261894394A.root'
        
# RelVal Zee, Zmm, We, Wm - 311 BAD ELECTRONS!
#       '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/FC71916C-756B-DE11-8631-000423D94700.root',
#       '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/B672A1C5-746B-DE11-93A8-000423D944F0.root',
#       '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/5A0F871C-756B-DE11-BFE1-001D09F2545B.root',
#       '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/44507D17-D66B-DE11-A165-000423D94524.root',
##        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/C8CEE598-CB6B-DE11-871F-001D09F2905B.root',
##        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/642F8176-C96B-DE11-9D10-000423D98BE8.root',
##        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/46AA6A11-D46B-DE11-A614-001D09F25438.root',
##        '/store/relval/CMSSW_3_1_1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/443BC1DD-CC6B-DE11-804C-000423D98EC4.root'
#       '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0003/08DBA390-F66B-DE11-A7EC-0030487A3C9A.root',
#       '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/88B35291-C16B-DE11-B9AB-000423D98F98.root',
#       '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/7A84CA74-BC6B-DE11-B2E1-000423D6CA6E.root',
#       '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/50D84AA9-C46B-DE11-8209-000423D98BC4.root',
#       '/store/relval/CMSSW_3_1_1/RelValWE/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/1E5726C8-E16B-DE11-A98B-000423D99AAE.root',
##        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0003/8AC3E97D-EF6B-DE11-ADBA-001D09F29619.root',
##        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/EA593FE4-E26B-DE11-8173-001D09F2438A.root',
##        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/9270F55B-E26B-DE11-994E-001D09F2AF1E.root',
##        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/8E5D0675-E36B-DE11-8F71-001D09F242EF.root'

#        'file:/data4/Wmunu-Summer09-MC_31X_V2_preproduction_311-v1/0011/F4C91F77-766D-DE11-981F-00163E1124E7.root'

# MinBias real data!
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/196/3C9489A4-B5E8-DE11-A475-001D09F2A465.root',
    #'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/188/34641279-B5E8-DE11-A475-001D09F2910A.root',

# Real data, run 124120
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/F08F782B-77E8-DE11-B1FC-0019B9F72BFF.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/EE9412FD-80E8-DE11-9FDD-000423D94908.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/7C9741F5-78E8-DE11-8E69-001D09F2AD84.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/44255E49-80E8-DE11-B6DB-000423D991F0.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/3C02A810-7CE8-DE11-BB51-003048D375AA.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04F15557-7BE8-DE11-8A41-003048D2C1C4.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root'


    )
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('ERROR')
    )
)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkElecDQM+process.dqmSaver)

