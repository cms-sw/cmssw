import FWCore.ParameterSet.Config as cms

isMC = True

process = cms.Process('RERECO')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

if (isMC):
    process.GlobalTag.globaltag ='START38_V10::All'
else:
    process.GlobalTag.globaltag ='GR_R_38X_V11::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Input source
if(isMC):
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('/store/mc/Summer10/TTbar-mcatnlo/GEN-SIM-RECO/START37_V5_S09-v1/0000/0268EFE9-DC85-DF11-ABC0-002618943902.root',
                                                                  '/store/mc/Summer10/TTbar-mcatnlo/GEN-SIM-RECO/START37_V5_S09-v1/0000/02C33E82-EA85-DF11-A055-0026189437FE.root',
                                                                  '/store/mc/Summer10/TTbar-mcatnlo/GEN-SIM-RECO/START37_V5_S09-v1/0000/02CCE997-E985-DF11-971D-00261894389E.root',
                                                                  '/store/mc/Summer10/TTbar-mcatnlo/GEN-SIM-RECO/START37_V5_S09-v1/0000/0425AE78-E985-DF11-9510-003048678A80.root',
                                                                  '/store/mc/Summer10/TTbar-mcatnlo/GEN-SIM-RECO/START37_V5_S09-v1/0000/06226586-DA85-DF11-8F5B-0026189438B5.root')
                                )
else:
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('/store/data/Commissioning10/MinimumBias/RECO/Apr20ReReco-v1/0160/00064D6A-8D4C-DF11-B7C4-002618943829.root',
                                                                  '/store/data/Commissioning10/MinimumBias/RECO/Apr20ReReco-v1/0160/0065C473-8F4C-DF11-A129-003048679080.root'
                                                                  )
                                )

# Output definition
process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('output.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    )
)
process.output.outputCommands.append('keep *_*_*_RERECO')

# Other statements
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# modify severity level computer
flag = 'HBHEIsolatedNoise'
foundLevel10 = False
for i in range(len(process.hcalRecAlgos.SeverityLevels)):
    level=process.hcalRecAlgos.SeverityLevels[i].Level.value()
    flags=process.hcalRecAlgos.SeverityLevels[i].RecHitFlags.value()
    if level!=10 and flag in flags: # remove flag if it's in a level != 10
        flags.remove(flag)
        process.hcalRecAlgos.SeverityLevels[i].RecHitFlags=flags
    elif level==10 and flags==['']: # add flag to level 10
        flags=[flag]
        process.hcalRecAlgos.SeverityLevels[i].RecHitFlags=flags
        foundLevel10 = True
    elif level==10 and flag not in flags:
        flags.append(flag)
        process.hcalRecAlgos.SeverityLevels[i].RecHitFlags=flags
        foundLevel10 = True
if not foundLevel10:
    process.hcalRecAlgos.SeverityLevels.append(cms.PSet(Level=cms.int32(10),
                                                        RecHitFlags=cms.vstring(flag),
                                                        ChannelStatus=cms.vstring("")))

# modify reconstruction sequence
process.hbhereflag = process.hbhereco.clone()
process.hbhereflag.hbheInput = 'hbhereco'
process.towerMaker.hbheInput = 'hbhereflag'
process.towerMakerWithHO.hbheInput = 'hbhereflag'
process.hcalnoise.recHitCollName = 'hbhereflag'
process.hcalnoise.fillDigis = False
process.rereco_step = cms.Path(process.hbhereflag*process.caloTowersRec*(process.recoJets*process.recoJetIds+process.recoTrackJets)*process.recoJetAssociations*process.metrecoPlusHCALNoise)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.rereco_step,
                                process.out_step)
