## --
## Start with pre-defined skeleton process
## --
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## ... and modify it according to the needs:
process.source.fileNames = [ '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/06ADBD4E-0E96-E011-A2FE-003048F1C836.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/06F5A72B-E195-E011-A988-001D09F231C9.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/08B4EC25-1396-E011-A4CA-003048D2C174.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/186E812D-0C96-E011-8D83-003048F11CF0.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/18FBAA04-E395-E011-AD96-001D09F24664.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/1C954EEF-1A96-E011-A306-0030487C6A66.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/2084822A-1396-E011-BCC7-003048D2BDD8.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/32C97086-1296-E011-8588-003048F11C5C.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/4620F471-1096-E011-9D47-0030487CD6D8.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/462E5C46-E495-E011-AB68-001D09F252E9.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/4A0AC5B1-DF95-E011-93C7-001D09F25041.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/5655A998-E595-E011-BA75-001D09F23D1D.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/56DA4E81-DC95-E011-835D-0030487A1990.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/5AB9C521-2D96-E011-8648-003048F117B6.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/6422B81F-D995-E011-BA28-003048F1C832.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/74B30E2A-E195-E011-A2A2-001D09F29538.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/7A81AEEF-1A96-E011-A1F6-0030487A18F2.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/8288E509-E395-E011-93F2-001D09F2960F.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/867A73F4-E295-E011-8598-003048F11CF0.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/86B21B4E-1C96-E011-9549-003048F024DE.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/8E25CC65-1096-E011-ADBB-003048F1BF68.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/9A59783C-E495-E011-A8B3-001D09F24600.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/9C51D93B-E495-E011-AA78-0030487CD184.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/9E75657D-DC95-E011-A8FF-001D09F290BF.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/A2B4BF06-E395-E011-8C8B-003048CFB40C.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/AE5CAA73-DC95-E011-BC78-003048F24A04.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/B40C9D36-0C96-E011-9584-003048D2C0F0.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/B48976EF-1A96-E011-96CD-0030487A195C.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/BCDFC3B7-ED95-E011-94F8-0030487CD7E0.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/BE64844F-0E96-E011-8B0E-003048F1C58C.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/C06E873D-E495-E011-8FF1-0019B9F72BAA.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/DE1F0652-0E96-E011-A96C-003048F11C58.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/E0030F46-0896-E011-95AC-003048F117B4.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/E2BD34C9-E995-E011-9B50-00304879EDEA.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/E4C1DE51-1596-E011-8B15-003048673374.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/EAA9643C-E495-E011-9D68-001D09F23A34.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/F61CA81C-0A96-E011-958B-003048D2C01E.root'
                           , '/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/166/841/F6421032-0C96-E011-977B-003048F1C420.root'
                           ]
process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange( '166841:1-166841:845'
                                                                   , '166841:851-166841:876'
                                                                   , '166841:882-166841:977'
                                                                   , '166841:984-166841:984'
                                                                   , '166841:988-166841:992'
                                                                   , '166841:998-166841:1015'
                                                                   )
# use the correct conditions
process.GlobalTag.globaltag = autoCond[ 'com10' ]
# use a sufficient number of events
process.maxEvents.input = 1000
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
process.patTrigger.exludeCollections = cms.vstring( "hltL1extraParticles*" )
# update event content to save
switchOnTrigger( process, sequence = 'p' )       # called once more to update the event content according to the changed parameters!!!
