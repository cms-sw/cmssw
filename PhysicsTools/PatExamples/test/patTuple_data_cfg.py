# This is an example PAT configuration showing the usage of PAT on minbias data

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

from PhysicsTools.PatAlgos.tools.coreTools import *

## global tag for data
process.GlobalTag.globaltag = cms.string('GR_R_38X_V14::All')

# add pf met
from PhysicsTools.PatAlgos.tools.metTools import *
removeMCMatching(process, ['All'])
addPfMET(process, 'PF')


# add PF jets
from PhysicsTools.PatAlgos.tools.jetTools import *
addJetCollection(process,cms.InputTag('ak5PFJets'),
                 'AK5', 'PF',
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = ('AK5PF', cms.vstring(['L2Relative', 'L3Absolute', 'L2L3Residual'])),
                 doType1MET   = False,
                 doL1Cleaning = False,                 
                 doL1Counters = False,
                 genJetCollection=cms.InputTag("ak5GenJets"),
                 doJetID      = False
                 )

# require physics declared
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

# require scraping filter
process.scrapingVeto = cms.EDFilter("FilterOutScraping",
                                    applyfilter = cms.untracked.bool(True),
                                    debugOn = cms.untracked.bool(False),
                                    numtrack = cms.untracked.uint32(10),
                                    thresh = cms.untracked.double(0.2)
                                    )


# switch on PAT trigger
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger( process )

process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                           minimumNDOF = cms.uint32(4) ,
                                           maxAbsZ = cms.double(15), 
                                           maxd0 = cms.double(2) 
                                           )

# Select jets
process.selectedPatJets.cut = cms.string('pt > 10')
process.selectedPatJetsAK5PF.cut = cms.string('pt > 10')


# Add the files 
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

readFiles.extend( [
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/940B070B-DA97-DF11-8617-002618943926.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/9281D65A-DD97-DF11-A291-002618FDA265.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/88DA882B-DE97-DF11-BC66-00248C55CC97.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/888AA340-AE97-DF11-A4F6-003048678A78.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/86341F45-B897-DF11-B123-003048678B14.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/846FB748-B697-DF11-9424-002618FDA216.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/74E90F3D-AA97-DF11-B477-0026189438D4.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/74CBB955-DD97-DF11-8A29-00261894393A.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/74422470-B997-DF11-B5B6-002618943964.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/7435EB3D-AA97-DF11-9542-002618943949.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/72DDC354-DD97-DF11-901F-002618FDA211.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/72686F6C-E497-DF11-9426-002618943836.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/6A237350-E097-DF11-A811-00261894396E.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/688CC9A8-D497-DF11-A82A-002618943926.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/66BB531E-D597-DF11-9B8A-0026189438BA.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/62AC2C6B-A997-DF11-98A4-002618FDA210.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/6299FC2A-E197-DF11-8773-002618FDA265.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/6091AA48-B197-DF11-9BCF-00261894396A.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/5CC39069-B397-DF11-893F-003048678A78.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/58D9079B-A897-DF11-8B6B-003048678A78.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/563D9DA8-AD97-DF11-8CC7-002618943946.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/54127B46-B497-DF11-8758-00261894393C.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/52445FC5-AD97-DF11-B021-002618FDA216.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/520B2B2F-DE97-DF11-80B3-002618FDA287.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4CB31A7F-DE97-DF11-B046-002618FDA262.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4C589749-B697-DF11-8059-00261894396A.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4AB60F1F-D597-DF11-9AC0-00248C55CC97.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4A3AC46E-A797-DF11-BB72-003048678B5E.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4809E146-B397-DF11-94B3-003048678B14.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4217C067-AB97-DF11-B4B0-002618FDA21D.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/4035A012-AF97-DF11-91C1-003048678B5E.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/3A8F3B49-AC97-DF11-9E4B-002618943946.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/3A75E46D-DA97-DF11-B845-002618943836.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/367EDB45-B197-DF11-84A9-002618FDA216.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/32571795-A697-DF11-8320-00248C55CC40.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/2C935C68-AD97-DF11-9519-0026189438D4.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/264BE66D-B297-DF11-82B3-002618943860.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/261BF88D-E297-DF11-9F26-002618FDA287.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/247AEBA5-9D97-DF11-B5F4-00261894385D.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/22744443-AF97-DF11-8450-002618943964.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/1E62C747-B697-DF11-A581-002618943949.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/1CE7213F-D797-DF11-A0C9-00248C55CC97.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/1A368544-B897-DF11-A6DB-00261894393C.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/1A11F06D-B797-DF11-857F-002618943964.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/1898FD6F-AF97-DF11-AF4D-002354EF3BE1.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/162AD99E-E297-DF11-87E2-002618943985.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/161AF945-B397-DF11-9FAE-002618FDA216.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/14DB0EA6-A697-DF11-B754-0026189438D4.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/14507168-AD97-DF11-99B4-0030486792DE.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/1273FA44-B397-DF11-BFB6-002618943946.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/10B71087-AF97-DF11-83CC-003048678A78.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/10A5BF6E-B797-DF11-84A5-002618943915.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/0C3A4055-E297-DF11-8321-00248C55CC97.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/06C5A183-F097-DF11-885C-002618FDA211.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/00E19D5E-DB97-DF11-A4D3-002618943926.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/009B303F-E597-DF11-9FC6-002618943811.root',
'/store/data/Run2010A/JetMETTau/RECO/Jul23ReReco_PreProd_v1/0157/0003F63E-E597-DF11-94EB-002618FDA208.root',


        ] );
process.source.fileNames = readFiles

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

# let it run

#print
#print "============== Warning =============="
#print "technical trigger filter:    DISABLED"
#print "physics declare bit filter:  DISABLED"
#print "primary vertex filter:       DISABLED"

process.p = cms.Path(
    process.scrapingVeto*
    process.primaryVertexFilter*
    process.patDefaultSequence
    )

# rename output file
process.out.fileName = cms.untracked.string('reco_7TeV_380_pat.root')

# reduce verbosity
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

# process all the events
process.maxEvents.input = 1000
process.options.wantSummary = True

from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerEventContent
process.out.outputCommands = patEventContentNoCleaning
process.out.outputCommands += patTriggerEventContent
process.out.outputCommands += [ 'keep recoTracks_generalTracks_*_*',
                                'keep LumiSummary_lumiProducer_*_*']
