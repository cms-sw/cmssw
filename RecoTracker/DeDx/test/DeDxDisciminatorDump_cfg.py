import FWCore.ParameterSet.Config as cms

process = cms.Process("DEDX")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(     threshold = cms.untracked.string('ERROR')    ),
    destinations = cms.untracked.vstring('cout')
)


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_P_V32::All'

process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')


process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(132440),
    lastValue  = cms.uint64(132440)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('Data7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

process.dedxDiscrimDump               = cms.EDProducer("DeDxDiscriminatorDumpFromDB",
    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    HistoFile          = cms.untracked.string("Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.root"),
)

process.p        = cms.Path(process.dedxDiscrimDump)



