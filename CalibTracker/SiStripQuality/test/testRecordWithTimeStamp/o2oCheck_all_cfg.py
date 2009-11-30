import FWCore.ParameterSet.Config as cms

process = cms.Process("o2oCheck")

process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    ReaderAll = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
    ),
                                    destinations = cms.untracked.vstring('ReaderAll') #Reader.log, cout
                                    )

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(

     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/DE8A122E-F3D7-DE11-A042-001D09F295FB.root", #   1
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/747C6137-F0D7-DE11-BE6C-001D09F242EF.root", #   2
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/08A81438-F0D7-DE11-A4A4-001D09F2AF96.root", #   3
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/1027F039-F0D7-DE11-A914-001D09F2423B.root", #   4
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/74F85837-F0D7-DE11-BAFB-001D09F276CF.root", #   5
    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/CE59F836-F0D7-DE11-A699-001D09F25208.root",  #   6
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/FA3FAB35-F0D7-DE11-85F1-0019B9F72BAA.root", #   7
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/D28BDD39-F0D7-DE11-BAFA-001D09F244DE.root", #   8
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/F26E75F8-F0D7-DE11-BE2B-001D09F25109.root", #   9
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/E6CC7BAE-F1D7-DE11-B7AC-001D09F23174.root", #   10
     "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/F2437DF8-F0D7-DE11-A630-001D09F2423B.root", #   11
    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/048FBAB2-F1D7-DE11-BDA4-001D09F24600.root",  #   12
    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/9407EDAE-F1D7-DE11-AA5D-001D09F2546F.root"  #13
    
 #    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/CA3089AF-5ED8-DE11-90CF-001D09F2545B.root", #23
 #    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/245448AF-5ED8-DE11-9E67-001D09F2527B.root", #24
 #    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/2288321E-60D8-DE11-B57D-001D09F2AF1E.root", #25
 #    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/3C69B237-62D8-DE11-AF15-001D09F2960F.root", #37   
 #    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/D00EC34F-64D8-DE11-A6D4-001D09F28F25.root", #38   
 #    "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/14B88715-65D8-DE11-B113-000423D94A20.root"  #39   
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )


#------------------------------------------
# Load standard sequences.
#------------------------------------------
#process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
#process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT09_R_V8::All' 

process.load("CalibTracker/Configuration/Tracker_DependentRecords_forGlobalTag_cff")

process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("test") ),
    #cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
    #  cms.PSet( record = cms.string("RunInfoRcd"),           tag    = cms.string("") ),
    #cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
    #cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
    )

process.poolDBESSource = cms.ESSource(
    "PoolDBESSource",
    appendToDataLabel= cms.string("test"),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(messageLevel = cms.untracked.int32(2),
                            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                            ),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(cms.PSet(timetype = cms.untracked.string('timestamp'),
                               record = cms.string('SiStripDetVOffRcd'),
                               tag = cms.string('SiStripDetVOff_Fake_31X')
                               )
    )
    )


process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                              dataLabel = cms.untracked.string(""),
                              TkMapFileName = cms.untracked.string(""),
                              SaveTkHistoMap = cms.untracked.bool(False)
)

process.reader = cms.EDFilter("SiStripDetVOffDummyPrinter")

process.p = cms.Path(process.stat+process.reader)

