import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')  ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(140058),
    lastValue  = cms.uint64(140058)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V7::All'
process.prefer("GlobalTag")

process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
process.SiStripCalib.InputFiles          = cms.vstring(
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_16_140058_1.root',   #Size=859Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_17_140059_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_43_140124_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_45_140126_1.root',   #Size=1781Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_59_140158_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_60_140159_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_1_140160_1.root',   #Size=1064Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_24_140331_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_30_140352_1.root',   #Size=1691Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_32_140361_1.root',   #Size=808Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_33_140362_1.root',   #Size=1392Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_42_140379_1.root',   #Size=1115Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_52_140382_1.root',   #Size=1786Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_46_140383_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_47_140385_1.root',   #Size=1172Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_49_140387_1.root',   #Size=828Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_50_140388_1.root',   #Size=680Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_56_140399_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_57_140401_1.root',   #Size=1596Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_118_140401_1.root',   #Size=1596Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_20_141874_1.root',   #Size=615Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_21_141876_1.root',   #Size=542Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_24_141880_1.root',   #Size=1149Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_25_141881_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_26_141882_1.root',   #Size=1332Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_42_142187_1.root',   #Size=1616Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_44_142189_1.root',   #Size=650Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_45_142191_1.root',   #Size=586Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_67_142265_1.root',   #Size=604Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_75_142305_1.root',   #Size=873Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_78_142311_1.root',   #Size=1687Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_79_142312_1.root',   #Size=781Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_5_142414_1.root',   #Size=591Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_9_142419_1.root',   #Size=611Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_10_142420_1.root',   #Size=587Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_11_142422_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_25_142524_1.root',   #Size=1140Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_26_142525_1.root',   #Size=519Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_27_142528_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_33_142558_1.root',   #Size=882Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_47_142662_1.root',   #Size=756Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_48_142663_1.root',   #Size=640Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_6_142928_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_8_142933_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_3_142953_1.root',   #Size=1132Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_4_142954_1.root',   #Size=985Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_12_142970_1.root',   #Size=686Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_13_142971_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_15_143005_1.root',   #Size=977Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_17_143007_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_24_143181_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_25_143187_1.root',   #Size=810Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_11_143320_1.root',   #Size=564Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_37_143322_1.root',   #Size=588Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_14_143323_1.root',   #Size=1603Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_41_143326_1.root',   #Size=612Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_43_143328_1.root',   #Size=825Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_16_143657_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_50_143727_1.root',   #Size=1450Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_30_143827_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_2_143833_1.root',   #Size=1791Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_28_143953_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_29_143954_1.root',   #Size=832Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_56_143957_1.root',   #Size=594Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_36_143961_1.root',   #Size=1117Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_13_143962_1.root',   #Size=1182Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_8_144011_1.root',   #Size=1888Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_18_144086_1.root',   #Size=1345Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_12_144089_1.root',   #Size=2047Mo
          'rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_51_144112_1.root',   #Size=2047Mo
)
process.SiStripCalib.FirstSetOfConstants = cms.untracked.bool(False)



process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:Gains_Sqlite.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('IdealGainTag')
    ))
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Gains_Tree.root')  
)

process.p = cms.Path(process.SiStripCalib)
