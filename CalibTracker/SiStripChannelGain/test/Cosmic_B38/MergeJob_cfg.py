# The following comments couldn't be translated into the new config version:

# XXX_SKIPEVENT_XXX

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(62966),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(62966),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    ),
    suppressDebug = cms.untracked.vstring('TrackRefitter'),
    suppressInfo = cms.untracked.vstring('TrackRefitter'),
    suppressWarning = cms.untracked.vstring('TrackRefitter')
)

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V4P::All"
process.prefer("GlobalTag")

process.TrackRefitter.src = 'ctfWithMaterialTracksP5'
process.TrackRefitter.TrajectoryInEvent = True


process.SiStripCalib = cms.EDFilter("SiStripGainFromData",
    AlgoMode            = cms.string('WriteOnDB'),

    VInputFiles         = cms.vstring(
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0000.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0001.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0002.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0003.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0004.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0005.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0006.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0007.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0008.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0009.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0010.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0011.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0012.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0013.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0014.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0015.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0016.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0017.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0018.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0019.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0020.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0021.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0022.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0023.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0024.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0025.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0026.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0027.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0028.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0029.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0030.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0031.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0032.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0033.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0034.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0035.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0036.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0037.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0038.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0039.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0040.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0041.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0042.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0043.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0044.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0045.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0046.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0047.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0048.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0049.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0050.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0051.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0052.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0053.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0054.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0055.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0056.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0057.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0058.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0059.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0060.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0061.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0062.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0063.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0064.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0065.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0066.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0067.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0068.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0069.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0070.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0071.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0072.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0073.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0074.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0075.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0076.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0077.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0078.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0079.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0080.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0081.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0082.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0083.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0084.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0085.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0086.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0087.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0088.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0089.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0090.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0091.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0092.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0093.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0094.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0095.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0096.root.root', 
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0097.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0098.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0099.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0100.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0101.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0102.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0103.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0104.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0105.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0106.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0107.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0108.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0109.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0110.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0111.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0112.root.root',
        'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CMSSW_2_1_10/src/CalibTracker/SiStripChannelGain/test/Cosmic_B38/FARM/RootFiles/ALCA_0113.root.root'
    ),

    OutputHistos        = cms.string('SiStripCalib.root'),
    OutputGains         = cms.string('SiStripCalib.txt'),

    TrajToTrackProducer = cms.string('TrackRefitter'),
    TrajToTrackLabel    = cms.string(''),

#    OutputHistos        = cms.string('SiStripCalib.root'),

    minTrackMomentum    = cms.untracked.double(1.0),
    minNrEntries        = cms.untracked.uint32(50),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(9999999.0),
    maxNrStrips         = cms.untracked.uint32(8),

    FirstSetOfConstants = cms.untracked.bool(False),

    CalibrationLevel    = cms.untracked.int32(2),

    SinceAppendMode     = cms.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(True)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:SiStrip_ChannelGain_Cosmic_Craft.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStrip_Gain_Cosmic_Craft_cosmic')
    ))
)

process.p = cms.Path(process.SiStripCalib)

