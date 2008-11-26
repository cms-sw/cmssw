import FWCore.ParameterSet.Config as cms

process = cms.Process("analyze")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

process.load("DQMServices.Core.DQM_cfg")

process.load("CalibTracker.SiStripLorentzAngle.SiStripLAProfileBooker_cfi")

process.load("CalibTracker.SiStripLorentzAngle.ALCARECOSiStripCalMinBias_cff")
 
process.load("CalibTracker.SiStripLorentzAngle.ALCARECOSiStripCalMinBias_Output_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    debug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('read', 
        'sistripLorentzAngle'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('debug_test_210')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(#'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/66CA0076-5060-DD11-8E3A-000423D99BF2.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/468FB321-2060-DD11-8359-000423D6CAF2.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/5C2AE0AE-7D60-DD11-948C-001617DF785A.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/66CA0076-5060-DD11-8E3A-000423D99BF2.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/A00242A0-4F60-DD11-9D54-000423D98EC4.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/C4355E5F-1D60-DD11-82A7-000423D98B6C.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/FE4027D7-5060-DD11-87CE-000423D94E1C.root',
                                      #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0024DAA6-2461-DD11-8E6D-001731AF68B3.root',
                                      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/007C02BF-2761-DD11-9E31-0018F3D09616.root',
                                      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/00C7F564-2561-DD11-8FAE-001731AF698D.root',
                                      'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/025870A4-2861-DD11-A5B5-001A92971B54.root')
)

process.trackrefitter = cms.EDFilter("TrackRefitter",
    #src = cms.InputTag("generalTracks"),
    src = cms.InputTag("ALCARECOSiStripCalMinBias"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string(''),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('ctf'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.recotracks = cms.Path(process.trackrefitter)
process.LorentzAngle = cms.Path(process.sistripLAProfile)
#process.ep = cms.EndPath(process.print)
#process.schedule = cms.Schedule(process.recotracks, process.LorentzAngle)
process.schedule = cms.Schedule(process.pathALCARECOSiStripCalMinBias, process.recotracks, process.LorentzAngle)

process.DQM.collectorHost = ''
process.sistripLAProfile.Tracks = 'trackrefitter'
process.sistripLAProfile.TIB_bin = 120
process.sistripLAProfile.TOB_bin = 120
process.sistripLAProfile.SUM_bin = 120
process.sistripLAProfile.Fit_Result = cms.bool(True)
process.sistripLAProfile.fileName = 'file:~/scratch0/CMSSW_2_1_6/src/CalibTracker/SiStripLorentzAngle/test/TEST_ALCARECO/histo_test_RelVal_210.root'
process.sistripLAProfile.treeName = 'file:~/scratch0/CMSSW_2_1_6/src/CalibTracker/SiStripLorentzAngle/test/TEST_ALCARECO/LATrees_test_RelVal_210.root'
process.sistripLAProfile.fitName = 'file:~/scratch0/CMSSW_2_1_6/src/CalibTracker/SiStripLorentzAngle/test/TEST_ALCARECO/fit_test_RelVal_210'


