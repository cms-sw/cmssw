import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer",sourceSeed = cms.untracked.string("$$"))

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All')

fileNames1 = cms.untracked.vstring()
fileNames2 = cms.untracked.vstring()
process.source = cms.Source(
    "PoolSource",
    fileNames = fileNames1,
    #skipEvents = cms.untracked.uint32(223),
    )

# Input files: RelVal QCD 80-120 GeV, STARTUP conditions, 9000 events, from CMSSW_3_2_5 (replace with 33X when available!)
fileNames1.extend( [
    '/store/relval/CMSSW_3_2_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V4-v1/0011/2AA47C1E-828E-DE11-B3C5-001D09F34488.root',
    '/store/relval/CMSSW_3_2_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V4-v1/0010/DA6FF61D-3D8E-DE11-938A-003048D37538.root',
    '/store/relval/CMSSW_3_2_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V4-v1/0010/82CF8666-398E-DE11-8F3B-000423D94A20.root',
    '/store/relval/CMSSW_3_2_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V4-v1/0010/6255E85C-3F8E-DE11-B46A-000423D6B48C.root',
    '/store/relval/CMSSW_3_2_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V4-v1/0010/3453CD32-418E-DE11-87D2-003048D2C020.root',
    '/store/relval/CMSSW_3_2_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V4-v1/0010/2EC02533-3B8E-DE11-BE85-003048D37514.root',
    ] );
fileNames2.extend( [
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_2_5/2AA47C1E-828E-DE11-B3C5-001D09F34488.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_2_5/DA6FF61D-3D8E-DE11-938A-003048D37538.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_2_5/82CF8666-398E-DE11-8F3B-000423D94A20.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_2_5/6255E85C-3F8E-DE11-B46A-000423D6B48C.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_2_5/3453CD32-418E-DE11-87D2-003048D2C020.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_2_5/2EC02533-3B8E-DE11-BE85-003048D37514.root',
    ] );

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# ZSP and JPT cff files

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# JPT scalar correction with IC5

process.ScalarCorrector = process.JetPlusTrackZSPCorrectorIcone5.clone()
process.ScalarCorrector.label = cms.string('ScalarCorrector')
process.ScalarCorrector.ElectronIds = cms.InputTag("eidTight")
process.ScalarCorrector.Verbose = cms.bool(False)
process.ScalarCorrector.VectorialCorrection = cms.bool(False)
process.ScalarCorrector.JetDirFromTracks    = cms.bool(False)

process.ScalarCorrection = process.JetPlusTrackZSPCorJetIcone5.clone()
process.ScalarCorrection.correctors = cms.vstring('ScalarCorrector')
process.ScalarCorrection.alias = cms.untracked.string('ScalarCorrection')

# JPT vectorial correction with IC5

process.VectorialCorrector = process.JetPlusTrackZSPCorrectorIcone5.clone()
process.VectorialCorrector.label = cms.string('VectorialCorrector')
process.VectorialCorrector.ElectronIds = cms.InputTag("eidTight")
process.VectorialCorrector.Verbose = cms.bool(False)
process.VectorialCorrector.VectorialCorrection = cms.bool(True)
process.VectorialCorrector.JetDirFromTracks    = cms.bool(True)

process.VectorialCorrection = process.JetPlusTrackZSPCorJetIcone5.clone()
process.VectorialCorrection.correctors = cms.vstring('VectorialCorrector')
process.VectorialCorrection.alias = cms.untracked.string('VectorialCorrection')

# Paths

process.p1 = cms.Path(
    process.ZSPJetCorJetIcone5 *
    process.ZSPiterativeCone5JetTracksAssociatorAtVertex *
    process.ZSPiterativeCone5JetTracksAssociatorAtCaloFace *
    process.ZSPiterativeCone5JetExtender *
    process.ScalarCorrection *
    process.VectorialCorrection 
    )

# EndPath

process.o = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep recoGenJets_iterativeCone5GenJets_*_*',
    'keep recoGenJets_iterativeCone5GenJetsNoNuBSM_*_*',
    'keep recoCaloJets_iterativeCone5CaloJets_*_*',
    'keep recoCaloJets_*_*_TEST',
    'keep *_sort_*_TEST',
    )
    )

process.e = cms.EndPath( process.o )

# Misc

process.MessageLogger = cms.Service(
    "MessageLogger",
    
    debug = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(False),
    ),

    info = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(False),
    ),

    warning = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(False),
    ),

    error = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(False),
    ),
    
    cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(False),
    ),

    destinations = cms.untracked.vstring(
    'debug', 
    'info', 
    'warning', 
    'error',
    'cerr'
    ),
    
    #@@ comment to suppress debug statements!
    #debugModules = cms.untracked.vstring('*'),
    
    )



