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
    )

# Input files: RelVal QCD 80-120 GeV, STARTUP conditions, 9000 events, from CMSSW_3_2_5 (replace with 33X when available!)
fileNames1.extend( [
    '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0000/F681AAB1-9AA6-DE11-ADF6-001D09F2523A.root',
    '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0000/EED11BA3-9FA6-DE11-BD00-001D09F2905B.root',
    '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0000/A0060696-71A7-DE11-B330-000423D6006E.root',
    '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0000/9C13D02A-9CA6-DE11-85A8-001D09F290CE.root',
    '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0000/32C70A52-9BA6-DE11-903E-001D09F2538E.root',
    '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0000/2098DAED-9DA6-DE11-B881-001D09F24637.root',
    ] );

fileNames2.extend( [
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_3_0_pre4/F681AAB1-9AA6-DE11-ADF6-001D09F2523A.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_3_0_pre4/EED11BA3-9FA6-DE11-BD00-001D09F2905B.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_3_0_pre4/A0060696-71A7-DE11-B330-000423D6006E.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_3_0_pre4/9C13D02A-9CA6-DE11-85A8-001D09F290CE.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_3_0_pre4/32C70A52-9BA6-DE11-903E-001D09F2538E.root',
    'file:/data2/bainbrid/data/RelValQCD_Pt_80_120/CMSSW_3_3_0_pre4/2098DAED-9DA6-DE11-B881-001D09F24637.root',
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
process.VectorialCorrector.VectorialCorrection  = cms.bool(True)
process.VectorialCorrector.UseResponseInVecCorr = cms.bool(True)

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



