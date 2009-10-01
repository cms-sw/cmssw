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

# -------------------- JPT from RECO --------------------

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# Default

process.JPTCorrectorIC5CaloDefault = process.JetPlusTrackZSPCorrectorIcone5.clone()
process.JPTCorrectorIC5CaloDefault.label = cms.string('JPTCorrectorIC5CaloDefault')
process.JPTCorrectorIC5CaloDefault.ElectronIds = cms.InputTag("eidTight")
process.JPTCorrectorIC5CaloDefault.Verbose = cms.bool(False)

process.JPTCorJetIC5CaloDefault = process.JetPlusTrackZSPCorJetIcone5.clone()
process.JPTCorJetIC5CaloDefault.correctors = cms.vstring('JPTCorrectorIC5CaloDefault')
process.JPTCorJetIC5CaloDefault.alias = cms.untracked.string('JPTCorJetIC5CaloDefault')

# None (no correction)

process.JPTCorrectorIC5CaloNone = process.JPTCorrectorIC5CaloDefault.clone()
process.JPTCorrectorIC5CaloNone.label = cms.string('JPTCorrectorIC5CaloNone')
process.JPTCorrectorIC5CaloNone.UseInConeTracks      = cms.bool(False)
process.JPTCorrectorIC5CaloNone.UseOutOfConeTracks   = cms.bool(False)
process.JPTCorrectorIC5CaloNone.UseOutOfVertexTracks = cms.bool(False)
process.JPTCorrectorIC5CaloNone.UseEfficiency        = cms.bool(False)
process.JPTCorrectorIC5CaloNone.UseMuons             = cms.bool(False)
process.JPTCorrectorIC5CaloNone.UseElectrons         = cms.bool(False)
process.JPTCorrectorIC5CaloNone.VectorialCorrection  = cms.bool(False)
process.JPTCorrectorIC5CaloNone.UseResponseInVecCorr = cms.bool(False)

process.JPTCorJetIC5CaloNone = process.JPTCorJetIC5CaloDefault.clone()
process.JPTCorJetIC5CaloNone.correctors = cms.vstring('JPTCorrectorIC5CaloNone')
process.JPTCorJetIC5CaloNone.alias = cms.untracked.string('JPTCorJetIC5CaloNone')

# + InCone (pions only)

process.JPTCorrectorIC5CaloInCone = process.JPTCorrectorIC5CaloNone.clone()
process.JPTCorrectorIC5CaloInCone.label = cms.string('JPTCorrectorIC5CaloInCone')
process.JPTCorrectorIC5CaloInCone.UseInConeTracks = cms.bool(True)

process.JPTCorJetIC5CaloInCone = process.JPTCorJetIC5CaloNone.clone()
process.JPTCorJetIC5CaloInCone.correctors = cms.vstring('JPTCorrectorIC5CaloInCone')
process.JPTCorJetIC5CaloInCone.alias = cms.untracked.string('JPTCorJetIC5CaloInCone')

# + OutOfCone (pions only)

process.JPTCorrectorIC5CaloOutOfCone = process.JPTCorrectorIC5CaloInCone.clone()
process.JPTCorrectorIC5CaloOutOfCone.label = cms.string('JPTCorrectorIC5CaloOutOfCone')
process.JPTCorrectorIC5CaloOutOfCone.UseOutOfConeTracks = cms.bool(True)

process.JPTCorJetIC5CaloOutOfCone = process.JPTCorJetIC5CaloInCone.clone()
process.JPTCorJetIC5CaloOutOfCone.correctors = cms.vstring('JPTCorrectorIC5CaloOutOfCone')
process.JPTCorJetIC5CaloOutOfCone.alias = cms.untracked.string('JPTCorJetIC5CaloOutOfCone')

# + OutOfVertex (pions only)

process.JPTCorrectorIC5CaloOutOfVertex = process.JPTCorrectorIC5CaloOutOfCone.clone()
process.JPTCorrectorIC5CaloOutOfVertex.label = cms.string('JPTCorrectorIC5CaloOutOfVertex')
process.JPTCorrectorIC5CaloOutOfVertex.UseOutOfVertexTracks = cms.bool(True)

process.JPTCorJetIC5CaloOutOfVertex = process.JPTCorJetIC5CaloOutOfCone.clone()
process.JPTCorJetIC5CaloOutOfVertex.correctors = cms.vstring('JPTCorrectorIC5CaloOutOfVertex')
process.JPTCorJetIC5CaloOutOfVertex.alias = cms.untracked.string('JPTCorJetIC5CaloOutOfVertex')

# + PionEff

process.JPTCorrectorIC5CaloPionEff = process.JPTCorrectorIC5CaloOutOfVertex.clone()
process.JPTCorrectorIC5CaloPionEff.label = cms.string('JPTCorrectorIC5CaloPionEff')
process.JPTCorrectorIC5CaloPionEff.UseEfficiency = cms.bool(True)

process.JPTCorJetIC5CaloPionEff = process.JPTCorJetIC5CaloOutOfVertex.clone()
process.JPTCorJetIC5CaloPionEff.correctors = cms.vstring('JPTCorrectorIC5CaloPionEff')
process.JPTCorJetIC5CaloPionEff.alias = cms.untracked.string('JPTCorJetIC5CaloPionEff')

# + Muons

process.JPTCorrectorIC5CaloMuons = process.JPTCorrectorIC5CaloPionEff.clone()
process.JPTCorrectorIC5CaloMuons.label = cms.string('JPTCorrectorIC5CaloMuons')
process.JPTCorrectorIC5CaloMuons.UseMuons = cms.bool(True)

process.JPTCorJetIC5CaloMuons = process.JPTCorJetIC5CaloPionEff.clone()
process.JPTCorJetIC5CaloMuons.correctors = cms.vstring('JPTCorrectorIC5CaloMuons')
process.JPTCorJetIC5CaloMuons.alias = cms.untracked.string('JPTCorJetIC5CaloMuons')

# + Electrons

process.JPTCorrectorIC5CaloElectrons = process.JPTCorrectorIC5CaloMuons.clone()
process.JPTCorrectorIC5CaloElectrons.label = cms.string('JPTCorrectorIC5CaloElectrons')
process.JPTCorrectorIC5CaloElectrons.UseElectrons = cms.bool(True)

process.JPTCorJetIC5CaloElectrons = process.JPTCorJetIC5CaloMuons.clone()
process.JPTCorJetIC5CaloElectrons.correctors = cms.vstring('JPTCorrectorIC5CaloElectrons')
process.JPTCorJetIC5CaloElectrons.alias = cms.untracked.string('JPTCorJetIC5CaloElectrons')

# + VectorialCorrection

process.JPTCorrectorIC5CaloVecTracks = process.JPTCorrectorIC5CaloElectrons.clone()
process.JPTCorrectorIC5CaloVecTracks.label = cms.string('JPTCorrectorIC5CaloVecTracks')
process.JPTCorrectorIC5CaloVecTracks.VectorialCorrection = cms.bool(True)

process.JPTCorJetIC5CaloVecTracks = process.JPTCorJetIC5CaloElectrons.clone()
process.JPTCorJetIC5CaloVecTracks.correctors = cms.vstring('JPTCorrectorIC5CaloVecTracks')
process.JPTCorJetIC5CaloVecTracks.alias = cms.untracked.string('JPTCorJetIC5CaloVecTracks')

# + UseResponseInVecCorr

process.JPTCorrectorIC5CaloVecResponse = process.JPTCorrectorIC5CaloVecTracks.clone()
process.JPTCorrectorIC5CaloVecResponse.label = cms.string('JPTCorrectorIC5CaloVecResponse')
process.JPTCorrectorIC5CaloVecResponse.UseResponseInVecCorr = cms.bool(True)

process.JPTCorJetIC5CaloVecResponse = process.JPTCorJetIC5CaloVecTracks.clone()
process.JPTCorJetIC5CaloVecResponse.correctors = cms.vstring('JPTCorrectorIC5CaloVecResponse')
process.JPTCorJetIC5CaloVecResponse.alias = cms.untracked.string('JPTCorJetIC5CaloVecResponse')

# Sequences

process.JPTValidation = cms.Sequence(
    process.JPTCorJetIC5CaloNone *
    process.JPTCorJetIC5CaloInCone *
    process.JPTCorJetIC5CaloOutOfCone *
    process.JPTCorJetIC5CaloOutOfVertex *
    process.JPTCorJetIC5CaloPionEff *
    process.JPTCorJetIC5CaloMuons *
    process.JPTCorJetIC5CaloElectrons * 
    process.JPTCorJetIC5CaloVecTracks *
    process.JPTCorJetIC5CaloVecResponse 
    )

# -------------------- JPT from PAT --------------------

process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.load("JetMETCorrections.JetPlusTrack.PatJPTCorrections_cff")
process.uncorrectedLayer1JetsIC5.JetCollection = cms.InputTag("allLayer1Jets")

# -------------------- Paths --------------------

process.JTA = cms.Sequence(
    process.ZSPiterativeCone5JetTracksAssociatorAtVertex *
    process.ZSPiterativeCone5JetTracksAssociatorAtCaloFace *
    process.ZSPiterativeCone5JetExtender 
    )

process.p = cms.Path(
    process.ZSPJetCorJetIcone5 *      # ZSP corrections
    process.JTA *                     # Jet-tracks association
    process.JPTCorJetIC5CaloDefault * # Default JPT corrections from RECO
    process.JPTValidation *           # All flavours of JPT corrections from RECO
    process.patDefaultSequence *      # PAT sequence (slow!)
    process.PatJPTCorrectionsIC5      # JPT corrections from PAT
    )

process.o = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep recoGenJets_iterativeCone5GenJets_*_*',
    'keep recoGenJets_iterativeCone5GenJetsNoNuBSM_*_*',
    'keep recoCaloJets_iterativeCone5CaloJets_*_*',
    'keep patJets_cleanLayer1JetsIC5_*_*',
    'keep recoCaloJets_*_*_TEST',
    'keep patJets_*_*_TEST',
    )
    )

process.e = cms.EndPath( process.o )

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
    
    # allows to suppress output from specific modules 
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring(),
    
    )



