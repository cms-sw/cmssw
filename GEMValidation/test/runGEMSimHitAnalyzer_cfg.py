import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

# the analyzer configuration
process.load('RPCGEM.GEMValidation.gemSimHitAnalyzer_cfi')


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS161_V12::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:out_sim.root'
#        'file:output_SingleMuPt40.root'
    'file:/afs/cern.ch/cms/MUON/gem/SingleMuPt40Fwd/SingleMuPt40Fwd_20121205_FixedGeometry_SIM.root'
    )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("gem_sh_ana.test.root")
)

process.p = cms.Path(process.gemSimHitAnalyzer)

