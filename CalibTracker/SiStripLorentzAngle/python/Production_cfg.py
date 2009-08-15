import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.SiStripLorentzAngle.LA_Tree_RECO_cff')
process.GlobalTag.globaltag = 'GR09_P_V1'

process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTree.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )
