import FWCore.ParameterSet.Config as cms

def customizeHLTDtUnpacking(process):
  """Adapt the HLT to run the legacy DT unpacking
     for pre2018 data/MC workflows as the default 
     unpacking in master/GRun is set to uROS 
     unacker"""

  if 'hltMuonDTDigis' in process.__dict__ :
      process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
                                               useStandardFEDid = cms.bool( True ),
                                               maxFEDid = cms.untracked.int32( 779 ),
                                               inputLabel = cms.InputTag( "rawDataCollector" ),
                                               minFEDid = cms.untracked.int32( 770 ),
                                               dataType = cms.string( "DDU" ),
                                               readOutParameters = cms.PSet( localDAQ = cms.untracked.bool( False ),
                                                                             debug = cms.untracked.bool( False ),
                                                                             rosParameters = cms.PSet( localDAQ = cms.untracked.bool( False ),
                                                                                                       debug = cms.untracked.bool( False ),
                                                                                                       writeSC = cms.untracked.bool( True ),
                                                                                                       readDDUIDfromDDU = cms.untracked.bool( True ),
                                                                                                       readingDDU = cms.untracked.bool( True ),
                                                                                                       performDataIntegrityMonitor = cms.untracked.bool( False )
                                                                                                       ),
                                                                             performDataIntegrityMonitor = cms.untracked.bool( False )
                                                                             ),
                                               dqmOnly = cms.bool( False )
                                               )
      
  return process
