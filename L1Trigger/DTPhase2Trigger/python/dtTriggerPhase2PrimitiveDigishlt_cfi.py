import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *

dtTriggerPhase2PrimitiveDigis = cms.EDProducer("DTTrigPhase2Prod",
                                               digiTag = cms.InputTag("CalibratedDigis"),
                                               trigger_with_sl = cms.untracked.int32(4),
                                               tanPhiTh = cms.untracked.double(1.),
                                               do_correlation = cms.untracked.bool(True),
                                               dT0_correlate_TP = cms.untracked.double(25.),
                                               minx_match_2digis = cms.untracked.double(2.1),
                                               p2_df = cms.untracked.bool(True),
                                               filter_primos = cms.untracked.bool(True),
                                               apply_txt_ttrig_bc0 = cms.untracked.bool(False),
                                               ttrig_filename = cms.untracked.string('/afs/cern.ch/user/c/carrillo/public/info/wire_rawId_ttrig.txt'),
                                               z_filename = cms.untracked.string('/afs/cern.ch/user/c/carrillo/public/info/wire_rawId_z.txt'),          
                                               shift_filename = cms.untracked.string('/afs/cern.ch/user/c/carrillo/public/info/wire_rawId_x.txt'),    
                                               #debugging
                                               debug = cms.untracked.bool(False),
                                               pinta = cms.untracked.bool(False),
                                               dt4DSegments = cms.InputTag('hltDt4DSegments'),
                                               min_phinhits_match_segment = cms.untracked.int32(8),
                                               min_dT0_match_segment = cms.untracked.double(12.5)
                                               )
