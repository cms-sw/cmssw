import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *
from L1Trigger.DTTriggerPhase2.HoughGrouping_cfi                  import HoughGrouping
from L1Trigger.DTTriggerPhase2.PseudoBayesGrouping_cfi            import PseudoBayesPattern

dtTriggerPhase2PrimitiveDigis = cms.EDProducer("DTTrigPhase2Prod",
                                               digiTag = cms.InputTag("CalibratedDigis"),
                                               geometry_tag = cms.untracked.string(""),
                                               trigger_with_sl = cms.untracked.int32(4),
                                               tanPhiTh = cms.untracked.double(1.), 
                                               tanPhiThw2max = cms.untracked.double(1.3), 
                                               tanPhiThw2min = cms.untracked.double(0.5), 
                                               tanPhiThw1max = cms.untracked.double(0.9), 
                                               tanPhiThw1min = cms.untracked.double(0.2), 
                                               tanPhiThw0 = cms.untracked.double(0.5), 
                                               chi2Th = cms.untracked.double(0.01), #in cm^2
                                               chi2corTh = cms.untracked.double(0.1), #in cm^2
                                               do_correlation = cms.bool(True),
                                               useBX_correlation = cms.untracked.bool(False),
                                               dT0_correlate_TP = cms.untracked.double(25.), 
                                               dBX_correlate_TP = cms.untracked.int32(0), 
                                               dTanPsi_correlate_TP = cms.untracked.double(99999.),
                                               clean_chi2_correlation = cms.untracked.bool(True),
                                               allow_confirmation = cms.untracked.bool(True),
                                               minx_match_2digis = cms.untracked.double(1.),
                                               scenario = cms.int32(0), #0 for mc, 1 for data, 2 for slice test
                                               filter_cousins = cms.untracked.bool(True),

                                               ttrig_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_ttrig.txt'),
                                               z_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_z.txt'),
                                               shift_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_x.txt'),
                                               shift_theta_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/theta_shift.txt'),
                                               global_coords_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/global_coord_perp_x_phi0.txt'),
                                               cmssw_for_global = cms.untracked.bool(False),
                                               algo = cms.int32(0), # 0 = STD gr., 2 = Hough transform, 1 = PseudoBayes Approach

                                               minHits4Fit = cms.untracked.int32(4),

                                               #debugging
                                               debug = cms.untracked.bool(False),
                                               dump  = cms.untracked.bool(False),
                                               print_prims = cms.untracked.bool(False),
                                               file_to_print = cms.untracked.string("debug.txt"),
                                               print_digis = cms.untracked.bool(False),
                                               digi_file_to_print = cms.untracked.string("digis_debug.txt"),

                                               #RPC
                                               rpcRecHits = cms.InputTag("rpcRecHits"),
                                               useRPC = cms.bool(False),
                                               bx_window = cms.untracked.int32(1), # will look for RPC cluster within a bunch crossing window of 'dt.BX +- bx_window' 
                                               phi_window = cms.untracked.double(50.), # will look for RPC cluster within a phi window of +- phi_window in arbitrary coordinates (plot the value we cut on in RPCIntergator to fine tune it)
                                               max_quality_to_overwrite_t0 = cms.untracked.int32(9), # will use RPC  to set 't0' for TP with quality < 'max_quality_to_overwrite_t0'
                                               storeAllRPCHits = cms.untracked.bool(False),
                                               activateBuffer  = cms.bool(False),
                                               superCelltimewidth = cms.double(400), # in nanoseconds
                                               superCellspacewidth = cms.int32(20), # in number of cells: IT MUST BE AN EVEN NUMBER
                                               )

dtTriggerPhase2PrimitiveDigis.HoughGrouping      = HoughGrouping
dtTriggerPhase2PrimitiveDigis.PseudoBayesPattern = PseudoBayesPattern
