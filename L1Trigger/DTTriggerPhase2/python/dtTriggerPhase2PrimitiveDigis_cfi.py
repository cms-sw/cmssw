import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *
from L1Trigger.DTTriggerPhase2.HoughGrouping_cfi                  import HoughGrouping
from L1Trigger.DTTriggerPhase2.PseudoBayesGrouping_cfi            import PseudoBayesPattern

dtTriggerPhase2PrimitiveDigis = cms.EDProducer("DTTrigPhase2Prod",
                                               digiTag = cms.InputTag("CalibratedDigis"),
                                               trigger_with_sl = cms.int32(4),
                                               tanPhiTh = cms.double(1.), 
                                               tanPhiThw2max = cms.double(1.3), 
                                               tanPhiThw2min = cms.double(0.5), 
                                               tanPhiThw1max = cms.double(0.9), 
                                               tanPhiThw1min = cms.double(0.2), 
                                               tanPhiThw0 = cms.double(0.5), 
                                               chi2Th = cms.double(0.01), #in cm^2
                                               chi2corTh = cms.double(0.1), #in cm^2
                                               useBX_correlation = cms.bool(False),
                                               dT0_correlate_TP = cms.double(25.), 
                                               dBX_correlate_TP = cms.int32(0), 
                                               dTanPsi_correlate_TP = cms.double(99999.),
                                               clean_chi2_correlation = cms.bool(True),
                                               allow_confirmation = cms.bool(True),
                                               minx_match_2digis = cms.double(1.),
                                               scenario = cms.int32(0), #0 for mc, 1 for data, 2 for slice test
                                               df_extended = cms.int32(0), # DF: 0 for standard, 1 for extended, 2 for both 
                                               max_primitives = cms.int32(999),

                                               ttrig_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_ttrig.txt'),
                                               z_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_z.txt'),
                                               shift_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_x.txt'),
                                               shift_theta_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/theta_shift.txt'),
                                               global_coords_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/global_coord_perp_x_phi0.txt'),
                                               algo = cms.int32(0), # 0 = STD gr., 2 = Hough transform, 1 = PseudoBayes Approach

                                               minHits4Fit = cms.int32(3),
                                               splitPathPerSL = cms.bool(True),

                                               #debugging
                                               debug = cms.untracked.bool(False),
                                               dump  = cms.untracked.bool(False),

                                               #RPC
                                               rpcRecHits = cms.InputTag("rpcRecHits"),
                                               useRPC = cms.bool(False),
                                               bx_window = cms.int32(1), # will look for RPC cluster within a bunch crossing window of 'dt.BX +- bx_window' 
                                               phi_window = cms.double(50.), # will look for RPC cluster within a phi window of +- phi_window in arbitrary coordinates (plot the value we cut on in RPCIntergator to fine tune it)
                                               max_quality_to_overwrite_t0 = cms.int32(9), # will use RPC  to set 't0' for TP with quality < 'max_quality_to_overwrite_t0'
                                               storeAllRPCHits = cms.bool(False),
                                               activateBuffer  = cms.bool(False),
                                               superCelltimewidth = cms.double(400), # in nanoseconds
                                               superCellspacewidth = cms.int32(20), # in number of cells: IT MUST BE AN EVEN NUMBER
                                               )

dtTriggerPhase2PrimitiveDigis.HoughGrouping      = HoughGrouping
dtTriggerPhase2PrimitiveDigis.PseudoBayesPattern = PseudoBayesPattern
