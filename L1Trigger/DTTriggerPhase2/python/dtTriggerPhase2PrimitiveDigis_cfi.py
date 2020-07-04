import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *
from L1Trigger.DTTriggerPhase2.HoughGrouping_cfi                  import HoughGrouping
from L1Trigger.DTTriggerPhase2.PseudoBayesGrouping_cfi            import PseudoBayesPattern

dtTriggerPhase2PrimitiveDigis = cms.EDProducer("DTTrigPhase2Prod",
                                               digiTag = cms.InputTag("CalibratedDigis"),
                                               trigger_with_sl = cms.untracked.int32(4),
                                               tanPhiTh = cms.untracked.double(1.),
                                               chi2Th = cms.untracked.double(0.01), #in cm^2
                                               chi2corTh = cms.untracked.double(0.1), #in cm^2
                                               do_correlation = cms.bool(True),
                                               useBX_correlation = cms.untracked.bool(False),
                                               dT0_correlate_TP = cms.untracked.double(25.), 
                                               dBX_correlate_TP = cms.untracked.int32(0), 
                                               dTanPsi_correlate_TP = cms.untracked.double(99999.),
                                               clean_chi2_correlation = cms.untracked.bool(True),
                                               allow_confirmation = cms.untracked.bool(True),
                                               use_LSB = cms.untracked.bool(True),
                                               tanPsi_precision = cms.untracked.double(1./4096.),
                                               x_precision = cms.untracked.double(1./160.),
                                               minx_match_2digis = cms.untracked.double(1.),
                                               scenario = cms.int32(0), #0 for mc, 1 for data, 2 for slice test
                                               filter_cousins = cms.untracked.bool(True),

                                               ttrig_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_ttrig.txt'),
                                               z_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_z.txt'),
                                               shift_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/wire_rawId_x.txt'),
                                               algo = cms.int32(0), # 0 = STD gr., 2 = Hough transform, 1 = PseudoBayes Approach

                                               minHits4Fit = cms.untracked.int32(4),

                                               #debugging
                                               debug = cms.untracked.bool(False),
                                               dump  = cms.untracked.bool(False),
                                               
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
