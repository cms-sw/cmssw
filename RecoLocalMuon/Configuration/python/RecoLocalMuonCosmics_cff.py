import FWCore.ParameterSet.Config as cms

#------------------------------------ DT ------------------------------------------------
# 1D RecHits
from RecoLocalMuon.DTRecHit.dt1DRecHits_LinearDriftFromDB_CosmicData_cfi import *
# 2D Segments
from RecoLocalMuon.DTSegment.dt2DSegments_MTPatternReco2D_LinearDriftFromDB_CosmicData_cfi import *
# 4D Segments
from RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDB_CosmicData_cfi import *
# T0 seg correction
from RecoLocalMuon.DTSegment.dt4DSegments_ApplyT0Correction_cfi import *

#------------------------------------ CSC -----------------------------------------------
# 2D RecHit
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi import *
# Segments
from RecoLocalMuon.CSCSegment.cscSegments_cfi import *
from CalibMuon.CSCCalibration.CSCChannelMapper_cfi import *
from CalibMuon.CSCCalibration.CSCIndexer_cfi import *

#------------------------------------ RPC -----------------------------------------------
# 1D RecHits
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *

#----------------------------------------------------------------------------------------
# DT sequence for the standard reconstruction chain 
# The reconstruction of the 2D segments are not required for the 4D segments reconstruction, they are used
# only for debuging purpose and for specific studies
dtlocalreco = cms.Sequence(dt1DRecHits*dt4DSegments)
# DT sequence with the 2D segment reconstruction
dtlocalreco_with_2DSegments = cms.Sequence(dt1DRecHits*dt2DSegments*dt4DSegments)
# DT sequence with T0seg correction
dtlocalrecoT0Seg = cms.Sequence(dt1DRecHits*dt4DSegments*dt4DSegmentsT0Seg)
# CSC sequence
csclocalreco = cms.Sequence(csc2DRecHits*cscSegments)
# DT, CSC and RPC together
muonlocalreco_with_2DSegments = cms.Sequence(dtlocalreco_with_2DSegments+csclocalreco+rpcRecHits)
# DT, CSC and RPC together (correct sequence for the standard path)
muonlocalreco = cms.Sequence(dtlocalreco+csclocalreco+rpcRecHits)
# DT, CSC and RPC together (with t0seg correction for DTs)
muonlocalrecoT0Seg = cms.Sequence(dtlocalrecoT0Seg+csclocalreco+rpcRecHits)
# all sequences to be used for GR
muonLocalRecoGR = cms.Sequence(muonlocalreco+muonlocalrecoT0Seg)

from RecoLocalMuon.GEMRecHit.gemLocalReco_cff import *
from RecoLocalMuon.GEMRecHit.me0LocalReco_cff import *

_run2_GEM_2017_muonlocalreco = muonlocalreco.copy()
_run2_GEM_2017_muonlocalreco += gemLocalReco

_run3_muonlocalreco = muonlocalreco.copy()
_run3_muonlocalreco += gemLocalReco

_phase2_muonlocalreco = _run3_muonlocalreco.copy()
_phase2_muonlocalreco += me0LocalReco

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( muonlocalreco , _run2_GEM_2017_muonlocalreco )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( muonlocalreco , _run3_muonlocalreco )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( muonlocalreco , _phase2_muonlocalreco )


# RPC New Readout Validation
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
_rpc_NewReadoutVal_muonlocalreco_with_2DSegments = muonlocalreco_with_2DSegments.copy()
_rpc_NewReadoutVal_muonlocalreco = muonlocalreco.copy()
_rpc_NewReadoutVal_muonlocalrecoT0Seg = muonlocalrecoT0Seg.copy()
_rpc_NewReadoutVal_muonlocalreco_with_2DSegments += rpcNewRecHits
_rpc_NewReadoutVal_muonlocalreco += rpcNewRecHits
_rpc_NewReadoutVal_muonlocalrecoT0Seg += rpcNewRecHits
stage2L1Trigger_2017.toReplaceWith(muonlocalreco_with_2DSegments, _rpc_NewReadoutVal_muonlocalreco_with_2DSegments)
stage2L1Trigger_2017.toReplaceWith(muonlocalreco, _rpc_NewReadoutVal_muonlocalreco)
stage2L1Trigger_2017.toReplaceWith(muonlocalrecoT0Seg, _rpc_NewReadoutVal_muonlocalrecoT0Seg)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(muonlocalreco_with_2DSegments, muonlocalreco_with_2DSegments.copyAndExclude([rpcNewRecHits]))
fastSim.toReplaceWith(muonlocalreco, muonlocalreco.copyAndExclude([rpcNewRecHits]))
fastSim.toReplaceWith(muonlocalrecoT0Seg, muonlocalrecoT0Seg.copyAndExclude([rpcNewRecHits]))

