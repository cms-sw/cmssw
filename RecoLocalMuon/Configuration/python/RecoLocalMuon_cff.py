import FWCore.ParameterSet.Config as cms

#------------------------------------ DT ------------------------------------------------
# 1D RecHits
from RecoLocalMuon.DTRecHit.dt1DRecHits_LinearDriftFromDB_cfi import *
# 2D Segments
from RecoLocalMuon.DTSegment.dt2DSegments_MTPatternReco2D_LinearDriftFromDB_cfi import *
# 4D Segments
from RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDB_cfi import *
# 4D segments with t0 correction
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
dtlocalrecoTask = cms.Task(dt1DRecHits,dt4DSegments,dt1DCosmicRecHits,dt4DCosmicSegments)
dtlocalreco = cms.Sequence(dtlocalrecoTask)
# DT sequence with the 2D segment reconstruction
dtlocalreco_with_2DSegmentsTask = cms.Task(dt1DRecHits,dt2DSegments,dt4DSegments,dt1DCosmicRecHits,dt2DCosmicSegments,dt4DCosmicSegments)
dtlocalreco_with_2DSegments = cms.Sequence(dtlocalreco_with_2DSegmentsTask)
# DT sequence with T0seg correction
# CSC sequence
csclocalrecoTask = cms.Task(csc2DRecHits,cscSegments)
csclocalreco = cms.Sequence(csclocalrecoTask)
# DT, CSC and RPC together
muonlocalreco_with_2DSegmentsTask = cms.Task(dtlocalreco_with_2DSegmentsTask,csclocalrecoTask,rpcRecHits)
muonlocalreco_with_2DSegments = cms.Sequence(muonlocalreco_with_2DSegmentsTask)
# DT, CSC and RPC together (correct sequence for the standard path)
muonlocalrecoTask = cms.Task(dtlocalrecoTask,csclocalrecoTask,rpcRecHits)
muonlocalreco = cms.Sequence(muonlocalrecoTask)

from RecoLocalMuon.GEMRecHit.gemLocalReco_cff import *
from RecoLocalMuon.GEMRecHit.me0LocalReco_cff import *

_run2_GEM_2017_muonlocalrecoTask = muonlocalrecoTask.copy()
_run2_GEM_2017_muonlocalrecoTask.add(gemLocalRecoTask)

_run3_muonlocalrecoTask = muonlocalrecoTask.copy()
_run3_muonlocalrecoTask.add(gemLocalRecoTask)

_phase2_muonlocalrecoTask = _run3_muonlocalrecoTask.copy()
_phase2_muonlocalrecoTask.add(me0LocalRecoTask)

_phase2_ge0_muonlocalrecoTask = _phase2_muonlocalrecoTask.copyAndExclude([me0LocalRecoTask])

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( muonlocalrecoTask , _run2_GEM_2017_muonlocalrecoTask )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( muonlocalrecoTask , _run3_muonlocalrecoTask )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( muonlocalrecoTask , _phase2_muonlocalrecoTask )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toReplaceWith( muonlocalrecoTask , _phase2_ge0_muonlocalrecoTask )
