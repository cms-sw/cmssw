import FWCore.ParameterSet.Config as cms

# Magnetic Field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# Geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
# DT Geometry
from Geometry.DTGeometry.dtGeometry_cfi import *
# CSC Geometry
from Geometry.CSCGeometry.cscGeometry_cfi import *
# RPC Geometry
from Geometry.RPCGeometry.rpcGeometry_cfi import *
#------------------------------------ DT ------------------------------------------------
# 1D RecHits
from RecoLocalMuon.DTRecHit.dt1DRecHits_ParamDrift_cfi import *
#	include "RecoLocalMuon/DTRecHit/data/dt1DRecHits_LinearDrift.cfi"
#	include "RecoLocalMuon/DTRecHit/data/dt1DRecHits_LinearDriftFromDB.cfi"
# 2D Segments
from RecoLocalMuon.DTSegment.dt2DSegments_CombPatternReco2D_ParamDrift_cfi import *
#	include "RecoLocalMuon/DTSegment/data/dt2DSegments_CombPatternReco2D_LinearDrift.cfi"
#	include "RecoLocalMuon/DTSegment/data/dt2DSegments_CombPatternReco2D_LinearDriftFromDB.cfi"
# 4D Segments
from RecoLocalMuon.DTSegment.dt4DSegments_CombPatternReco4D_ParamDrift_cfi import *
#	include "RecoLocalMuon/DTSegment/data/dt4DSegments_CombPatternReco4D_LinearDrift.cfi"
#	include "RecoLocalMuon/DTSegment/data/dt4DSegments_CombPatternReco4D_LinearDriftFromDB.cfi"
#------------------------------------ CSC -----------------------------------------------
# 2D RecHit	
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi import *
# Segments
from RecoLocalMuon.CSCSegment.cscSegments_cfi import *
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
# CSC sequence
csclocalreco = cms.Sequence(csc2DRecHits*cscSegments)
# DT, CSC and RPC together
muonlocalreco_with_2DSegments = cms.Sequence(dtlocalreco_with_2DSegments+csclocalreco+rpcRecHits)
# DT, CSC and RPC together (correct sequence for the standard path)
muonlocalreco = cms.Sequence(dtlocalreco+csclocalreco+rpcRecHits)

