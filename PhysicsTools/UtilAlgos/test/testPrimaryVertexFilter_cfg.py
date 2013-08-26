import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "/store/relval/CMSSW_6_2_0_pre8/RelValZTT/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/9ABFC689-F9E0-E211-9DD2-02163E008EAE.root"
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.MessageLogger = cms.Service("MessageLogger")

## ---
## This is an example of the use of the EDFilterWrapper to exploit C++ selector classes as defined in the
## PhysicsTools/SelectorUtils package wrapping them into an EDFilter. You can find the implementation of
## the EDFilter in PhysicsTools/UtilAlgos/plugins/PrimaryVertexFilter.cc. You can find the EDfilterWrapper
## class in PhysicsTools/UtilAlgos/interface/EDFilterWrapper.h. The class that has been exploited here is
## the PVSelector class of the PhysicsTools/SelectorUtils package, which makes implicit use of the
## PVObjectSelector class in the same package.You cna find the input parameters there.
process.primaryVertexFilter = cms.EDFilter("PrimaryVertexFilter",
  pvSrc   = cms.InputTag("offlinePrimaryVertices"),
  minNdof = cms.double( 4 ),
  maxZ    = cms.double( 24.0 ),
  maxRho  = cms.double(2.0),
  NPV     = cms.int32(1)
)

process.p = cms.Path(process.primaryVertexFilter)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
