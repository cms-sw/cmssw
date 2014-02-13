import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_0_0_pre11/RelValProdTTbar/GEN-SIM-RECO/START70_V4-v1/00000/0EA82C3C-646A-E311-9CB3-0025905A6070.root'
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.MessageLogger = cms.Service("MessageLogger")

## ---
## This is an example of the use of the EDFilterObjectWrapper to exploit C++ selector classes as defined
## in the PhysicsTools/SelectorUtils package wrapping them into an EDFilter that in addition creates a
## collection of objects passing the selection criteria. You can find the implementation of the EDProducer
## in PhysicsTools/UtilAlgos/plugins/PrimaryVertexFilter.cc. You can find the EDfilterWrapper class in
## PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h. The class that has been exploited here is
## the PVObjectSelector class of the PhysicsTools/SelectorUtils package.You cna find the input parameters
## there.
process.primaryVertexObjectFilter = cms.EDFilter("PrimaryVertexObjectFilter",
  src   = cms.InputTag("offlinePrimaryVertices"),
  filterParams = cms.PSet(
    minNdof = cms.double( 4 ),
    maxZ    = cms.double( 2 ),
    maxRho  = cms.double(0.2)
  )
)

process.p = cms.Path(process.primaryVertexObjectFilter)

