import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "/store/relval/CMSSW_5_0_0_pre6/RelValZTT/GEN-SIM-RECO/START50_V5-v1/0202/A0D83694-4917-E111-85E1-0026189438F2.root"
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

