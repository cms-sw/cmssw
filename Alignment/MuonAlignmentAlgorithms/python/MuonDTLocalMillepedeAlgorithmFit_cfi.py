import FWCore.ParameterSet.Config as cms

MuonDTLocalMillepedeAlgorithm = cms.PSet(
   algoName = cms.string('MuonDTLocalMillepedeAlgorithm'),
                  ntuplePath = cms.string("castor:/castor/cern.ch/user/s/scodella/InternalAlignment/"),
                  numberOfRootFiles = cms.int32(100),
                  ptMax = cms.double(99999.),            
                  ptMin = cms.double(20.),
                  numberOfSigmasX = cms.double(3.), 
                  numberOfSigmasDXDZ = cms.double(3.), 
                  numberOfSigmasY = cms.double(3.), 
                  numberOfSigmasDYDZ = cms.double(3.),                              
                  nPhihits = cms.double(8),                            
                  nThetahits = cms.double(4), 
                  workingMode = cms.int32(2),
                  nMtxSection = cms.int32(0)

)


